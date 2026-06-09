"""
Tests for BlackTelperion.analyse.deshadow — out-of-core (streamed) shadow
detection / removal.

The streaming path must match the in-memory analyse.deshadow result. Data is
synthetic and written to a temporary ENVI file via io.save, then streamed back.

Orientation: GDAL is [row, col] = [y, x]; BlackImage is [x, y, band]. Parity
assertions transpose the streamed 2-D mask before comparing to the in-memory
[x, y] mask.
"""

import os
import tempfile
import unittest

import numpy as np

from BlackTelperion import BlackImage, io
from BlackTelperion.io.images import loadWithGDAL
from BlackTelperion.analyse.deshadow.core import (
    shadow_mask, deshadow_image, _integrate_strip, _trapezoid_weights,
)

try:
    from osgeo import gdal, osr
    _HAS_GDAL = True
except Exception:
    _HAS_GDAL = False


def _load_raw(path):
    """Load a written file faithfully (no 0->NaN mask_zero convention)."""
    return loadWithGDAL(path, mask_zero=False).data

# Reuse the synthetic scene definition from the in-memory suite.
from tests.test_deshadow_core import (
    make_scene, NX, NY, NB, WAV, WSPAN, BASE, SHADOW, NAN_BLOCK, SPECKLE,
)

from BlackTelperion.analyse.deshadow import (
    shadow_mask_large, deshadow_large_image,
)


def _to_xy(gdal_2d):
    """[row, col] = [y, x]  ->  [x, y]."""
    return np.asarray(gdal_2d).T


def _save_scene(d, name="scene.dat", scene=None):
    scene = make_scene() if scene is None else scene
    path = os.path.join(d, name)
    io.save(path, scene)
    return path, scene


@unittest.skipUnless(_HAS_GDAL, "GDAL required for the out-of-core path")
class TestStreaming(unittest.TestCase):

    # -- T1 ----------------------------------------------------------------
    def test_streaming_integral_uses_shared_helper(self):
        weights = _trapezoid_weights(WAV)
        strip = np.full((4, NX, NB), 0.3, dtype=np.float32)   # (rows, cols, bands)
        ref = _integrate_strip(strip, weights)
        self.assertTrue(np.allclose(ref, 0.3 * WSPAN, atol=1.0))

    # -- T2: orientation contract -----------------------------------------
    def test_gdal_orientation_contract(self):
        scene = make_scene()
        scene.data[7, 3, :] = 0.123456                         # distinctive (x=7, y=3)
        with tempfile.TemporaryDirectory() as d:
            path, _ = _save_scene(d, scene=scene)
            ds = gdal.Open(path)
            band0 = ds.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)[0]  # [y, x]
            ds = None
            self.assertAlmostEqual(float(band0[3, 7]), 0.123456, places=4)

    # -- T3: detection parity, no cleanup ---------------------------------
    def test_detect_parity_no_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            path, scene = _save_scene(d)
            out = shadow_mask_large(path, os.path.join(d, "mask.dat"),
                                           threshold=30.0)
            streamed = _load_raw(out)[..., 0]                  # [x, y]
            in_mem = shadow_mask(scene, threshold=30.0).data[..., 0]
            np.testing.assert_array_equal(streamed, in_mem)

    # -- T4: detection parity with morphology -----------------------------
    def test_detect_parity_with_morphology(self):
        with tempfile.TemporaryDirectory() as d:
            path, scene = _save_scene(d)
            out = shadow_mask_large(path, os.path.join(d, "mask.dat"),
                                           threshold=30.0,
                                           min_shadow_size=5, min_gap_size=5)
            streamed = _load_raw(out)[..., 0]
            in_mem = shadow_mask(scene, threshold=30.0,
                                 min_shadow_size=5, min_gap_size=5).data[..., 0]
            np.testing.assert_array_equal(streamed, in_mem)

    # -- T5: strip-size invariance ----------------------------------------
    def test_strip_size_invariance(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = _save_scene(d)
            a = _load_raw(shadow_mask_large(
                path, os.path.join(d, "a.dat"), threshold=30.0, strip_rows=7))
            b = _load_raw(shadow_mask_large(
                path, os.path.join(d, "b.dat"), threshold=30.0, strip_rows=NY))
            np.testing.assert_array_equal(a, b)

    # -- T6: apply parity --------------------------------------------------
    def test_apply_parity(self):
        with tempfile.TemporaryDirectory() as d:
            path, scene = _save_scene(d)
            out = deshadow_large_image(path, os.path.join(d, "cube.dat"),
                                           threshold=30.0, min_shadow_size=5,
                                           flag=np.nan)
            streamed = _load_raw(out)                           # [x, y, band]
            in_mem = deshadow_image(scene, threshold=30.0, min_shadow_size=5,
                                    flag=np.nan).data
            np.testing.assert_allclose(streamed, in_mem, equal_nan=True,
                                       rtol=1e-5, atol=1e-5)

    # -- T7: custom flag ---------------------------------------------------
    def test_apply_custom_flag(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = _save_scene(d)
            out = deshadow_large_image(path, os.path.join(d, "cube.dat"),
                                           threshold=30.0, min_shadow_size=5,
                                           flag=0.0)
            cube = _load_raw(out)
            # shadow block was set to 0.0 across all bands
            self.assertTrue(np.all(cube[SHADOW] == 0.0))

    # -- T8: nodata border -------------------------------------------------
    def test_nodata_border_is_shadow(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = _save_scene(d, scene=make_scene(nan_block=True))
            out = shadow_mask_large(path, os.path.join(d, "mask.dat"),
                                           threshold=30.0)
            streamed = _load_raw(out)[..., 0]                  # [x, y]
            self.assertTrue(np.all(streamed[NAN_BLOCK] == 1.0))

    # -- T9: geo metadata preserved ---------------------------------------
    def test_geo_metadata_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = _save_scene(d)
            src = gdal.Open(path)
            src_gt, src_pj = src.GetGeoTransform(), src.GetProjection()
            src = None
            for fn, kw in ((shadow_mask_large, {}),
                           (deshadow_large_image, {"min_shadow_size": 5})):
                out = fn(path, os.path.join(d, fn.__name__ + ".dat"),
                         threshold=30.0, **kw)
                ods = gdal.Open(out)
                self.assertEqual(ods.GetGeoTransform(), src_gt)
                a, b = osr.SpatialReference(), osr.SpatialReference()
                a.ImportFromWkt(ods.GetProjection())
                b.ImportFromWkt(src_pj)
                self.assertTrue(a.IsSame(b))
                ods = None

    # -- T10: validation + single-strip edge ------------------------------
    def test_threshold_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = _save_scene(d)
            for bad in (0.0, 100.0, -1.0):
                with self.assertRaises(ValueError):
                    shadow_mask_large(path, os.path.join(d, "m.dat"),
                                             threshold=bad)

    def test_single_strip_whole_image(self):
        with tempfile.TemporaryDirectory() as d:
            path, scene = _save_scene(d)
            out = shadow_mask_large(path, os.path.join(d, "m.dat"),
                                           threshold=30.0, strip_rows=10 * NY)
            streamed = _load_raw(out)[..., 0]
            in_mem = shadow_mask(scene, threshold=30.0).data[..., 0]
            np.testing.assert_array_equal(streamed, in_mem)

    # -- T10b: integer-source guard ---------------------------------------
    def test_integer_source_with_nan_flag_raises(self):
        with tempfile.TemporaryDirectory() as d:
            # build an int16 cube and save it
            scene = make_scene()
            scene.data = (scene.data * 1000).astype(np.int16)
            path = os.path.join(d, "intcube.dat")
            io.save(path, scene)
            with self.assertRaises(ValueError):
                deshadow_large_image(path, os.path.join(d, "out.dat"),
                                         threshold=30.0, flag=np.nan)

    # -- T11: round-trip through io ---------------------------------------
    def test_mask_roundtrip_band_name(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = _save_scene(d)
            out = shadow_mask_large(path, os.path.join(d, "mask.dat"),
                                           threshold=30.0)
            mask = io.load(out)
            self.assertIsInstance(mask, BlackImage)
            self.assertEqual(mask.band_count(), 1)
            self.assertEqual(list(mask.get_band_names()), ["shadow_mask"])

    def test_cube_roundtrip_band_count(self):
        with tempfile.TemporaryDirectory() as d:
            path, scene = _save_scene(d)
            out = deshadow_large_image(path, os.path.join(d, "cube.dat"),
                                           threshold=30.0, min_shadow_size=5)
            cube = io.load(out)
            self.assertEqual(cube.band_count(), scene.band_count())


if __name__ == "__main__":
    unittest.main()
