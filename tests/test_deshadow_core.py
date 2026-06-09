"""
Tests for BlackTelperion.analyse.deshadow (in-memory) — integral-based shadow
detection (implemented in commit de5f1c7).

All data is synthetic and built in-code, so the suite has no dependency on the
(gitignored) test_data/ directory.

Synthetic scene (BlackImage, data[x, y, band]):
  - flat reflectance 0.5 everywhere               -> integral 0.5 * (1700-900) = 400
  - SHADOW block  x[5:20], y[5:20]  reflectance*0.2 (=0.1) -> integral 80
  - SPECKLE       single pixel (45, 5)  reflectance*0.1     -> isolated shadow, integral 40
  - NAN block     x[30:40], y[40:50] = NaN across all bands -> all-NaN pixels
"""

import unittest

import numpy as np

from BlackTelperion import BlackImage
from BlackTelperion.analyse.deshadow.core import shadow_mask, deshadow_image

NX, NY, NB = 50, 60, 40
WAV = np.linspace(900.0, 1700.0, NB)
WSPAN = float(WAV[-1] - WAV[0])                 # 800
BASE = 0.5

SHADOW = (slice(5, 20), slice(5, 20))           # x, y
NAN_BLOCK = (slice(30, 40), slice(40, 50))      # x, y
SPECKLE = (45, 5)                               # isolated shadow pixel


def make_scene(nbands=NB, nan_block=False):
    wav = np.linspace(900.0, 1700.0, nbands)
    data = np.full((NX, NY, nbands), BASE, dtype=np.float32)
    data[SHADOW] *= 0.2
    data[SPECKLE[0], SPECKLE[1], :] *= 0.1
    if nan_block:
        data[NAN_BLOCK] = np.nan
    return BlackImage(data, wav=wav)


class TestOutputStructure(unittest.TestCase):

    def test_structure_and_band_name(self):
        out = shadow_mask(make_scene(), threshold=30.0)
        self.assertIsInstance(out, BlackImage)
        self.assertEqual(out.band_count(), 1)
        self.assertEqual(out.data.shape[:2], (NX, NY))
        self.assertEqual(out.data.dtype, np.float32)
        self.assertEqual(list(out.get_band_names()), ["shadow_mask"])

    def test_values_are_binary(self):
        out = shadow_mask(make_scene(), threshold=30.0)
        vals = np.unique(out.data[np.isfinite(out.data)])
        self.assertTrue(set(vals.tolist()).issubset({0.0, 1.0}))


class TestDetection(unittest.TestCase):

    def test_recovers_shadow_region(self):
        m = shadow_mask(make_scene(), threshold=30.0).data[..., 0]
        self.assertGreater((m[SHADOW] == 1.0).mean(), 0.9)   # shadow block flagged
        self.assertEqual(m[0, 0], 0.0)                        # lit corner sunlit

    def test_threshold_monotonic(self):
        scene = make_scene()
        counts = [int((shadow_mask(scene, threshold=t).data[..., 0] == 1.0).sum())
                  for t in (10.0, 30.0, 60.0)]
        self.assertLessEqual(counts[0], counts[1])
        self.assertLessEqual(counts[1], counts[2])
        # shadow block (integral 80) is below 30% cutoff (120) but above 10% cutoff (40)
        self.assertLess(counts[0], counts[1])

    def test_threshold_validation(self):
        scene = make_scene()
        for bad in (0.0, 100.0, -5.0, 150.0):
            with self.assertRaises(ValueError):
                shadow_mask(scene, threshold=bad)


class TestStripInvariance(unittest.TestCase):

    def test_strip_rows_does_not_change_result(self):
        scene = make_scene()
        whole = shadow_mask(scene, threshold=30.0, strip_rows=NX).data
        strips = shadow_mask(scene, threshold=30.0, strip_rows=7).data
        np.testing.assert_array_equal(whole, strips)


class TestMorphology(unittest.TestCase):

    def test_min_shadow_size_removes_speckle(self):
        scene = make_scene()
        kept = shadow_mask(scene, threshold=30.0).data[..., 0]              # default min_shadow_size=0
        removed = shadow_mask(scene, threshold=30.0, min_shadow_size=5).data[..., 0]
        self.assertEqual(kept[SPECKLE[0], SPECKLE[1]], 1.0)                 # speckle present
        self.assertEqual(removed[SPECKLE[0], SPECKLE[1]], 0.0)             # speckle removed
        self.assertGreater((removed[SHADOW] == 1.0).mean(), 0.9)           # big block survives

    def test_min_gap_size_fills_small_holes(self):
        # punch a 1-pixel sunlit hole inside the shadow block
        scene = make_scene()
        scene.data[12, 12, :] = BASE          # bright hole inside shadow region
        no_fill = shadow_mask(scene, threshold=30.0).data[..., 0]
        filled = shadow_mask(scene, threshold=30.0, min_gap_size=5).data[..., 0]
        self.assertEqual(no_fill[12, 12], 0.0)
        self.assertEqual(filled[12, 12], 1.0)


class TestNanHandling(unittest.TestCase):
    """Docstring states all-NaN pixels are classified as shadow (1.0)."""

    def test_all_nan_pixels_classified_as_shadow(self):
        # _apply_threshold now classifies non-finite (all-NaN/nodata) integrals as
        # shadow, matching the documented behaviour (fixed in the deshadow merge).
        m = shadow_mask(make_scene(nan_block=True), threshold=30.0).data[..., 0]
        self.assertTrue(np.all(m[NAN_BLOCK] == 1.0))


class TestDeshadowImage(unittest.TestCase):
    """deshadow_image bridges shadow_mask -> BlackImage.mask (in-memory apply)."""

    def test_flags_shadow_pixels_nan_across_bands(self):
        scene = make_scene()
        out = deshadow_image(scene, threshold=30.0, min_shadow_size=5)
        self.assertIsInstance(out, BlackImage)
        self.assertEqual(out.band_count(), scene.band_count())
        # every band of the shadow block is NaN
        self.assertTrue(np.all(np.isnan(out.data[SHADOW])))
        # lit corner is untouched
        self.assertTrue(np.allclose(out.data[0, 0, :], BASE))

    def test_not_inplace_by_default(self):
        scene = make_scene()
        deshadow_image(scene, threshold=30.0, min_shadow_size=5)
        self.assertFalse(np.any(np.isnan(scene.data[SHADOW])))   # original untouched

    def test_inplace(self):
        scene = make_scene()
        deshadow_image(scene, threshold=30.0, min_shadow_size=5, inplace=True)
        self.assertTrue(np.all(np.isnan(scene.data[SHADOW])))

    def test_custom_fill_value(self):
        scene = make_scene()
        out = deshadow_image(scene, threshold=30.0, min_shadow_size=5, flag=0.0)
        self.assertTrue(np.all(out.data[SHADOW] == 0.0))


class TestTrapezoidMath(unittest.TestCase):
    """Validate the integral helpers directly."""

    def test_weights_sum_to_span(self):
        from BlackTelperion.analyse.deshadow.core import _trapezoid_weights
        w = _trapezoid_weights(WAV)
        self.assertAlmostEqual(float(w.sum()), WSPAN, delta=1e-2)

    def test_integrate_flat_and_linear(self):
        from BlackTelperion.analyse.deshadow.core import _trapezoid_weights, _integrate_strip
        w = _trapezoid_weights(WAV)
        flat = np.full((4, 4, NB), 0.2, dtype=np.float32)
        i1 = _integrate_strip(flat, w)
        self.assertAlmostEqual(float(i1[2, 2]), 0.2 * WSPAN, delta=1.0)
        i2 = _integrate_strip(flat * 3.0, w)
        self.assertAlmostEqual(float(i2[2, 2]) / float(i1[2, 2]), 3.0, places=4)


if __name__ == "__main__":
    unittest.main()
