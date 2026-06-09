"""
Shadow detection and removal for images too large to fit in memory.

:func:`shadow_mask_large` and :func:`deshadow_large_image` are the streamed,
file-in/file-out counterparts of
:func:`~BlackTelperion.analyse.deshadow.core.shadow_mask` and
:func:`~BlackTelperion.analyse.deshadow.core.deshadow_image`. The cube is read in
horizontal row-strips via GDAL windows, reusing the integral / threshold /
cleanup primitives from :mod:`BlackTelperion.analyse.deshadow.core` (no detection
math is re-implemented here). See :mod:`BlackTelperion.analyse.deshadow` for the
method.

Because the threshold is a percentage of the *global* maximum integral, the
single-band integral image is materialised in full (one float32 per pixel — many
times smaller than the cube) before thresholding; the cube itself is never fully
loaded.

.. note::
   GDAL ``ReadAsArray()`` is ``[band, row, col] = [band, y, x]`` while
   ``BlackImage`` is ``[x, y, band]`` (``loadWithGDAL`` transposes). The 2-D
   integral and mask are kept in GDAL ``[row, col] = [y, x]`` orientation
   throughout; only the single-band mask, which fits in RAM, is transposed back
   to build a BlackImage.
"""

import os
import re
import shutil

import numpy as np

from BlackTelperion import BlackImage, io
from BlackTelperion.analyse.deshadow.core import (
    _trapezoid_weights,
    _integrate_strip,
    _apply_threshold,
    _remove_small_regions,
)

# ---------------------------------------------------------------------------
# GDAL helpers
# ---------------------------------------------------------------------------

# GDAL integer data-type constants -> numpy dtypes (for the integer-source guard).
_GDAL_INT_TYPES = {"Byte", "Int8", "UInt16", "Int16", "UInt32", "Int32",
                   "UInt64", "Int64", "CInt16", "CInt32"}


def _gdal():
    try:
        import osgeo.gdal as gdal
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        return gdal
    except Exception:
        raise ImportError(
            "GDAL (osgeo) is required for the large-image shadow functions "
            "(BlackTelperion.analyse.deshadow.large_image)."
        )


def _source_wavelengths(input_path, n_bands):
    """Read wavelengths from the source ENVI header; fall back to band indices."""
    try:
        hdr_path, _ = io.matchHeader(input_path)
        if hdr_path is not None:
            header = io.loadHeader(hdr_path)
            if header.has_wavelengths():
                return np.asarray(header.get_wavelengths(), dtype=np.float64)
    except Exception:
        pass
    return np.arange(n_bands, dtype=np.float64)


# ---------------------------------------------------------------------------
# Pass 1 — streamed integral, threshold, cleanup
# ---------------------------------------------------------------------------

def _build_integral(ds, weights, strip_rows):
    """Stream row-strips and return the full ``[row, col]`` float32 integral."""
    n_cols, n_rows = ds.RasterXSize, ds.RasterYSize
    integrals = np.empty((n_rows, n_cols), dtype=np.float32)
    for r0 in range(0, n_rows, strip_rows):
        n = min(strip_rows, n_rows - r0)
        strip = ds.ReadAsArray(0, r0, n_cols, n)               # (bands, n, cols)
        strip = np.transpose(strip, (1, 2, 0)).astype(np.float32)  # (n, cols, bands)
        integrals[r0:r0 + n] = _integrate_strip(strip, weights)
    return integrals


def _detect_2d(input_path, threshold, min_shadow_size, min_gap_size, strip_rows):
    """Open the source and return ``(ds, shadow)`` with ``shadow`` in ``[row, col]``.

    The returned GDAL dataset stays open so callers can reuse it for pass 2.
    """
    if not (0.0 < threshold < 100.0):
        raise ValueError(
            "threshold must be in the open interval (0, 100), got %s." % threshold
        )

    gdal = _gdal()
    ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError("GDAL could not open %s" % input_path)

    weights = _trapezoid_weights(_source_wavelengths(input_path, ds.RasterCount))
    integrals = _build_integral(ds, weights, strip_rows)

    shadow = _apply_threshold(integrals, threshold)
    if min_shadow_size > 0 or min_gap_size > 0:
        shadow = _remove_small_regions(shadow, min_shadow_size, min_gap_size)
    return ds, shadow


def _out_hdr(output_path):
    """Return the .hdr path that saveWithGDAL / the ENVI driver wrote for output_path."""
    base = os.path.splitext(output_path)[0]
    for c in (base + ".hdr", output_path + ".hdr"):
        if os.path.exists(c):
            return c
    return base + ".hdr"


def _set_hdr_field(hdr_path, field, value):
    """Set/replace a single scalar field in an ENVI .hdr text file."""
    with open(hdr_path, "r") as f:
        text = f.read()
    pat = r"(?im)^\s*%s\s*=.*$" % re.escape(field)
    line = "%s = %s" % (field, value)
    text = re.sub(pat, line, text) if re.search(pat, text) else text.rstrip() + "\n" + line + "\n"
    with open(hdr_path, "w") as f:
        f.write(text)


def _save_mask(shadow, output_path, geotransform, projection):
    """Build a single-band BlackImage mask (fits RAM) and save via :mod:`io`.

    The on-disk mask is float32 with 1.0=shadow, 0.0=sunlit. Note that the default
    ``io.load`` (``mask_zero=True``) maps sunlit 0.0 -> NaN; use
    ``loadWithGDAL(path, mask_zero=False)`` to read the raw 0/1 values.
    """
    mask = BlackImage(shadow.T[..., np.newaxis].astype(np.float32))  # [x, y, 1]
    mask.affine = list(geotransform)
    if projection:
        mask.set_projection(projection)
    mask.set_band_names(["shadow_mask"])
    mask.push_to_header()
    io.save(output_path, mask)
    # saveWithGDAL's GA_Update step clobbers band names in the .hdr; restore it.
    _set_hdr_field(_out_hdr(output_path), "band names", "{shadow_mask}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shadow_mask_large(input_path, output_path, threshold=30.0,
                      min_shadow_size=0, min_gap_size=0, strip_rows=256):
    """
    Shadow detection for a cube too large to fit in memory (large-image variant
    of :func:`~BlackTelperion.analyse.deshadow.core.shadow_mask`).

    The cube is read in horizontal row-strips via GDAL windows; each pixel's
    spectrum is integrated (trapezoid rule) and thresholded at *threshold*
    percent of the global maximum integral. All-NaN (no-data) pixels are
    classified as shadow.

    Args:
        input_path (str): path to the source cube (any GDAL-readable raster).
        output_path (str): path for the single-band float32 shadow mask
            (1.0=shadow, 0.0=sunlit), written as ENVI BSQ with the source
            projection and geotransform.
        threshold (float): percentage of the maximum integral below which a
            pixel is shadow. Must be in (0, 100). Default 30.
        min_shadow_size (int): remove connected shadow regions smaller than this.
            0 disables. Default 0.
        min_gap_size (int): fill connected sunlit regions smaller than this.
            0 disables. Default 0.
        strip_rows (int): number of rows read per GDAL window. Default 256.

    Returns:
        str: *output_path*.
    """
    ds, shadow = _detect_2d(input_path, threshold,
                            min_shadow_size, min_gap_size, strip_rows)
    gt, pj = ds.GetGeoTransform(), ds.GetProjection()
    ds = None
    _save_mask(shadow, output_path, gt, pj)
    return output_path


def deshadow_large_image(input_path, output_path, threshold=30.0,
                         min_shadow_size=0, min_gap_size=0, strip_rows=256,
                         flag=np.nan, mask_path=None):
    """
    Remove shadows from a cube too large to fit in memory (large-image variant
    of :func:`~BlackTelperion.analyse.deshadow.core.deshadow_image`).

    Detects shadows (as :func:`shadow_mask_large`) then streams the cube a second
    time, setting every band of each shadow pixel to *flag*, and writes a full
    output cube preserving the source dtype, projection and geotransform.

    Args:
        input_path (str): path to the source cube.
        output_path (str): path for the masked output cube (ENVI BSQ).
        threshold, min_shadow_size, min_gap_size, strip_rows: see
            :func:`shadow_mask_large`.
        flag (float): value written to shadow pixels across all bands. Default
            ``np.nan`` (treated as no-data throughout BlackTelperion).
        mask_path (str or None): if given, also persist the intermediate shadow
            mask there (single band). Default None.

    Returns:
        str: *output_path*.

    Raises:
        ValueError: if the source cube has an integer dtype but *flag* cannot be
            represented in it (e.g. ``np.nan``). No silent upcast is performed —
            check whether the cube should be decompressed / converted to float
            reflectance before deshadowing.
    """
    gdal = _gdal()
    ds, shadow = _detect_2d(input_path, threshold,
                            min_shadow_size, min_gap_size, strip_rows)

    n_cols, n_rows, n_bands = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
    src_gdal_dtype = ds.GetRasterBand(1).DataType
    type_name = gdal.GetDataTypeName(src_gdal_dtype)

    # Integer-source guard: never silently upcast.
    if type_name in _GDAL_INT_TYPES and not np.isfinite(flag):
        ds = None
        raise ValueError(
            "Source cube %s has integer dtype (%s) but flag=%r cannot be stored "
            "in it. Check whether the cube should be decompressed / converted to "
            "float reflectance before deshadowing." % (input_path, type_name, flag)
        )

    gt, pj = ds.GetGeoTransform(), ds.GetProjection()

    if mask_path is not None:
        _save_mask(shadow, mask_path, gt, pj)

    # Create the output cube once; copy geo verbatim (offsets place each strip).
    out = gdal.GetDriverByName("ENVI").Create(
        output_path, n_cols, n_rows, n_bands, src_gdal_dtype,
        options=["INTERLEAVE=BSQ"],
    )
    if out is None:
        ds = None
        raise RuntimeError("GDAL could not create output cube %s" % output_path)
    out.SetGeoTransform(gt)
    if pj:
        out.SetProjection(pj)

    # Pass 2 — stream strips, flag shadow pixels across all bands, write windowed.
    for r0 in range(0, n_rows, strip_rows):
        n = min(strip_rows, n_rows - r0)
        block = ds.ReadAsArray(0, r0, n_cols, n)        # (bands, n, cols)
        m = shadow[r0:r0 + n] > 0                        # (n, cols)
        block[:, m] = flag
        out.WriteArray(block, 0, r0)                     # xoff=0, yoff=r0

    out = None
    ds = None

    _propagate_header(input_path, output_path)
    return output_path


def _propagate_header(input_path, output_path):
    """Copy the source ENVI header to the output (preserving wavelengths / map
    info / coordinate system), forcing BSQ interleave so :func:`io.load` round-trips.

    The output cube keeps the source dtype, dims and geo, so the source header is
    valid for it apart from the interleave field.
    """
    try:
        src_hdr, _ = io.matchHeader(input_path)
    except Exception:
        src_hdr = None
    if not src_hdr or not os.path.exists(src_hdr):
        return

    out_hdr = _out_hdr(output_path)
    shutil.copyfile(src_hdr, out_hdr)          # preserves wavelengths / map info / coord sys
    _set_hdr_field(out_hdr, "interleave", "bsq")
