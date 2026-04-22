"""
Shadow detection for hyperspectral imagery via spectral integral thresholding.

Shadowed pixels reflect less energy across the entire spectrum, so their
spectral integral (area under the reflectance curve) is systematically lower
than sunlit pixels. A threshold expressed as a percentage of the maximum
observed integral separates the two classes.

To handle large images without exhausting RAM, the integral computation is
performed in row-strips whose size is controlled by the ``strip_rows``
parameter. Only one strip is held in memory at a time; the integral array
(one float32 per pixel) accumulates the results and is the only full-image
allocation during that phase.

Typical usage
-------------
    from BlackTelperion.filter.shadows import shadow_mask

    mask = shadow_mask(image, threshold=30.0, min_shadow_size=500, min_gap_size=500)

The returned object is a single-band BlackData instance whose values are:
    1.0 → shadow pixel
    0.0 → sunlit pixel
"""

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shadow_mask(data, threshold: float = 30.0,
                min_shadow_size: int = 0,
                min_gap_size: int = 0,
                strip_rows: int = 256):
    """
    Build a binary shadow mask from a hyperspectral dataset.

    The spectrum of each pixel is integrated using the trapezoidal rule. A
    pixel is classified as shadow when its integral falls below *threshold*
    percent of the maximum integral observed across the image. NaN bands within
    a pixel are zeroed before integration so they contribute nothing to the
    sum; pixels whose entire spectrum is NaN produce a NaN integral and are
    classified as shadow.

    To keep peak memory proportional to one strip rather than the whole image,
    the integration is performed in horizontal row-strips of *strip_rows* rows.
    The only full-image allocation is the float32 integral array (~4 bytes per
    pixel) and the final uint8 mask (~1 byte per pixel).

    An optional morphological clean-up step removes spatial noise:
      - isolated shadow blobs smaller than *min_shadow_size* pixels are
        reclassified as sunlit.
      - isolated sunlit gaps smaller than *min_gap_size* pixels that are
        enclosed by shadow are filled in as shadow.

    Args:
        data:
            A BlackData (or BlackImage) instance whose last dimension indexes
            spectral bands.
        threshold (float):
            Percentage of the maximum spectral integral below which a pixel is
            labelled as shadow. Must be in the range (0, 100). Default is 30.
        min_shadow_size (int):
            Connected shadow regions with fewer pixels than this value are
            removed (set to sunlit). 0 disables this step. Default is 0.
        min_gap_size (int):
            Connected sunlit regions enclosed by shadow with fewer pixels than
            this value are filled in as shadow. 0 disables this step.
            Default is 0.
        strip_rows (int):
            Number of rows processed at once. Reduce this value if memory is
            still insufficient; increase it for faster processing when RAM
            allows. Default is 256.

    Returns:
        A new BlackData instance of the same concrete type as *data* containing
        a single band with dtype ``float32``:
            1.0 → shadow
            0.0 → sunlit

        The output header carries the band name ``"shadow_mask"``.

    Raises:
        ValueError: if *threshold* is not in the range (0, 100).
    """
    if not (0.0 < threshold < 100.0):
        raise ValueError(
            f"threshold must be in the open interval (0, 100), got {threshold}."
        )

    wavelengths = data.get_wavelengths()
    weights = _trapezoid_weights(wavelengths)

    integrals = _spectral_integrals_stripped(data.data, weights, strip_rows)
    shadow = _apply_threshold(integrals, threshold)

    if min_shadow_size > 0 or min_gap_size > 0:
        shadow = _remove_small_regions(
            shadow,
            min_shadow_size=min_shadow_size,
            min_gap_size=min_gap_size,
        )

    out = data.copy(data=False)
    out.header.drop_all_bands()
    out.data = shadow[..., np.newaxis]
    out.set_band_names(["shadow_mask"])
    out.set_wavelengths(None)
    out.push_to_header()

    return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _trapezoid_weights(wavelengths: np.ndarray) -> np.ndarray:
    """
    Precompute per-band trapezoidal weights from a wavelength axis.

    The trapezoidal rule integral(f, w) = f @ weights, where:
        weights[0]    = dw[0] / 2
        weights[k]    = (dw[k-1] + dw[k]) / 2   for 0 < k < n-1
        weights[-1]   = dw[-1] / 2

    Args:
        wavelengths: 1-D array of length ``bands``.

    Returns:
        Float32 weight vector of length ``bands``.
    """
    dw = np.diff(wavelengths.astype(np.float32))
    weights = np.empty(len(wavelengths), dtype=np.float32)
    weights[0]    = dw[0] / 2.0
    weights[1:-1] = (dw[:-1] + dw[1:]) / 2.0
    weights[-1]   = dw[-1] / 2.0
    return weights


def _integrate_strip(strip: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Integrate one strip of rows using the precomputed trapezoidal weights.

    Args:
        strip:   Float32 array of shape (rows, cols, bands) or (n_points, bands).
        weights: Float32 array of length ``bands``.

    Returns:
        Float32 array of shape (rows, cols) or (n_points,).
    """
    spatial_shape = strip.shape[:-1]
    n_bands = strip.shape[-1]
    flat = strip.reshape(-1, n_bands).astype(np.float32)

    # Zero NaN bands so they contribute nothing; track all-NaN pixels.
    all_nan = ~np.isfinite(flat).any(axis=-1)
    safe = np.where(np.isfinite(flat), flat, 0.0)

    result = safe @ weights
    result[all_nan] = np.nan

    return result.reshape(spatial_shape)


def _spectral_integrals_stripped(raw: np.ndarray,
                                  weights: np.ndarray,
                                  strip_rows: int) -> np.ndarray:
    """
    Compute spectral integrals for the full image, one strip at a time.

    Each strip is converted to float32, integrated, and discarded before the
    next strip is loaded. The only full-image allocation is the output array.

    Args:
        raw:        Data array of shape (x, y, bands) as stored in BlackImage,
                    or (n_points, bands) for point clouds.
        weights:    Float32 trapezoidal weight vector of length ``bands``.
        strip_rows: Number of rows per strip.

    Returns:
        Float32 array of shape (x, y) or (n_points,).
    """
    if raw.ndim == 2:
        return _integrate_strip(raw, weights)

    n_rows = raw.shape[0]
    integrals = np.empty(raw.shape[:-1], dtype=np.float32)

    for start in range(0, n_rows, strip_rows):
        end = min(start + strip_rows, n_rows)
        integrals[start:end] = _integrate_strip(raw[start:end], weights)

    return integrals


def _apply_threshold(integrals: np.ndarray, threshold: float) -> np.ndarray:
    """
    Classify pixels as shadow or sunlit based on an integral threshold.

    The cutoff is *threshold* percent of the maximum finite integral. NaN
    pixels fall below any finite cutoff and are therefore labelled as shadow.

    Args:
        integrals: Float array of shape (...,).
        threshold: Percentage in (0, 100).

    Returns:
        Float32 array of shape (...,) with values 1.0 (shadow) or 0.0 (sunlit).
    """
    max_integral = float(np.nanmax(integrals))
    cutoff = max_integral * (threshold / 100.0)
    return np.where(integrals < cutoff, 1.0, 0.0).astype(np.float32)


def _remove_small_regions(shadow: np.ndarray,
                           min_shadow_size: int,
                           min_gap_size: int) -> np.ndarray:
    """
    Morphological clean-up of the binary shadow mask.

    Args:
        shadow:          Float32 array of shape (...,) with values 0.0 or 1.0.
        min_shadow_size: Shadow blobs smaller than this are set to 0.0 (sunlit).
        min_gap_size:    Sunlit regions smaller than this are set to 1.0 (shadow).

    Returns:
        Cleaned float32 array of the same shape as *shadow*.
    """
    result = shadow.copy()

    if min_shadow_size > 0:
        labeled, _ = ndimage.label((result == 1.0).astype(np.uint8))
        sizes = np.bincount(labeled.ravel())
        sizes[0] = min_shadow_size          # never remove background label
        small_mask = np.isin(labeled, np.where(sizes < min_shadow_size)[0])
        result[small_mask] = 0.0

    if min_gap_size > 0:
        labeled, _ = ndimage.label((result == 0.0).astype(np.uint8))
        sizes = np.bincount(labeled.ravel())
        sizes[0] = min_gap_size             # never fill background label
        small_mask = np.isin(labeled, np.where(sizes < min_gap_size)[0])
        result[small_mask] = 1.0

    return result