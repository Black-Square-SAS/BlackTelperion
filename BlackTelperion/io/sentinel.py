"""
Load Sentinel-2 multispectral data as a hyperspectral cube (BlackImage).

Sentinel-2 distributes each spectral band as a separate JP2 file. This module
stacks the available spectral bands for a given resolution into a single
BlackImage, assigning correct central wavelengths, FWHM, band names, and
georeferencing derived from the JP2 metadata.

JP2 files are read directly via GDAL, bypassing the generic ``load()``
dispatcher (which requires a sidecar ``.hdr`` file and does not support
bare ``.jp2`` paths).

Typical usage::

    from BlackTelperion.io.sentinel import loadSentinel2
    image = loadSentinel2("/path/to/R60m/")
    image = loadSentinel2("/path/to/S2A.SAFE/", resolution=20)
"""

import os
import re
import warnings
import numpy as np
from BlackTelperion import BlackImage, BlackHeader

# ---------------------------------------------------------------------------
# Sentinel-2 spectral band definitions
# Source: ESA Sentinel-2 spectral response functions (central wavelength / FWHM)
# ---------------------------------------------------------------------------
_S2_BANDS = {
    #  name:  (central_wavelength_nm, fwhm_nm)
    "B01": (442.7,   21),
    "B02": (492.4,   66),
    "B03": (559.8,   36),
    "B04": (664.6,   31),
    "B05": (704.1,   15),
    "B06": (740.5,   15),
    "B07": (782.8,   20),
    "B08": (833.0,  106),
    "B8A": (864.7,   21),
    "B09": (945.1,   20),
    "B11": (1613.7,  91),
    "B12": (2202.4, 175),
}

# Regex matching Sentinel-2 spectral band filenames only, e.g.:
#   T18NYL_20260213T151721_B8A_60m.jp2
_BAND_PATTERN = re.compile(r".*_(B[0-9]{2}|B8A)_.*\.jp2$", re.IGNORECASE)


def _find_band_files(directory, resolution):
    """
    Scan *directory* for JP2 files belonging to the requested *resolution*.

    Args:
        directory (str): directory containing Sentinel-2 JP2 files.
        resolution (int): target resolution in metres (10, 20, or 60).

    Returns:
        dict: mapping band name (e.g. ``'B04'``) to absolute file path.
    """
    res_token = "_%dm." % resolution   # e.g. "_60m."
    found = {}
    for fname in os.listdir(directory):
        if not _BAND_PATTERN.match(fname):
            continue
        if res_token.lower() not in fname.lower():
            continue
        match = re.search(r"_(B[0-9]{2}|B8A)_", fname, re.IGNORECASE)
        if match:
            found[match.group(1).upper()] = os.path.join(directory, fname)
    return found


def _resolve_directory(path, resolution):
    """
    Resolve the directory that directly contains JP2 files for *resolution*.

    Walks down from *path*, handling .SAFE roots, GRANULE directories, and
    IMG_DATA sub-folders transparently.

    Args:
        path (str): root path supplied by the user.
        resolution (int): target resolution in metres.

    Returns:
        str: first directory found that contains matching JP2 files, or
             *path* itself as a fallback.
    """
    token = "_%dm." % resolution
    for root, _, files in os.walk(path):
        if any(token.lower() in f.lower() and f.lower().endswith(".jp2")
               for f in files):
            return root
    return path


def _load_jp2_gdal(fpath):
    """
    Read a single-band JP2 file with GDAL.

    Args:
        fpath (str): path to the JP2 file.

    Returns:
        tuple: ``(data, geo_meta)`` where *data* is a float32 ndarray of shape
               ``(rows, cols, 1)`` and *geo_meta* is a dict with keys
               ``'transform'`` and ``'projection'``, or an empty dict if no
               georeferencing is available.
    """
    from osgeo import gdal
    ds = gdal.Open(fpath, gdal.GA_ReadOnly)
    assert ds is not None, "GDAL could not open: %s" % fpath

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.int16)   # (rows, cols)

    # Mask no-data — Sentinel-2 uses 0 as fill value
    nodata = band.GetNoDataValue()
    if nodata is not None:
        data[data == nodata] = 0

    geo = {}
    gt = ds.GetGeoTransform()    # (x_origin, px_w, 0, y_origin, 0, -px_h)
    proj = ds.GetProjectionRef()
    if gt and any(v != 0 for v in gt):
        geo = {"transform": gt, "projection": proj}

    ds = None                                          # close dataset
    return data[:, :, np.newaxis], geo                 # (rows, cols, 1)


def _geo_to_envi_map_info(transform, projection):
    """
    Convert a GDAL GeoTransform and WKT projection to ENVI header strings.

    Args:
        transform (tuple): GDAL GeoTransform —
            ``(x_origin, px_w, 0, y_origin, 0, -px_h)``.
        projection (str): WKT projection string from GDAL.

    Returns:
        tuple: ``(map_info_str, crs_str)`` ready to store in a BlackHeader.
    """
    try:
        from osgeo import osr
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection)
        epsg = srs.GetAuthorityCode(None) or "unknown"
        proj_name = (srs.GetAttrValue("PROJCS")
                     or srs.GetAttrValue("GEOGCS")
                     or "unknown")
    except Exception:
        epsg, proj_name = "unknown", "unknown"

    x_origin, px_w, _, y_origin, _, px_h = transform
    map_info = (
        "{%s, 1, 1, %.4f, %.4f, %.4f, %.4f, %s, units=Meters}"
        % (proj_name, x_origin, y_origin, abs(px_w), abs(px_h), epsg)
    )
    return map_info, projection


def loadSentinel2(path, resolution=60):
    """
    Load Sentinel-2 spectral bands into a single BlackImage cube.

    Individual JP2 band files are discovered automatically within *path* (or
    its sub-directories when a .SAFE / GRANULE root is provided). Only spectral
    bands are loaded; auxiliary products (SCL, TCI, AOT, WVP, etc.) are
    ignored. Bands are sorted by central wavelength before stacking.

    JP2 files are read directly with GDAL (``osgeo.gdal``), which must be
    installed.

    Args:
        path (str): path to a directory containing Sentinel-2 JP2 files, or
            to the root of a .SAFE product folder.
        resolution (int or str): target spatial resolution in metres — one of
            ``10``, ``20``, or ``60``. Pass ``'auto'`` (default) to select the
            finest resolution for which files are present.

    Returns:
        BlackImage: stacked hyperspectral cube with wavelengths (nm), FWHM, band
        names, and ENVI-compatible georeferencing populated in the header.

    Raises:
        AssertionError: if *path* does not exist, *resolution* is invalid, or
            no spectral band files are found.
        ImportError: if GDAL (``osgeo``) is not installed.

    Example::

        img = loadSentinel2("/data/S2A.SAFE/")
        img = loadSentinel2("/data/S2A.SAFE/GRANULE/.../IMG_DATA/R20m/", resolution=20)
    """
    try:
        from osgeo import gdal  # noqa: F401
    except ImportError:
        raise ImportError(
            "GDAL is required to load Sentinel-2 JP2 files. "
            "Install it with:  conda install gdal  or  pip install gdal"
        )

    assert os.path.exists(path), "Error: path does not exist — %s" % path
    assert resolution in (10, 20, 60, "auto"), (
        "Error: resolution must be 10, 20, 60, or 'auto', got %r." % resolution
    )

    # Locate the directory containing JP2s; try finest resolution first
    candidates = [10, 20, 60] if resolution == "auto" else [int(resolution)]
    band_files = {}
    for res in candidates:
        directory = _resolve_directory(path, res)
        band_files = _find_band_files(directory, res)
        if band_files:
            break

    assert band_files, (
        "Error: no Sentinel-2 spectral band files found in '%s' "
        "for resolution %s m." % (path, resolution)
    )

    # Drop bands absent from the spectral lookup table (warn, don't crash)
    unknown = set(band_files) - set(_S2_BANDS)
    if unknown:
        warnings.warn("Skipping unrecognised band(s): %s" % ", ".join(sorted(unknown)))
    known = {b: p for b, p in band_files.items() if b in _S2_BANDS}

    # Sort by central wavelength
    sorted_bands = sorted(known.items(), key=lambda kv: _S2_BANDS[kv[0]][0])
    band_names  = [b for b, _ in sorted_bands]
    wavelengths = np.array([_S2_BANDS[b][0] for b in band_names])
    fwhm        = np.array([_S2_BANDS[b][1] for b in band_names])

    # Load each JP2 directly via GDAL
    arrays = []
    geo_meta = {}
    for _, fpath in sorted_bands:
        data, geo = _load_jp2_gdal(fpath)
        if not geo_meta and geo:
            geo_meta = geo
        arrays.append(data)                          # (rows, cols, 1)

    # Stack → (rows, cols, n_bands), then transpose to BlackTelperion convention
    stacked = np.concatenate(arrays, axis=-1)        # (rows, cols, n_bands)
    stacked = np.transpose(stacked, (1, 0, 2))       # → (cols, rows, n_bands)

    # Build header
    header = BlackHeader()
    header["file type"]         = "ENVI Standard"
    header["wavelength units"]  = "nm"
    header["data ignore value"] = str(0)
    header.set_wavelengths(wavelengths)
    header.set_fwhm(fwhm)
    header.set_band_names(band_names)

    if geo_meta:
        map_info, crs_str = _geo_to_envi_map_info(
            geo_meta["transform"], geo_meta["projection"]
        )
        header["map info"] = map_info
        header["coordinate system string"] = crs_str

    out = BlackImage(stacked, header=header)
    out.push_to_header()
    return out