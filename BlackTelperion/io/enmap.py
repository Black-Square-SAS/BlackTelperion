"""
Load EnMAP hyperspectral data as a BlackImage.

EnMAP L2A products are distributed as a directory containing a BSQ spectral
image with an ENVI ``.HDR`` sidecar and a ``-METADATA.XML`` file that holds
per-band wavelengths, FWHM, and quality metadata.

The image is loaded as **raw int16** so that the existing
``BlackData.decompress()`` workflow can be used to convert to float32
reflectance.

Typical usage::

    from BlackTelperion.io.enmap import loadEnMAP
    image = loadEnMAP("/path/to/ENMAP01-____L2A-.../")
    image.decompress()   # → float32 reflectance with NaN nodata
"""

import os
import xml.etree.ElementTree as ET

import numpy as np


# ---------------------------------------------------------------------------
# Product detection
# ---------------------------------------------------------------------------

def is_enmap_product(path):
    """
    Return True if *path* (a directory) looks like an EnMAP product.

    Detection is based on the presence of a ``*-SPECTRAL_IMAGE.BSQ`` file
    somewhere under *path*.
    """
    for root, _, files in os.walk(path):
        if any(f.upper().endswith("-SPECTRAL_IMAGE.BSQ") for f in files):
            return True
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_enmap_files(directory):
    """
    Locate the BSQ, HDR and METADATA.XML inside an EnMAP product directory.

    Handles both the outer and inner directory levels of the standard EnMAP
    distribution layout.

    Args:
        directory (str): path to the EnMAP product (outer or inner dir).

    Returns:
        dict: ``{'bsq': ..., 'hdr': ..., 'xml': ...}`` with absolute paths.

    Raises:
        FileNotFoundError: if required files are not found.
    """
    bsq = hdr = xml = None

    for root, _, files in os.walk(directory):
        for f in files:
            fu = f.upper()
            full = os.path.join(root, f)
            if fu.endswith("-SPECTRAL_IMAGE.BSQ"):
                bsq = full
            elif fu.endswith("-SPECTRAL_IMAGE.HDR"):
                hdr = full
            elif fu.endswith("-METADATA.XML"):
                xml = full

    if bsq is None:
        raise FileNotFoundError(
            "No *-SPECTRAL_IMAGE.BSQ found under %s" % directory
        )
    if hdr is None:
        raise FileNotFoundError(
            "No *-SPECTRAL_IMAGE.HDR found under %s" % directory
        )
    return {"bsq": bsq, "hdr": hdr, "xml": xml}


def _parse_metadata_xml(xml_path):
    """
    Extract per-band wavelengths and FWHM from an EnMAP METADATA.XML.

    Args:
        xml_path (str): absolute path to the XML file.

    Returns:
        dict: with keys ``'wavelengths'`` (ndarray, nm), ``'fwhm'``
              (ndarray, nm), and ``'processing_level'`` (str or None).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    wavelengths = []
    fwhm = []

    for band_el in root.iter("bandID"):
        wl_el = band_el.find("wavelengthCenterOfBand")
        fw_el = band_el.find("FWHMOfBand")
        if wl_el is not None and fw_el is not None:
            wavelengths.append(float(wl_el.text))
            fwhm.append(float(fw_el.text))

    processing_level = None
    pl_el = root.find(".//processingLevel")
    if pl_el is not None:
        processing_level = pl_el.text

    return {
        "wavelengths": np.array(wavelengths, dtype=np.float64),
        "fwhm": np.array(fwhm, dtype=np.float64),
        "processing_level": processing_level,
    }


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def loadEnMAP(path):
    """
    Load an EnMAP L2A product into a BlackImage with int16 data.

    The returned image keeps the raw integer values.  Use
    ``image.decompress()`` to apply the reflectance scale factor and mask
    nodata pixels.

    Args:
        path (str): path to the EnMAP product directory (outer or inner).

    Returns:
        BlackImage: 224-band hyperspectral cube (int16) with wavelengths,
        FWHM, georeferencing, and ``reflectance scale factor = 10000``
        populated in the header.

    Raises:
        FileNotFoundError: if required product files are missing.

    Example::

        img = loadEnMAP("/data/ENMAP01-____L2A-DT.../")
        img.decompress()          # convert to float32 reflectance
        img.quick_plot(bands=(50, 30, 10))
    """
    assert os.path.isdir(path), "Error: path is not a directory — %s" % path

    files = _find_enmap_files(path)

    # --- load the BSQ via the existing GDAL / numpy loaders ----------------
    from .images import loadWithGDAL, loadWithNumpy

    try:
        from osgeo import gdal  # noqa: F401
        img = loadWithGDAL(files["hdr"], dtype=np.int16, mask_zero=False)
    except (ImportError, Exception):
        img = loadWithNumpy(files["hdr"], dtype=np.int16, mask_zero=False)

    # --- enrich header with XML metadata -----------------------------------
    if files["xml"] is not None:
        meta = _parse_metadata_xml(files["xml"])

        if meta["wavelengths"].size == img.band_count():
            img.header.set_wavelengths(meta["wavelengths"])
        if meta["fwhm"].size == img.band_count():
            img.header.set_fwhm(meta["fwhm"])

    # Set band names
    band_names = ["Band %d" % (i + 1) for i in range(img.band_count())]
    img.header.set_band_names(band_names)

    # Ensure critical header fields for decompress()
    img.header["wavelength units"] = "nm"
    img.header["reflectance scale factor"] = 10000
    img.header["data ignore value"] = -32768

    img.push_to_header()
    return img
