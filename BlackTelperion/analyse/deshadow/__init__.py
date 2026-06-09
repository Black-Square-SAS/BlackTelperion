"""
Shadow detection and removal for hyperspectral imagery.

Shadowed pixels reflect less energy across the whole spectrum, so the *spectral
integral* of a pixel (the area under its reflectance curve, by the trapezoidal
rule) is systematically lower than for sunlit pixels. Classifying every pixel
whose integral falls below a chosen percentage of the scene's maximum integral
yields a shadow mask; an optional morphological pass removes speckle and fills
small gaps. Pixels whose spectrum is entirely ``NaN`` (no-data) are treated as
shadow.

The integral is accumulated in horizontal row-strips so peak memory stays
proportional to a strip rather than the whole cube. Two pairs of functions are
provided — one for images that fit in memory (operating on a
:class:`~BlackTelperion.blackimage.BlackImage`) and one for images that do not
(streaming a file on disk via GDAL windows):

.. autosummary::
   :nosignatures:

   shadow_mask
   deshadow_image
   shadow_mask_large
   deshadow_large_image

``shadow_mask`` / ``shadow_mask_large`` return just the mask; ``deshadow_image``
/ ``deshadow_large_image`` apply it, flagging every band of each shadow pixel.

Example
-------
.. code-block:: python

    from BlackTelperion.analyse import deshadow_image, deshadow_large_image

    # image that fits in memory -> corrected BlackImage
    clean = deshadow_image(img, threshold=30.0)

    # image too large for memory -> streamed file in / file out
    deshadow_large_image("cube.dat", "cube_clean.dat", threshold=30.0)
"""

from .core import shadow_mask, deshadow_image
from .large_image import shadow_mask_large, deshadow_large_image

__all__ = [
    "shadow_mask",
    "deshadow_image",
    "shadow_mask_large",
    "deshadow_large_image",
]
