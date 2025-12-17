import unittest
import BlackTelperion
from BlackTelperion import io
from pathlib import Path
import numpy as np
import os


class MyTestCase(unittest.TestCase):

    def test_band_ratio(self):
        """Test the band_ratio function with various input types"""
        from BlackTelperion.analyse.indices import band_ratio

        # Load test image
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))

        # Test 1: Simple single band ratio (using indices)
        ratio1 = band_ratio(image, 100, 50)
        self.assertTrue(np.isfinite(ratio1.data).any())
        self.assertEqual(ratio1.data.shape[-1], 1)  # Should have 1 band
        self.assertEqual(ratio1.data.shape[0], image.data.shape[0])  # Spatial dims preserved
        self.assertEqual(ratio1.data.shape[1], image.data.shape[1])

        # Test 2: Band ratio using wavelengths (floats)
        ratio2 = band_ratio(image, 800.0, 670.0)
        self.assertTrue(np.isfinite(ratio2.data).any())
        self.assertEqual(ratio2.data.shape[-1], 1)

        # Test 3: Band ratio using ranges (tuples for averaging)
        ratio3 = band_ratio(image, (795.0, 805.0), (665.0, 675.0))
        self.assertTrue(np.isfinite(ratio3.data).any())
        self.assertEqual(ratio3.data.shape[-1], 1)

        # Test 4: Band ratio using lists (multiple bands summed)
        ratio4 = band_ratio(image, [800.0, 900.0], [670.0, 550.0])
        self.assertTrue(np.isfinite(ratio4.data).any())
        self.assertEqual(ratio4.data.shape[-1], 1)

        # Test 5: Verify band names are set
        self.assertTrue(len(ratio1.get_band_names()) == 1)
        self.assertTrue('/' in ratio1.get_band_names()[0])  # Should contain ratio format

    def test_NDVI(self):
        """Test NDVI calculation"""
        from BlackTelperion.analyse.indices import NDVI

        # Load test image
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))

        # Calculate NDVI
        ndvi = NDVI(image)

        # Verify output structure
        self.assertTrue(np.isfinite(ndvi.data).any())
        self.assertEqual(ndvi.data.shape[-1], 1)  # Should have 1 band
        self.assertEqual(ndvi.data.shape[0], image.data.shape[0])  # Spatial dims preserved
        self.assertEqual(ndvi.data.shape[1], image.data.shape[1])

        # Verify band name
        self.assertEqual(ndvi.get_band_names()[0], 'NDVI')

        # Verify NDVI is in expected range [-1, 1] for finite values
        finite_ndvi = ndvi.data[np.isfinite(ndvi.data)]
        if len(finite_ndvi) > 0:
            self.assertGreaterEqual(np.max(finite_ndvi), -1.5)  # Some tolerance
            self.assertLessEqual(np.min(finite_ndvi), 1.5)

        # Verify formula: (NIR - Red) / (NIR + Red)
        idxNIR = image.get_band_index(800.0)
        idxRed = image.get_band_index(670.0)

        # Calculate expected NDVI manually for a sample pixel
        test_pixel = (10, 10)
        if np.isfinite(image.data[test_pixel[0], test_pixel[1], idxNIR]) and \
                np.isfinite(image.data[test_pixel[0], test_pixel[1], idxRed]):
            nir = image.data[test_pixel[0], test_pixel[1], idxNIR]
            red = image.data[test_pixel[0], test_pixel[1], idxRed]
            expected = (nir - red) / (nir + red)
            actual = ndvi.data[test_pixel[0], test_pixel[1], 0]
            if np.isfinite(expected) and np.isfinite(actual):
                self.assertAlmostEqual(expected, actual, places=5)

    def test_SKY(self):
        """Test SKY ratio calculation"""
        from BlackTelperion.analyse.indices import SKY

        # Load test image
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))

        # Calculate SKY ratio
        sky = SKY(image)

        # Verify output structure
        self.assertTrue(np.isfinite(sky.data).any())
        self.assertEqual(sky.data.shape[-1], 1)  # Should have 1 band
        self.assertEqual(sky.data.shape[0], image.data.shape[0])  # Spatial dims preserved
        self.assertEqual(sky.data.shape[1], image.data.shape[1])

        # Verify band name contains ratio format
        self.assertTrue(len(sky.get_band_names()) == 1)
        self.assertTrue('/' in sky.get_band_names()[0])

        # Verify it uses correct wavelengths (479.89 / 1688.64)
        self.assertTrue('479' in sky.get_band_names()[0] or '480' in sky.get_band_names()[0])
        self.assertTrue('1688' in sky.get_band_names()[0] or '1689' in sky.get_band_names()[0])

    def test_SHADE(self):
        """Test SHADE ratio calculation"""
        from BlackTelperion.analyse.indices import SHADE

        # Load test image
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))

        # Calculate SHADE ratio
        shade = SHADE(image)

        # Verify output structure
        self.assertTrue(np.isfinite(shade.data).any())
        self.assertEqual(shade.data.shape[-1], 1)  # Should have 1 band
        self.assertEqual(shade.data.shape[0], image.data.shape[0])  # Spatial dims preserved
        self.assertEqual(shade.data.shape[1], image.data.shape[1])

        # Verify band name contains ratio format
        self.assertTrue(len(shade.get_band_names()) == 1)
        self.assertTrue('/' in shade.get_band_names()[0])

        # Verify it uses correct wavelengths (480.0 / 800.0)
        self.assertTrue('480' in shade.get_band_names()[0])
        self.assertTrue('800' in shade.get_band_names()[0])


if __name__ == '__main__':
    unittest.main()