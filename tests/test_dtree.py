import unittest
import BlackTelperion
from BlackTelperion import io
from pathlib import Path
import numpy as np
import os


class MyTestCase(unittest.TestCase):
    def test_dtree(self):
        from BlackTelperion.analyse.dtree import decision_tree

        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))

        # Create simple boolean layers based on spectral thresholds
        # Layer 1: High reflectance in NIR (~800nm)
        nir_band = image.get_band_index(800.0)
        layer1 = image.data[:, :, nir_band] > np.nanmedian(image.data[:, :, nir_band])

        # Layer 2: High reflectance in SWIR (~2200nm)
        swir_band = image.get_band_index(2200.0)
        layer2 = image.data[:, :, swir_band] > np.nanmedian(image.data[:, :, swir_band])

        # Layer 3: Low reflectance in visible (~600nm)
        vis_band = image.get_band_index(600.0)
        layer3 = image.data[:, :, vis_band] < np.nanmedian(image.data[:, :, vis_band])

        # Define decision tree with simple 2-level structure
        labels = {
            (True, True): "Vegetation",
            (True, False): "Soil",
            (False, True): "Rock",
            (False, False): "Shadow"
        }

        # Run decision tree
        result, names = decision_tree([layer1, layer2], labels)

        # Test output shape matches input
        self.assertEqual(result.shape, layer1.shape)

        # Test that result contains valid class labels (0-4)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= len(labels)))

        # Test names list
        self.assertEqual(names[0], "Unknown")
        self.assertEqual(len(names), len(labels) + 1)
        self.assertIn("Vegetation", names)
        self.assertIn("Soil", names)
        self.assertIn("Rock", names)
        self.assertIn("Shadow", names)

        # Test 3-level decision tree with None branches
        labels_3level = {
            (True, True, True): "Class1",
            (True, False, None): "Class2",
            (False, None, True): "Class3"
        }

        result_3level, names_3level = decision_tree([layer1, layer2, layer3], labels_3level)

        # Test output shape
        self.assertEqual(result_3level.shape, layer1.shape)

        # Test that result contains valid labels
        self.assertTrue(np.all(result_3level >= 0))

        # Test names
        self.assertEqual(names_3level[0], "Unknown")
        self.assertIn("Class1", names_3level)
        self.assertIn("Class2", names_3level)
        self.assertIn("Class3", names_3level)