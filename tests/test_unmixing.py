import unittest
import BlackTelperion
from BlackTelperion import io
from pathlib import Path
import numpy as np
import os

class MyTestCase(unittest.TestCase):
    def test_unmixing(self):
        from BlackTelperion.analyse.unmixing import mix, unmix, endmembers

        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))
        print("Loaded image:", type(image))


        for data in [image]:
            # Get valid (non-zero) spectra
            X_all = data.X(onlyFinite=True)
            print("Image shape:", image.data.shape)
            print("Wavelengths:", data.get_wavelengths()[:5], "... total:", len(data.get_wavelengths()))

            # Find pixels that are not all zeros
            non_zero_mask = (X_all != 0).any(axis=-1)
            X_valid = X_all[non_zero_mask]
            print("X_all shape:", X_all.shape)
            print("Valid mask count:", np.sum(non_zero_mask))
            print("X_valid shape:", X_valid.shape)

            print("Example spectrum (first valid): min=", np.min(X_valid[0]),
                  "max=", np.max(X_valid[0]))
            print("Example spectrum (last valid): min=", np.min(X_valid[-1]),
                  "max=", np.max(X_valid[-1]))


            if len(X_valid) < 2:
                self.skipTest("Not enough valid pixels in test data")

            # Use first and last VALID pixels as endmembers
            em1 = X_valid[0]
            em2 = X_valid[-1]

            # Verify endmembers are valid
            self.assertGreater(em1.max(), 0, "em1 should have non-zero values")
            self.assertGreater(em2.max(), 0, "em2 should have non-zero values")

            E = BlackTelperion.BlackLibrary(np.vstack([em1, em2]),
                                            lab=['A', 'B'], wav=data.get_wavelengths())
            print("Selected endmember em1: min=", np.min(em1), "max=", np.max(em1))
            print("Selected endmember em2: min=", np.min(em2), "max=", np.max(em2))
            print("Endmember library E shape:", E.data.shape)

            # Create abundance map
            A = data.copy()
            A.data = np.random.uniform(size=(data.data.shape[:-1]) + (2,))
            A.data = A.data / np.sum(A.data, axis=-1)[..., None]
            print("Abundance A shape:", A.data.shape)
            print("Abundance sum check (mean should be 1):",
                  np.mean(np.sum(A.data, axis=-1)))
            print("Abundance min/max:", np.min(A.data), np.max(A.data))

            # Run forward model
            X = mix(A, E)
            self.assertTrue(X.data.shape[-1] == E.data.shape[-1])
            print("Mixed X shape:", X.data.shape)
            print("Mixed spectra example: min=", np.min(X.data), "max=", np.max(X.data))


            # Run backward model
            for m in ['nnls', 'fcls']:
                A2 = unmix(X, E, method=m)
                print(f"Running unmix method: {m}")
                print("Recovered abundance A2: shape:", A2.data.shape)
                print("Mean abs error:", np.mean(np.abs(A2.data - A.data)))

                self.assertLess(np.mean(np.abs(A2.data - A.data)), 1e-4)

            # Test endmember extraction
            for m in ['atgp', 'fippi', 'nfindr', 'ppi']:
                em, ix = endmembers(X, 3, method=m)
                print(f"\nEndmember extraction method: {m}")
                print("Returned em shape:", em.data.shape)
                print("Returned indices ix:", ix)

                # Compare first returned EM with X at index
                if len(ix.shape) > 1:
                    diff = np.max(np.abs(em.data[0, 0, :] - X.data[tuple(ix[0]), :]))
                else:
                    diff = np.max(np.abs(em.data[0, 0, :] - X.data[ix[0], :]))

                print("Difference between extracted EM and pixel X:", diff)

                # Compare similarity to original em1/em2
                dist1 = np.mean(np.abs(em1 - em.data[:, 0, :]))
                dist2 = np.mean(np.abs(em2 - em.data[:, 0, :]))
                print("Distance to em1:", dist1)
                print("Distance to em2:", dist2)
                print("Best match:", "em1" if dist1 < dist2 else "em2")

                # Check that endmembers match their indices
                if len(ix.shape) > 1:
                    self.assertLess(np.max(np.abs(em.data[0, 0, :] - X.data[tuple(ix[0]), :])), 1e-6)
                else:
                    self.assertLess(np.max(np.abs(em.data[0, 0, :] - X.data[ix[0], :])), 1e-6)

                # Check that the endmembers are at least similar to the real ones
                self.assertLess(min(np.mean(np.abs(em1 - em.data[:, 0, :])),
                                    np.mean(np.abs(em2 - em.data[:, 0, :]))), 0.05)