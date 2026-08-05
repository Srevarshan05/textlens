"""
Tests for textlens.sdk module using standard unittest.
"""

import unittest
from textlens import TextLens, is_cuda_available


class TestSDK(unittest.TestCase):

    def test_sdk_init_lazy(self):
        """Test initializing TextLens with auto_load=False."""
        ocr = TextLens(auto_load=False)
        self.assertFalse(ocr.is_loaded)
        self.assertIn(ocr.device, ("cuda", "cpu"))
        self.assertIsInstance(ocr.is_cuda(), bool)

    def test_sdk_hardware_property(self):
        """Test accessing hardware info via SDK instance."""
        ocr = TextLens(auto_load=False)
        hw = ocr.hardware
        self.assertIsNotNone(hw)
        self.assertTrue(hasattr(hw, "gpu_available"))

    def test_sdk_invalid_image_source(self):
        """Test passing invalid input format to read raises ValueError."""
        ocr = TextLens(auto_load=False)
        ocr._is_loaded = True
        with self.assertRaises(ValueError):
            ocr.read(12345)


if __name__ == "__main__":
    unittest.main()
