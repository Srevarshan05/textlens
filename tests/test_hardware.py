"""
Tests for textlens.hardware module using standard unittest.
"""

import unittest
import textlens
from textlens.hardware import HardwareInfo, is_cuda_available, get_hardware_info, print_hardware_status


class TestHardware(unittest.TestCase):

    def test_is_cuda_available(self):
        """Verify is_cuda_available returns boolean."""
        res = is_cuda_available()
        self.assertIsInstance(res, bool)
        self.assertEqual(res, textlens.is_cuda_available())

    def test_get_hardware_info(self):
        """Verify get_hardware_info returns populated HardwareInfo object."""
        info = get_hardware_info()
        self.assertIsInstance(info, HardwareInfo)
        self.assertIsInstance(info.gpu_available, bool)
        self.assertIn(info.device_type, ("cuda", "cpu"))
        self.assertIsInstance(info.cpu_count, int)
        self.assertGreaterEqual(info.cpu_count, 1)

        d = info.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("gpu_available", d)
        self.assertIn("device_type", d)

    def test_print_hardware_status(self):
        """Verify print_hardware_status executes cleanly."""
        print_hardware_status()


if __name__ == "__main__":
    unittest.main()
