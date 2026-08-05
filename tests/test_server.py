"""
Tests for textlens.server module using standard unittest.
"""

import unittest
from textlens import TextLens

try:
    from fastapi.testclient import TestClient
    from textlens.server import create_app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class TestServer(unittest.TestCase):

    def test_create_app_routes(self):
        """Verify FastAPI application route registration and root endpoint."""
        if not FASTAPI_AVAILABLE:
            self.skipTest("FastAPI not installed")

        engine = TextLens(auto_load=False)
        app = create_app(engine=engine)
        client = TestClient(app)

        # Test root endpoint
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "TextLens OCR REST Service")
        self.assertEqual(data["status"], "online")
        self.assertIn("docs", data)

        # Test health endpoint
        health_resp = client.get("/api/v1/health")
        self.assertEqual(health_resp.status_code, 200)

        # Test hardware endpoint
        hw_resp = client.get("/api/v1/hardware")
        self.assertEqual(hw_resp.status_code, 200)
        hw_data = hw_resp.json()
        self.assertIn("gpu_available", hw_data)


if __name__ == "__main__":
    unittest.main()
