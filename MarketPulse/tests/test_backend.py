import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure workspace is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MarketPulse.backend.main import app
from MarketPulse.database.connection import SessionLocal
from MarketPulse.database.models import Stock

class TestMarketPulsePlatform(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    def test_health_endpoint(self):
        """Verifies health check endpoint returns 200 and healthy status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_stocks_endpoint(self):
        """Verifies GET /stocks returns list of tracked assets."""
        response = self.client.get("/stocks")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        if len(data) > 0:
            self.assertIn("ticker", data[0])
            self.assertIn("category", data[0])

    def test_predictions_endpoint(self):
        """Verifies GET /predictions response schema."""
        response = self.client.get("/predictions")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_analytics_endpoint(self):
        """Verifies GET /analytics runs custom SQL metrics."""
        response = self.client.get("/analytics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("top_gainers", data)
        self.assertIn("top_losers", data)
        self.assertIn("highest_volume", data)

if __name__ == "__main__":
    unittest.main()
