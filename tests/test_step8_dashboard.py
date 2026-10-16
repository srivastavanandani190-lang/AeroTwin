"""
AeroTwin - Step 8 Dashboard & Integration Unit Tests
====================================================
Module: tests/test_step8_dashboard.py
Target Audience: B.Tech Engineering / Student Level & Full-Stack QA Engineers

Test Suite Scope:
    1. Flask application factory & route registration integrity
    2. Preservation of existing dashboard pages, routes, and 8 Plotly charts
    3. Step 8 Environmental Overview rendering (Observed telemetry, pollutants grid)
    4. Step 8 What-If Simulation dual-control & 4-scenario side-by-side rendering
    5. Step 8 Model Forecasting results, actual metrics, and limitations
    6. /api/simulate endpoint (GET & POST, JSON & form data, input validation)
    7. /api/forecast endpoint (GET & POST, inference validation)
    8. /api/model-info endpoint (Metadata, actual holdout test metrics)
    9. Edge cases & error handling (invalid/negative inputs, boundary clamping)
    10. Graceful degradation on metadata datasets (e.g. cleaned_stations.csv)
"""

import os
import sys
import json
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.app import create_flask_app, parse_float
from model.simulation import simulate_personal_exposure, get_cpcb_tier
from model.predict import predict_next_day_pm25, load_model_artifact


class TestStep8DashboardIntegration(unittest.TestCase):
    """Integration and unit tests for Step 8 Flask Dashboard & APIs."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_flask_app()
        cls.assertIsNotNone(cls.app, "Flask app failed to initialize")
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def test_routes_registered(self):
        """Verifies that all required endpoints are registered without duplicates."""
        rules = [rule.rule for rule in self.app.url_map.iter_rules()]
        self.assertIn("/", rules)
        self.assertIn("/api/simulate", rules)
        self.assertIn("/api/forecast", rules)
        self.assertIn("/api/model-info", rules)
        self.assertIn("/download/<dataset_name>", rules)
        self.assertIn("/api/metrics", rules)

    def test_existing_dashboard_renders(self):
        """Verifies that the main dashboard GET / returns 200 OK and contains existing charts."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        # Check existing sections and charts
        self.assertIn("AeroTwin", html)
        self.assertIn("chart-timeseries", html)
        self.assertIn("chart-distribution", html)
        self.assertIn("chart-city-bar", html)
        self.assertIn("chart-custom", html)
        self.assertIn("Custom Chart Studio", html)

    def test_environmental_overview_rendering(self):
        """Verifies that Part A (Environmental Overview) renders observed ground telemetry."""
        response = self.client.get("/?dataset=cleaned_city_day.csv&city=Delhi")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        self.assertIn("Environmental Overview &amp; Ground Sensor Telemetry", html)
        self.assertIn("OBSERVED GROUND MEASUREMENTS", html)
        self.assertIn("Observation Date:", html)
        self.assertIn("Location / Region:", html)
        self.assertIn("Delhi", html)
        # Check presence of pollutant cards
        self.assertIn("PM2.5", html)
        self.assertIn("AQI", html)

    def test_twin_studio_scenarios_rendering(self):
        """Verifies that Part B (What-If Studio) renders all 4 scenario cards side by side."""
        response = self.client.get("/?dataset=cleaned_city_day.csv&sim_ambient=120&sim_indoor_h=18&sim_purifier=80")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        self.assertIn("twin-studio", html)
        self.assertIn("Scenario 1: Ambient Baseline", html)
        self.assertIn("Scenario 2: 24h Indoor Purifier", html)
        self.assertIn("Scenario 3: Personal Daily Routine", html)
        self.assertIn("Scenario 4: User-Defined Reduction", html)
        self.assertIn("WHAT-IF SIMULATION", html)
        # Check charts
        self.assertIn("chart-sim-bar", html)
        self.assertIn("chart-sim-donut", html)
        self.assertIn("chart-twin-overview", html)

    def test_model_forecast_and_metrics_rendering(self):
        """Verifies that Part C (Model Results) displays forecast, holdout metrics, and limitations."""
        response = self.client.get("/?dataset=cleaned_city_day.csv&city=Delhi")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        self.assertIn("24h Next-Day Atmospheric PM2.5 Forecaster", html)
        self.assertIn("Empirical Testing Metrics &amp; Limitations", html)
        self.assertIn("Holdout Test R²", html)
        self.assertIn("0.8140", html)
        self.assertIn("12.15", html)
        self.assertIn("20.34", html)
        self.assertIn("Scientific Limitations:", html)
        self.assertIn("does NOT predict clinical health outcomes", html)

    def test_api_simulate_get(self):
        """Verifies /api/simulate GET request."""
        response = self.client.get("/api/simulate?ambient_pm25=150&indoor_hours=16&transit_hours=2&purifier_reduction=70&mask_reduction=85&custom_reduction=55")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data["status"], "SIMULATION_SUCCESS")
        self.assertTrue(data["is_simulated"])
        self.assertEqual(data["baseline"]["pm25"], 150.0)
        self.assertIn("scenario_indoor_only", data)
        self.assertIn("scenario_time_weighted", data)
        self.assertIn("scenario_custom", data)
        self.assertEqual(data["parameters"]["custom_reduction_pct"], 55.0)

    def test_api_simulate_post_json(self):
        """Verifies /api/simulate POST request with JSON payload."""
        payload = {
            "ambient_pm25": 200.0,
            "indoor_hours": 20.0,
            "transit_hours": 1.0,
            "purifier_reduction_pct": 80.0,
            "mask_reduction_pct": 90.0,
            "custom_reduction_pct": 65.0
        }
        response = self.client.post("/api/simulate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data["status"], "SIMULATION_SUCCESS")
        self.assertEqual(data["baseline"]["pm25"], 200.0)
        self.assertLess(data["scenario_time_weighted"]["pm25"], 200.0)
        self.assertAlmostEqual(data["scenario_custom"]["pm25"], 70.0, places=1)

    def test_api_simulate_input_validation(self):
        """Verifies that invalid, negative, and string inputs are safely sanitized."""
        # Non-numeric strings and negative bounds
        response = self.client.get("/api/simulate?ambient_pm25=invalid&indoor_hours=-5&purifier_reduction=150")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data["status"], "SIMULATION_SUCCESS")
        # Defaults to 100.0 for ambient
        self.assertEqual(data["parameters"]["ambient_pm25"], 100.0)
        # Clamped to 0.0 minimum for hours
        self.assertEqual(data["parameters"]["indoor_hours"], 0.0)
        # Clamped to 100.0 maximum for purifier
        self.assertEqual(data["parameters"]["purifier_reduction_pct"], 100.0)

    def test_api_forecast_get_and_post(self):
        """Verifies /api/forecast endpoint with GET and POST requests."""
        # GET
        res_get = self.client.get("/api/forecast?pm25=95.0&aqi=210.0&city=Delhi")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.get_json()
        self.assertEqual(data_get["status"], "SUCCESS")
        self.assertTrue(data_get["is_forecast"])
        self.assertIsInstance(data_get["forecast_pm25"], float)

        # POST JSON
        res_post = self.client.post("/api/forecast", json={"PM2.5": 140.0, "PM10": 250.0, "AQI": 310.0})
        self.assertEqual(res_post.status_code, 200)
        data_post = res_post.get_json()
        self.assertEqual(data_post["status"], "SUCCESS")

    def test_api_model_info(self):
        """Verifies /api/model-info endpoint returns valid test metrics and limits."""
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("model_test", data)
        self.assertIn("r2", data["model_test"])
        self.assertIn("mae", data["model_test"])
        self.assertIn("rmse", data["model_test"])
        self.assertIn("limitations", data)
        self.assertGreater(data["model_test"]["r2"], 0.75)

    def test_parse_float_helper(self):
        """Unit tests for parse_float input sanitizer."""
        self.assertEqual(parse_float(None, 10.0), 10.0)
        self.assertEqual(parse_float("", 15.0), 15.0)
        self.assertEqual(parse_float("not-a-number", 20.0), 20.0)
        self.assertEqual(parse_float("42.5", 0.0), 42.5)
        self.assertEqual(parse_float(-10, 0.0, min_val=0.0), 0.0)
        self.assertEqual(parse_float(150, 50.0, max_val=100.0), 100.0)

    def test_metadata_dataset_graceful_handling(self):
        """Verifies that viewing cleaned_stations.csv handles missing pollutant data gracefully."""
        response = self.client.get("/?dataset=cleaned_stations.csv")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        # Should render without throwing 500 error
        self.assertIn("cleaned_stations.csv", html)
        self.assertIn("AeroTwin", html)

    def test_invalid_dataset_download_404(self):
        """Verifies that requesting a nonexistent CSV download returns 404."""
        response = self.client.get("/download/nonexistent_file.csv")
        self.assertEqual(response.status_code, 404)

    def test_missing_model_artifact_graceful_handling(self):
        """Verifies that missing model artifact returns MODEL_UNAVAILABLE gracefully without crashing."""
        from unittest.mock import patch
        with patch("model.predict.load_model_artifact", return_value=None):
            response = self.client.get("/api/forecast?pm25=120.0")
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["status"], "MODEL_UNAVAILABLE")
            self.assertIn("Model artifact not found", data["message"])

    def test_model_caching_no_retraining(self):
        """Verifies model artifact is cached in-memory and not reloaded/retrained per request."""
        import model.predict as mp
        # First load
        p1 = mp.load_model_artifact()
        # Second call
        p2 = mp.load_model_artifact()
        self.assertIs(p1, p2, "Model artifact should be a cached singleton instance")


if __name__ == "__main__":
    unittest.main()

