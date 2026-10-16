"""
AeroTwin - Unit Tests for Step 7 Model & Simulation Engine
==========================================================
Module: model/test_simulation.py
Target Audience: B.Tech Engineering / Student Level & QA Engineers

Tests:
    1. Simulation Math & Weighting: Verifies micro-environment time-weighted calculations.
    2. Zero & Maximum Efficiency Bounds: Verifies 0% and 100% mitigation scenarios.
    3. Boundary & Negative Clipping: Verifies safe capping of input hours and negative values.
    4. CPCB AQI Tier Mapping: Verifies cutoffs for Good, Satisfactory, Moderate, Poor, Very Poor, Severe.
    5. Model Inference & Fallback: Verifies predict_next_day_pm25 execution and artifact loading.
    6. Metrics JSON Integrity: Verifies presence and contents of model artifacts.
"""

import os
import json
import unittest
import numpy as np

from model.simulation import simulate_personal_exposure, get_cpcb_tier
from model.predict import predict_next_day_pm25, load_model_artifact, get_cpcb_pm25_category


class TestSimulationEngine(unittest.TestCase):
    """Test suite for What-If Exposure Simulation Engine."""

    def test_standard_simulation_math(self):
        """Verifies standard time-weighted exposure calculation."""
        # 100 ug/m3 ambient, 18h indoor at 80% reduction (20 ug/m3), 2h transit at 90% (10 ug/m3), 4h outdoor at 0% (100 ug/m3)
        # Expected: (18*20 + 2*10 + 4*100) / 24 = (360 + 20 + 400) / 24 = 780 / 24 = 32.5 ug/m3
        res = simulate_personal_exposure(
            ambient_pm25=100.0,
            indoor_hours=18.0,
            transit_hours=2.0,
            purifier_reduction_pct=80.0,
            mask_reduction_pct=90.0
        )
        self.assertEqual(res["status"], "SIMULATION_SUCCESS")
        self.assertAlmostEqual(res["scenario_time_weighted"]["pm25"], 32.5, places=1)
        self.assertAlmostEqual(res["scenario_time_weighted"]["net_reduction_pct"], 67.5, places=1)

    def test_zero_mitigation_boundary(self):
        """When reduction is 0%, daily exposure should exactly equal ambient."""
        res = simulate_personal_exposure(
            ambient_pm25=120.0,
            indoor_hours=16.0,
            transit_hours=4.0,
            purifier_reduction_pct=0.0,
            mask_reduction_pct=0.0
        )
        self.assertAlmostEqual(res["scenario_time_weighted"]["pm25"], 120.0, places=2)
        self.assertAlmostEqual(res["scenario_time_weighted"]["net_reduction_pct"], 0.0, places=1)

    def test_complete_mitigation_boundary(self):
        """When 24h is spent under 100% filtration, daily exposure should equal 0."""
        res = simulate_personal_exposure(
            ambient_pm25=200.0,
            indoor_hours=24.0,
            transit_hours=0.0,
            purifier_reduction_pct=100.0,
            mask_reduction_pct=100.0
        )
        self.assertAlmostEqual(res["scenario_time_weighted"]["pm25"], 0.0, places=2)
        self.assertAlmostEqual(res["scenario_time_weighted"]["net_reduction_pct"], 100.0, places=1)

    def test_hour_capping_and_safety(self):
        """Hours exceeding 24 should be safely capped without throwing errors."""
        res = simulate_personal_exposure(
            ambient_pm25=80.0,
            indoor_hours=30.0,  # exceeds 24
            transit_hours=10.0,
            purifier_reduction_pct=50.0,
            mask_reduction_pct=50.0
        )
        self.assertEqual(res["parameters"]["indoor_hours"], 24.0)
        self.assertEqual(res["parameters"]["transit_hours"], 0.0)
        self.assertEqual(res["parameters"]["outdoor_hours"], 0.0)

    def test_cpcb_tier_classification(self):
        """Verifies accurate categorization of concentrations into CPCB tiers."""
        self.assertEqual(get_cpcb_tier(20.0)["tier"], "Good")
        self.assertEqual(get_cpcb_tier(45.0)["tier"], "Satisfactory")
        self.assertEqual(get_cpcb_tier(75.0)["tier"], "Moderate")
        self.assertEqual(get_cpcb_tier(105.0)["tier"], "Poor")
        self.assertEqual(get_cpcb_tier(180.0)["tier"], "Very Poor")
        self.assertEqual(get_cpcb_tier(350.0)["tier"], "Severe")

    def test_custom_reduction_scenario(self):
        """Verifies adjustable user-defined direct reduction scenario."""
        res = simulate_personal_exposure(
            ambient_pm25=150.0,
            custom_reduction_pct=60.0
        )
        self.assertIn("scenario_custom", res)
        # 150 * (1 - 0.6) = 60 ug/m3
        self.assertAlmostEqual(res["scenario_custom"]["pm25"], 60.0, places=1)
        self.assertAlmostEqual(res["scenario_custom"]["reduction_pct"], 60.0, places=1)
        self.assertEqual(res["scenario_custom"]["tier"], "Satisfactory")
        self.assertAlmostEqual(res["scenario_custom"]["net_reduction_ug"], 90.0, places=1)


class TestModelPrediction(unittest.TestCase):
    """Test suite for Next-Day PM2.5 Prediction Model."""

    def test_model_artifact_loading(self):
        """Verifies that the trained model artifact loads cleanly."""
        payload = load_model_artifact()
        self.assertIsNotNone(payload, "Model artifact should be loaded from disk")
        self.assertIn("model", payload)
        self.assertIn("feature_cols", payload)
        self.assertGreater(len(payload["feature_cols"]), 50)

    def test_prediction_output_structure(self):
        """Verifies prediction output structure and sanity."""
        sample = {
            "PM2.5": 95.0,
            "PM10": 180.0,
            "NO2": 35.0,
            "SO2": 12.0,
            "CO": 1.1,
            "O3": 28.0,
            "AQI": 215.0,
            "PM2.5_lag_1d": 90.0,
            "PM2.5_roll_mean_7d": 88.0,
            "Month": 10,
            "Is_Weekend": 0
        }
        res = predict_next_day_pm25(sample)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIsInstance(res["forecast_pm25"], float)
        self.assertGreater(res["forecast_pm25"], 0.0)
        self.assertIn(res["cpcb_category"], ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])
        self.assertTrue(res["is_forecast"])

    def test_metrics_json_integrity(self):
        """Verifies model metrics JSON file."""
        metrics_file = os.path.join(os.path.dirname(__file__), "artifacts", "model_metrics.json")
        self.assertTrue(os.path.exists(metrics_file), "model_metrics.json must exist")
        with open(metrics_file, "r") as f:
            data = json.load(f)
        self.assertIn("model_test", data)
        self.assertIn("baseline_persistence", data)
        self.assertGreater(data["model_test"]["r2"], 0.75)


if __name__ == "__main__":
    unittest.main()
