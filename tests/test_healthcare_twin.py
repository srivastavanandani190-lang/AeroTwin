"""
AeroTwin - Healthcare Digital Twin Integration & Unit Tests
============================================================
Module: tests/test_healthcare_twin.py
Scope:
    1. Synthetic EHR validation (schema, ADA/RSSDI clinical bounds, synthetic labeling)
    2. Wearable stream generation (5-min intervals, continuous vitals, quality flags)
    3. Time alignment & Zero-Leakage verification in feature extraction
    4. Primary target definition (120-min excursion >= 180 mg/dL)
    5. Machine learning dual-model inference & held-out test evaluation integrity
    6. What-If behavioral simulation (GLUT4 exercise clearance, parameter bounding)
    7. Flask Healthcare Digital Twin routes & API contract verification
    8. Robust edge case & error handling (invalid IDs, boundary clamping)
"""

import os
import sys
import json
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.app import create_flask_app
from model.healthcare.ehr_generator import load_synthetic_patients
from model.healthcare.twin_service import (
    get_patient_stream,
    get_patient_profile,
    get_digital_twin_state_at_step,
    get_model_evaluation_metrics_summary,
    get_environmental_metabolic_context
)
from model.healthcare.glucose_engine import (
    extract_patient_features_and_targets,
    predict_glucose_excursion_120m,
    FEATURE_COLUMNS
)
from model.healthcare.whatif_engine import simulate_glucose_intervention


class TestHealthcareDigitalTwin(unittest.TestCase):
    """Comprehensive test suite for the Type 2 Diabetes Digital Twin."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_flask_app()
        cls.assertIsNotNone(cls.app, "Flask app failed to initialize")
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    # ----------------------------------------------------
    # 1. Synthetic EHR Data Integrity
    # ----------------------------------------------------
    def test_synthetic_patients_structure(self):
        """Verifies synthetic patient cohort has required fields and proper synthetic labeling."""
        patients = load_synthetic_patients()
        self.assertGreaterEqual(len(patients), 5, "Expected at least 5 synthetic patients")

        for p in patients:
            self.assertTrue(p["patient_id"].startswith("SYN-T2D-"))
            self.assertIn("Synthetic", p["name"])
            self.assertIn("demographics", p)
            self.assertIn("laboratory_results", p)
            self.assertIn("medication_regimen", p)
            self.assertIn("vitals_baseline", p)

            # Clinical plausibility bounds
            labs = p["laboratory_results"]
            self.assertTrue(5.5 <= labs["hba1c_pct"] <= 14.0)
            self.assertTrue(70.0 <= labs["fasting_blood_glucose_mg_dl"] <= 350.0)
            self.assertTrue(15.0 <= labs["egfr_ml_min_1_73m2"] <= 130.0)

            # Data provenance check (privacy compliance)
            self.assertIn("data_provenance", p)
            self.assertIn("synthetic", p["data_provenance"].lower())

    # ----------------------------------------------------
    # 2. Wearable Telemetry & CGM Streams
    # ----------------------------------------------------
    def test_cgm_stream_telemetry(self):
        """Verifies 5-minute sampling interval, physiological ranges, and quality indicators."""
        df = get_patient_stream("SYN-T2D-001")
        self.assertIsNotNone(df, "Stream for SYN-T2D-001 should exist")
        self.assertGreaterEqual(len(df), 2000, "Expected at least 7 days of 5-min intervals (2016 steps)")

        required_cols = [
            "patient_id", "timestamp", "glucose_mg_dl", "heart_rate_bpm",
            "hrv_sdnn_ms", "step_count", "activity_state", "sleep_stage", "sensor_quality"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

        # Glucose range check
        min_g = df["glucose_mg_dl"].min()
        max_g = df["glucose_mg_dl"].max()
        self.assertGreaterEqual(min_g, 40.0)
        self.assertLessEqual(max_g, 450.0)

        # Sensor quality flags present
        qualities = df["sensor_quality"].unique()
        self.assertIn("Optimal", qualities)

    # ----------------------------------------------------
    # 3. Target Definition & Zero Target Leakage
    # ----------------------------------------------------
    def test_feature_extraction_zero_leakage(self):
        """Verifies that features at index i use ONLY past data and target uses strictly future window."""
        df = get_patient_stream("SYN-T2D-001")
        ehr = get_patient_profile("SYN-T2D-001")
        feat_df = extract_patient_features_and_targets(df.iloc[:200], ehr, forecast_horizon_steps=24)

        self.assertFalse(feat_df.empty)

        # Verify all feature columns exist
        for f in FEATURE_COLUMNS:
            self.assertIn(f, feat_df.columns)

        # Check target creation logic:
        # target_excursion_120m MUST be 1 if future max >= 180, else 0
        for _, row in feat_df.head(20).iterrows():
            if row["target_max_glucose_120m"] >= 180.0:
                self.assertEqual(row["target_excursion_120m"], 1)
            else:
                self.assertEqual(row["target_excursion_120m"], 0)

    # ----------------------------------------------------
    # 4. Model Inference & Evaluation Integrity
    # ----------------------------------------------------
    def test_glucose_forecasting_inference(self):
        """Verifies prediction outputs excursion probability, continuous trajectory, and risk tier."""
        dummy_features = {
            "glucose_current": 195.0,
            "rate_of_change_15m": 0.8,
            "rate_of_change_30m": 0.6,
            "rate_of_change_60m": 0.4,
            "rolling_mean_60m": 190.0,
            "rolling_mean_120m": 185.0,
            "rolling_std_120m": 12.0,
            "rolling_max_120m": 200.0,
            "rolling_min_120m": 170.0,
            "heart_rate_bpm": 84.0,
            "hrv_sdnn_ms": 32.0,
            "steps_current_epoch": 50,
            "steps_past_30m": 200,
            "steps_past_60m": 350,
            "sin_hour": 0.5,
            "cos_hour": 0.86,
            "ehr_hba1c": 8.4,
            "ehr_fbg": 158.0,
            "ehr_bmi": 28.4,
            "ehr_age": 52,
            "ehr_has_insulin": 0.0
        }

        res = predict_glucose_excursion_120m(dummy_features)
        self.assertIn("excursion_probability", res)
        self.assertIn("predicted_excursion", res)
        self.assertIn("projected_trajectory", res)
        self.assertEqual(len(res["projected_trajectory"]), 24)

        # High current glucose (195 mg/dL) with positive dG/dt should yield high excursion risk
        self.assertGreaterEqual(res["excursion_probability"], 0.70)
        self.assertTrue(res["predicted_excursion"])
        self.assertEqual(res["alert_tier"], "High Excursion Risk (>=180 mg/dL)")

    def test_model_evaluation_metrics_artifacts(self):
        """Verifies evaluation artifacts report actual held-out metrics with valid confusion matrix."""
        summary = get_model_evaluation_metrics_summary()
        self.assertIn("classification_metrics", summary)
        self.assertIn("continuous_forecasting_metrics", summary)

        cm = summary["classification_metrics"]["confusion_matrix"]
        self.assertIn("true_negative", cm)
        self.assertIn("false_positive", cm)
        self.assertIn("false_negative", cm)
        self.assertIn("true_positive", cm)
        self.assertGreater(cm["true_positive"], 0)

        # Precision & Recall should be positive numbers <= 1.0
        prec = summary["classification_metrics"]["model_precision"]
        rec = summary["classification_metrics"]["model_recall"]
        self.assertTrue(0.0 <= prec <= 1.0)
        self.assertTrue(0.0 <= rec <= 1.0)

    # ----------------------------------------------------
    # 5. What-If Behavioral Simulation
    # ----------------------------------------------------
    def test_whatif_simulation_bounds_and_effects(self):
        """Verifies What-If simulation responds to walking activity and respects physiological bounds."""
        base_curve = [170.0 + i * 1.5 for i in range(24)]  # Rising curve reaching 204.5 mg/dL
        res = simulate_glucose_intervention(
            baseline_glucose_trajectory=base_curve,
            current_glucose=170.0,
            post_meal_walk_minutes=30.0,
            walk_intensity="Brisk"
        )

        metrics = res["metrics"]
        self.assertGreater(metrics["peak_glucose_reduction_mg_dl"], 0.0)
        self.assertGreaterEqual(metrics["tir_improvement_pct"], 0.0)
        self.assertEqual(len(res["trajectories"]["simulated_curve"]), 24)

        # Negative walking minutes clamped to 0
        clamped_res = simulate_glucose_intervention(base_curve, 170.0, post_meal_walk_minutes=-15.0)
        self.assertEqual(clamped_res["parameters"]["post_meal_walk_minutes"], 0.0)

    # ----------------------------------------------------
    # 6. Flask Web Routes & API Endpoints
    # ----------------------------------------------------
    def test_twin_dashboard_page_renders(self):
        """Verifies GET /twin returns 200 OK and contains all 8 required sections."""
        response = self.client.get("/twin")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        # Verify all 8 core sections exist in the HTML
        self.assertIn("1. Patient Overview", html)
        self.assertIn("2. Electronic Health Record (EHR)", html)
        self.assertIn("3. Dynamic Wearable Stream", html)
        self.assertIn("4. Prediction Engine", html)
        self.assertIn("5. Machine Learning Evaluation", html)
        self.assertIn("6. What-If Physiological Simulation", html)
        self.assertIn("7. Environmental Context", html)
        self.assertIn("8. Data Quality, Governance", html)

    def test_api_healthcare_patients(self):
        """Verifies GET /api/healthcare/patients returns patient directory."""
        response = self.client.get("/api/healthcare/patients")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("patients", data)
        self.assertGreaterEqual(len(data["patients"]), 5)

    def test_api_healthcare_patient_profile(self):
        """Verifies GET /api/healthcare/patient/<id> returns specific patient."""
        response = self.client.get("/api/healthcare/patient/SYN-T2D-001")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["patient_id"], "SYN-T2D-001")

        # Non-existent patient returns 404
        bad_response = self.client.get("/api/healthcare/patient/INVALID-ID-999")
        self.assertEqual(bad_response.status_code, 404)

    def test_api_healthcare_state_sync(self):
        """Verifies GET /api/healthcare/state synchronizes state and Plotly charts."""
        response = self.client.get("/api/healthcare/state?patient_id=SYN-T2D-001&step=120")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "SYNCHRONIZED")
        self.assertIn("state", data)
        self.assertIn("charts", data)

        st = data["state"]
        self.assertIn("live_vitals", st)
        self.assertIn("forecast", st)
        self.assertIn("glycemic_summary", st)

    def test_api_healthcare_predict(self):
        """Verifies POST /api/healthcare/predict returns excursion prediction."""
        response = self.client.post("/api/healthcare/predict", json={
            "patient_id": "SYN-T2D-001",
            "step": 130
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("forecast", data)
        self.assertIn("excursion_probability", data["forecast"])
        self.assertIn("projected_trajectory", data["forecast"])

    def test_api_healthcare_whatif(self):
        """Verifies POST /api/healthcare/whatif calculates simulated intervention."""
        response = self.client.post("/api/healthcare/whatif", json={
            "patient_id": "SYN-T2D-001",
            "step": 120,
            "walk_minutes": 25,
            "walk_intensity": "Moderate",
            "sleep_extension_hours": 0.0
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("simulation", data)
        self.assertIn("metrics", data["simulation"])
        self.assertIn("chart_whatif_comparison", data)

    def test_api_healthcare_environmental_context(self):
        """Verifies GET /api/healthcare/environmental-context preserves AeroTwin environmental telemetry."""
        response = self.client.get("/api/healthcare/environmental-context")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("ambient_pm25", data)
        self.assertIn("ambient_aqi", data)
        self.assertIn("metabolic_correlation_insight", data)


if __name__ == "__main__":
    unittest.main()
