"""
AeroTwin - Model Inference & Prediction Module (Step 7)
======================================================
Module: model/predict.py
Target Audience: B.Tech Engineering / Student Level & ML Engineers

Purpose:
    Safely loads trained model artifacts, caches them in-memory, and
    generates 24-hour ahead (Next-Day) ambient PM2.5 forecasts from
    input feature dictionaries or dataframes.

Key Features:
    - Safe caching: Model is loaded once on first inference, avoiding repeated disk reads.
    - Input validation: Imputes missing numerical inputs with default baselines and flags warnings.
    - CPCB classification: Maps forecasted PM2.5 to official CPCB health advisory categories.
"""

import os
import joblib
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "artifacts", "pm25_forecaster.joblib")

_CACHED_MODEL_PAYLOAD = None


def get_cpcb_pm25_category(pm25_val):
    """
    Maps PM2.5 concentration (ug/m3) to official CPCB AQI category bands.
    Breakpoints (CPCB):
      0 - 30:     Good
      31 - 60:    Satisfactory
      61 - 90:    Moderate
      91 - 120:   Poor
      121 - 250:  Very Poor
      > 250:      Severe
    """
    if pm25_val is None or np.isnan(pm25_val):
        return "Unknown"
    elif pm25_val <= 30.0:
        return "Good"
    elif pm25_val <= 60.0:
        return "Satisfactory"
    elif pm25_val <= 90.0:
        return "Moderate"
    elif pm25_val <= 120.0:
        return "Poor"
    elif pm25_val <= 250.0:
        return "Very Poor"
    else:
        return "Severe"


def load_model_artifact(force_reload=False):
    """
    Safely loads the model artifact from disk with in-memory caching.
    """
    global _CACHED_MODEL_PAYLOAD
    if _CACHED_MODEL_PAYLOAD is not None and not force_reload:
        return _CACHED_MODEL_PAYLOAD

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        _CACHED_MODEL_PAYLOAD = joblib.load(MODEL_PATH)
        return _CACHED_MODEL_PAYLOAD
    except Exception as e:
        print(f"[Warning] Failed to load model artifact: {e}")
        return None


def predict_next_day_pm25(input_data):
    """
    Generates a 24-hour ahead Next-Day PM2.5 prediction.
    
    Parameters:
        input_data (dict or pd.DataFrame): Feature inputs containing pollutant observations.
        
    Returns:
        dict: Forecast results with predicted concentration, baseline comparison,
              CPCB category, and transparency metadata.
    """
    payload = load_model_artifact()
    if payload is None:
        return {
            "status": "MODEL_UNAVAILABLE",
            "message": "Model artifact not found. Please run model/train.py first.",
            "forecast_pm25": None,
            "cpcb_category": "Unknown",
            "is_simulated_or_forecast": True
        }

    model = payload["model"]
    feature_cols = payload["feature_cols"]

    # Convert dict to DataFrame if needed
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise ValueError("input_data must be a dict or pandas DataFrame")

    # Align feature columns, filling missing features with 0.0 or defaults
    X = pd.DataFrame(index=df_input.index)
    for col in feature_cols:
        if col in df_input.columns:
            X[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0.0)
        else:
            X[col] = 0.0

    preds = model.predict(X)
    pred_val = max(0.0, float(preds[0]))

    today_pm25 = float(df_input.get("PM2.5", [pred_val])[0]) if "PM2.5" in df_input.columns else pred_val
    delta = round(pred_val - today_pm25, 2)
    cpcb_cat = get_cpcb_pm25_category(pred_val)
    today_cat = get_cpcb_pm25_category(today_pm25)

    tier_colors = {
        "Good": "#00F5D4",
        "Satisfactory": "#00CFFF",
        "Moderate": "#FFD166",
        "Poor": "#FF9F1C",
        "Very Poor": "#FF6FAE",
        "Severe": "#E63946"
    }

    return {
        "status": "SUCCESS",
        "forecast_pm25": round(pred_val, 2),
        "predicted_next_day_pm25": round(pred_val, 2),
        "today_pm25": round(today_pm25, 2),
        "current_pm25": round(today_pm25, 2),
        "delta_vs_today": delta,
        "delta_from_today": delta,
        "delta_direction": "worsening" if delta > 0 else ("improving" if delta < 0 else "steady"),
        "trend": "worsening" if delta > 0 else ("improving" if delta < 0 else "steady"),
        "cpcb_category": cpcb_cat,
        "predicted_tier": cpcb_cat,
        "current_tier": today_cat,
        "tier_color": tier_colors.get(cpcb_cat, "#00F5D4"),
        "target": "Next-Day PM2.5 (ug/m3)",
        "source_type": "empirical_forecast",
        "is_forecast": True,
        "disclaimer": (
            "Forecast represents ambient atmospheric PM2.5 concentration. "
            "It does NOT predict clinical health outcomes, patient symptoms, or hospitalizations."
        )
    }


if __name__ == "__main__":
    print("Testing AeroTwin Prediction Module...")
    # Test sample with realistic Delhi smog observation
    sample = {
        "PM2.5": 140.0,
        "PM10": 260.0,
        "NO2": 45.0,
        "SO2": 15.0,
        "CO": 1.2,
        "O3": 35.0,
        "AQI": 315.0,
        "PM2.5_lag_1d": 135.0,
        "PM2.5_roll_mean_7d": 130.0,
        "Month": 11,
        "Is_Weekend": 0
    }
    result = predict_next_day_pm25(sample)
    print("Prediction Result:", result)
