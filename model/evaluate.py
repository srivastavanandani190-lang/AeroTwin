"""
AeroTwin - Model Evaluation Module (Step 7)
===========================================
Module: model/evaluate.py
Target Audience: B.Tech Engineering / Student Level & ML Engineers

Purpose:
    Performs comprehensive evaluation of the trained environmental forecasting
    model against the test holdout set (2019-10-01 to 2020-07-01).
    Computes both continuous regression metrics (MAE, RMSE, R2) and categorical
    CPCB AQI tier classification metrics (Accuracy, Macro F1, Confusion Matrix).
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    accuracy_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "features_multivariate_city_day.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "artifacts", "pm25_forecaster.joblib")
EVAL_RESULTS_PATH = os.path.join(BASE_DIR, "model", "artifacts", "evaluation_summary.json")

CPCB_TIERS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def get_cpcb_category(val):
    """Categorizes PM2.5 concentration into official CPCB tiers."""
    if val <= 30.0: return "Good"
    elif val <= 60.0: return "Satisfactory"
    elif val <= 90.0: return "Moderate"
    elif val <= 120.0: return "Poor"
    elif val <= 250.0: return "Very Poor"
    else: return "Severe"


def run_comprehensive_evaluation():
    """Evaluates the trained model on the chronological test holdout set."""
    print("\n" + "=" * 65)
    print(">>> AEROTWIN - MODEL EVALUATION MODULE")
    print("=" * 65)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Train model first.")

    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    # Load data
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    df["target_pm25_1d"] = df.groupby("City")["PM2.5"].shift(-1)
    df = df.dropna(subset=["target_pm25_1d"]).reset_index(drop=True)

    # Test holdout set: 2019-10-01 to 2020-07-01
    test_df = df[df["Date"] >= "2019-10-01"].copy()
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["target_pm25_1d"].astype(float)

    # Predictions
    preds = model.predict(X_test)
    preds = np.clip(preds, 0.0, None)
    test_df["predicted_pm25"] = preds

    # 1. Continuous Regression Metrics
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    max_err = float(np.max(np.abs(y_test - preds)))

    # Persistence Baseline
    base_preds = test_df["PM2.5"]
    base_mae = float(mean_absolute_error(y_test, base_preds))
    base_rmse = float(np.sqrt(mean_squared_error(y_test, base_preds)))
    base_r2 = float(r2_score(y_test, base_preds))

    print("\n1. Regression Metrics on Unseen Test Set (6,614 days across 26 cities):")
    print(f"   • ML Model:    MAE = {mae:.2f} ug/m3 | RMSE = {rmse:.2f} ug/m3 | R2 = {r2:.4f}")
    print(f"   • Baseline:    MAE = {base_mae:.2f} ug/m3 | RMSE = {base_rmse:.2f} ug/m3 | R2 = {base_r2:.4f}")
    print(f"   • Max Error:   {max_err:.2f} ug/m3")

    # 2. CPCB Category Classification Metrics
    actual_cats = y_test.apply(get_cpcb_category)
    pred_cats = pd.Series(preds).apply(get_cpcb_category)

    cat_acc = float(accuracy_score(actual_cats, pred_cats))
    cm = confusion_matrix(actual_cats, pred_cats, labels=CPCB_TIERS)

    print(f"\n2. CPCB Health Tier Classification Agreement:")
    print(f"   • Overall Category Accuracy: {cat_acc * 100:.2f}%")
    print(f"   • Confusion Matrix (CPCB Tiers: {CPCB_TIERS}):\n{cm}")

    eval_summary = {
        "test_records": len(test_df),
        "test_period": "2019-10-01 to 2020-07-01",
        "regression_metrics": {
            "model_mae": round(mae, 2),
            "model_rmse": round(rmse, 2),
            "model_r2": round(r2, 4),
            "baseline_mae": round(base_mae, 2),
            "baseline_rmse": round(base_rmse, 2),
            "baseline_r2": round(base_r2, 4)
        },
        "classification_agreement": {
            "cpcb_category_accuracy": round(cat_acc, 4),
            "tiers": CPCB_TIERS,
            "confusion_matrix": cm.tolist()
        },
        "scientific_integrity_notice": (
            "Evaluation measures empirical accuracy in predicting ambient atmospheric PM2.5. "
            "It does NOT evaluate clinical risk, patient physiological outcomes, or medical diagnostic efficacy."
        )
    }

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nEvaluation summary saved to: {EVAL_RESULTS_PATH}")
    print("=" * 65 + "\n")
    return eval_summary


if __name__ == "__main__":
    run_comprehensive_evaluation()
