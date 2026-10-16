"""
AeroTwin - Model Training Module (Step 7: Environmental Forecasting)
=====================================================================
Module: model/train.py
Target Audience: B.Tech Engineering / Student Level & ML Engineers

Purpose:
    Trains an empirical, anti-leakage time-series machine learning model
    to forecast Next-Day ambient PM2.5 concentrations (PM2.5 lead 1d).
    Uses a strict chronological train/test split to prevent temporal
    autocorrelation leakage and evaluates against a persistence baseline.

Target Definition:
    - Target: PM2.5(t+1) in ug/m3 (Next-Day PM2.5).
    - Nature: Environmental atmospheric indicator (NOT a clinical patient outcome).

Chronological Split:
    - Training Set:   2015-01-01 to 2018-12-31 (~16,259 rows)
    - Validation Set: 2019-01-01 to 2019-09-30 (~5,209 rows)
    - Test Holdout:   2019-10-01 to 2020-07-01 (~6,614 rows)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor


# Setup Paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "features_multivariate_city_day.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "pm25_forecaster.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "model_metrics.json")


def load_and_prepare_training_data():
    """
    Loads `features_multivariate_city_day.csv`, creates the 1-day ahead
    target (PM2.5 at t+1) strictly per city, and executes a chronological split.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}. Please run model/prepare_data.py first.")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    # Construct Next-Day Target strictly grouped by City (t+1)
    df["target_pm25_1d"] = df.groupby("City")["PM2.5"].shift(-1)
    # Drop terminal days per city where next-day reading is not yet observed
    df = df.dropna(subset=["target_pm25_1d"]).reset_index(drop=True)

    # Select numerical & boolean features, excluding target and unencoded strings
    num_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in ["target_pm25_1d"]]

    # Chronological Split
    split_train_date = pd.Timestamp("2018-12-31")
    split_val_date = pd.Timestamp("2019-09-30")

    train_df = df[df["Date"] <= split_train_date]
    val_df = df[(df["Date"] > split_train_date) & (df["Date"] <= split_val_date)]
    test_df = df[df["Date"] > split_val_date]

    X_train, y_train = train_df[feature_cols], train_df["target_pm25_1d"]
    X_val, y_val = val_df[feature_cols], val_df["target_pm25_1d"]
    X_test, y_test = test_df[feature_cols], test_df["target_pm25_1d"]

    print(f"Data Loaded: {len(df):,} usable time-series records across {df['City'].nunique()} cities.")
    print(f"Features:    {len(feature_cols)} numerical & cyclical features used.")
    print(f"Train Set:   {len(X_train):,} rows ({train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"Val Set:     {len(X_val):,} rows ({val_df['Date'].min().strftime('%Y-%m-%d')} to {val_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"Test Set:    {len(X_test):,} rows ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def evaluate_predictions(y_true, y_pred):
    """Computes MAE, RMSE, and R2 regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}


def train_forecasting_model():
    """
    Trains the Persistence Baseline and the Gradient Boosted Tree Regressor,
    evaluates both on the unseen test set, and exports artifacts.
    """
    print("\n" + "=" * 65)
    print(">>> AEROTWIN - TRAINING NEXT-DAY PM2.5 FORECASTING MODEL")
    print("=" * 65)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_and_prepare_training_data()

    # 1. Baseline Model: Lag-1 Persistence Baseline (PM2.5(t+1) = PM2.5(t))
    print("\n1. Evaluating Persistence Baseline (Naive Benchmark)...")
    base_preds = X_test["PM2.5"]
    base_metrics = evaluate_predictions(y_test, base_preds)
    print(f"   • Baseline Test MAE:  {base_metrics['mae']} ug/m3")
    print(f"   • Baseline Test RMSE: {base_metrics['rmse']} ug/m3")
    print(f"   • Baseline Test R2:   {base_metrics['r2']}")

    # 2. Machine Learning Model: HistGradientBoostingRegressor
    print("\n2. Training Gradient Boosted Tree Regressor...")
    model = HistGradientBoostingRegressor(
        max_iter=150,
        max_depth=6,
        learning_rate=0.08,
        min_samples_leaf=20,
        l2_regularization=0.5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3. Model Evaluation on Validation and Unseen Test Set
    val_preds = model.predict(X_val)
    val_metrics = evaluate_predictions(y_val, val_preds)

    test_preds = model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, test_preds)

    print("\n3. Evaluation Results on Unseen Future Data:")
    print(f"   • Validation MAE:  {val_metrics['mae']} ug/m3 | RMSE: {val_metrics['rmse']} ug/m3 | R2: {val_metrics['r2']}")
    print(f"   • Test Set MAE:    {test_metrics['mae']} ug/m3 | RMSE: {test_metrics['rmse']} ug/m3 | R2: {test_metrics['r2']}")

    # Check improvement over baseline
    rmse_improvement = round((base_metrics["rmse"] - test_metrics["rmse"]) / base_metrics["rmse"] * 100, 2)
    print(f"   • RMSE Reduction vs Baseline: {rmse_improvement}%")

    # 4. Save Model Artifacts
    artifact_payload = {
        "model": model,
        "feature_cols": feature_cols,
        "target_name": "target_pm25_1d",
        "target_description": "Next-Day Ambient PM2.5 (ug/m3)",
        "model_type": "HistGradientBoostingRegressor"
    }
    joblib.dump(artifact_payload, MODEL_PATH)

    metrics_payload = {
        "model_type": "HistGradientBoostingRegressor",
        "target": "Next-Day PM2.5 (ug/m3)",
        "features_count": len(feature_cols),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "train_period": "2015-01-01 to 2018-12-31",
        "val_period": "2019-01-01 to 2019-09-30",
        "test_period": "2019-10-01 to 2020-07-01",
        "baseline_persistence": base_metrics,
        "model_validation": val_metrics,
        "model_test": test_metrics,
        "rmse_improvement_pct": rmse_improvement,
        "limitations": (
            "Model forecasts ambient atmospheric PM2.5 concentrations based on historical time-series. "
            "It does NOT predict clinical health outcomes, patient admissions, or physiological symptoms."
        )
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nModel artifact saved to:   {MODEL_PATH}")
    print(f"Metrics metadata saved to: {METRICS_PATH}")
    print("=" * 65 + "\n")
    return model, metrics_payload


if __name__ == "__main__":
    train_forecasting_model()
