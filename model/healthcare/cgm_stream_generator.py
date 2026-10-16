"""
AeroTwin Healthcare Digital Twin - Wearable & CGM Time-Series Generator
========================================================================
Module: model/healthcare/cgm_stream_generator.py
Purpose: Generates high-fidelity, deterministic 5-minute sampled continuous physiological
         wearable streams (CGM, Heart Rate, HRV, Steps, Sleep Stage, Sensor Quality)
         for synthetic Type 2 Diabetes patients.

Physiological Dynamics:
    1. Basal Glucose Dynamics & Hepatic Glucose Output
    2. Postprandial Glucose Excursions (Carbohydrate Intake Peaks)
    3. Insulin-Independent Exercise Glucose Clearance (dG/dt drops with activity)
    4. Circadian Rhythm & Dawn Phenomenon
    5. Autonomic Nervous System Telemetry (Inverse HR vs HRV SDNN coupling)
    6. Sensor Signal Quality Flags (Simulating subcutaneous sensor noise/dropout)
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model.healthcare.ehr_generator import load_synthetic_patients


def generate_patient_cgm_stream(
    patient_ehr,
    start_date="2026-03-01 00:00:00",
    days=7,
    seed=42
):
    """
    Generates a 5-minute resolution time-series dataframe for a specific patient.
    
    Parameters:
        patient_ehr (dict): Patient EHR profile.
        start_date (str): Timestamp for stream inception.
        days (int): Number of days to simulate (7 days = 2,016 samples).
        seed (int): Deterministic random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Timestamped multimodal physiological telemetry.
    """
    np.random.seed(seed)
    patient_id = patient_ehr["patient_id"]
    hba1c = patient_ehr["laboratory_results"]["hba1c_pct"]
    base_fbg = patient_ehr["laboratory_results"]["fasting_blood_glucose_mg_dl"]
    base_hr = patient_ehr["vitals_baseline"]["resting_heart_rate_bpm"]
    base_hrv = patient_ehr["vitals_baseline"]["baseline_hrv_sdnn_ms"]

    # Overall glucose baseline derived from clinical HbA1c conversion: eAG = 28.7 * HbA1c - 46.7
    eag = 28.7 * hba1c - 46.7

    # Total 5-minute intervals (288 steps per 24 hours)
    total_steps = days * 288
    base_time = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")

    timestamps = [base_time + timedelta(minutes=5 * i) for i in range(total_steps)]

    glucose_readings = []
    heart_rates = []
    hrv_sdnn = []
    step_counts = []
    sleep_stages = []
    sensor_qualities = []
    activity_states = []

    # State variables for continuous physiological simulation
    current_glucose = base_fbg
    glucose_velocity = 0.0

    # Meal schedule & carb loads (hour, std, carb_factor)
    meals = [
        {"name": "Breakfast", "hour": 8.0, "duration_h": 2.5, "peak_add": 55.0 * (hba1c / 7.5)},
        {"name": "Lunch", "hour": 13.0, "duration_h": 3.0, "peak_add": 65.0 * (hba1c / 7.5)},
        {"name": "Snack", "hour": 17.5, "duration_h": 1.5, "peak_add": 30.0 * (hba1c / 7.5)},
        {"name": "Dinner", "hour": 20.5, "duration_h": 3.0, "peak_add": 60.0 * (hba1c / 7.5)}
    ]

    for i, ts in enumerate(timestamps):
        hour_of_day = ts.hour + ts.minute / 60.0
        day_of_sim = i // 288

        # 1. Sleep vs Awake State (11:00 PM to 6:30 AM = sleep)
        is_sleep = (hour_of_day >= 23.0) or (hour_of_day < 6.5)
        if is_sleep:
            if hour_of_day >= 23.0 or hour_of_day < 2.0:
                stage = "Light Sleep"
            elif 2.0 <= hour_of_day < 4.5:
                stage = "Deep Sleep"
            else:
                stage = "REM Sleep"
            current_activity = "Sleep"
            steps = 0
        else:
            stage = "Awake"
            # Random walking/routine movements throughout the day
            # Scheduled active walk (e.g. 7:00-7:45 AM or 18:30-19:15 PM)
            if 7.0 <= hour_of_day < 7.75:
                current_activity = "Brisk Walking"
                steps = int(np.random.normal(380, 40))
            elif 18.5 <= hour_of_day < 19.25 and patient_id in ["SYN-T2D-002", "SYN-T2D-004"]:
                current_activity = "Moderate Cardio"
                steps = int(np.random.normal(450, 50))
            elif 9.0 <= hour_of_day < 18.0:
                # Office desk hours with occasional movement
                if np.random.rand() < 0.25:
                    current_activity = "Light Activity"
                    steps = int(np.random.normal(85, 25))
                else:
                    current_activity = "Sedentary"
                    steps = int(np.random.normal(10, 5))
            else:
                current_activity = "Resting/Relaxing"
                steps = int(np.random.normal(25, 10))

        steps = max(0, steps)

        # 2. Meal Excursion Influence
        meal_glucose_impulse = 0.0
        for m in meals:
            time_since_meal = (hour_of_day - m["hour"]) % 24.0
            if 0.0 <= time_since_meal <= m["duration_h"]:
                # Asymmetric gamma-like absorption curve
                t = time_since_meal / m["duration_h"]
                absorption_curve = math.sin(t * math.pi) ** 1.8
                meal_glucose_impulse += m["peak_add"] * absorption_curve

        # 3. Exercise Glucose Clearance (Active steps reduce glucose)
        exercise_clearance = 0.0
        if steps > 150:
            exercise_clearance = (steps / 300.0) * 8.5  # Clear 4-15 mg/dL per high active epoch

        # 4. Circadian Basal Drift & Dawn Phenomenon (Rise between 4:30 AM and 7:30 AM)
        dawn_phenomenon = 0.0
        if 4.5 <= hour_of_day <= 7.5:
            dawn_t = (hour_of_day - 4.5) / 3.0
            dawn_phenomenon = 18.0 * math.sin(dawn_t * math.pi)

        # 5. Continuous Autoregressive Glucose Update
        target_glucose = eag + meal_glucose_impulse + dawn_phenomenon - exercise_clearance
        # Patient specific noise
        physio_noise = np.random.normal(0, 1.8)
        # Gentle mean-reversion with inertia
        current_glucose = current_glucose * 0.94 + target_glucose * 0.06 + physio_noise
        current_glucose = max(55.0, min(380.0, current_glucose))

        # 6. Cardiac Telemetry (Coupled to activity and metabolic stress)
        # Increased glucose and steps elevate HR; resting HR in deep sleep drops
        if is_sleep:
            hr = base_hr - 10.0 + np.random.normal(0, 2.0)
            hrv = base_hrv + 12.0 + np.random.normal(0, 3.0)
        else:
            activity_hr_bonus = (steps / 400.0) * 35.0
            glycemic_stress_hr = max(0.0, (current_glucose - 140.0) / 40.0) * 3.0
            hr = base_hr + activity_hr_bonus + glycemic_stress_hr + np.random.normal(0, 3.0)
            hrv = max(15.0, base_hrv - (activity_hr_bonus * 0.4) - (glycemic_stress_hr * 0.8) + np.random.normal(0, 3.5))

        hr = max(45.0, min(180.0, hr))
        hrv = max(10.0, min(120.0, hrv))

        # 7. Sensor Quality Assurance Flag
        # Rare transient noise simulating physical pressure or interstitial lag
        rand_qual = np.random.rand()
        if rand_qual < 0.975:
            quality = "Optimal"
        elif rand_qual < 0.992:
            quality = "Sensor Calibration / Minor Noise"
            current_glucose += np.random.normal(0, 6.0)
        else:
            quality = "Transient Signal Compression"
            current_glucose -= np.random.normal(8.0, 3.0)

        glucose_readings.append(round(float(current_glucose), 1))
        heart_rates.append(round(float(hr), 1))
        hrv_sdnn.append(round(float(hrv), 1))
        step_counts.append(int(steps))
        sleep_stages.append(stage)
        sensor_qualities.append(quality)
        activity_states.append(current_activity)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "glucose_mg_dl": glucose_readings,
        "heart_rate_bpm": heart_rates,
        "hrv_sdnn_ms": hrv_sdnn,
        "step_count": step_counts,
        "activity_state": activity_states,
        "sleep_stage": sleep_stages,
        "sensor_quality": sensor_qualities,
        "data_provenance": "Synthetic continuous 5-minute physiological stream"
    })

    return df


def generate_all_patient_streams(output_dir=None, days=7):
    """
    Generates and saves the wearable time-series dataset for all synthetic patients.
    """
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(base_dir, "data", "healthcare")

    os.makedirs(output_dir, exist_ok=True)
    patients = load_synthetic_patients()

    all_dfs = []
    for idx, p in enumerate(patients):
        # Vary seed deterministically per patient
        seed = 100 + idx * 42
        df_p = generate_patient_cgm_stream(p, days=days, seed=seed)
        all_dfs.append(df_p)

        single_patient_file = os.path.join(output_dir, f"cgm_stream_{p['patient_id']}.csv")
        df_p.to_csv(single_patient_file, index=False)
        print(f"[CGM Generator] Saved {len(df_p)} records for {p['patient_id']} -> {single_patient_file}")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_file = os.path.join(output_dir, "cgm_streams_all_patients.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"[CGM Generator] Combined dataset saved: {len(combined_df)} total records -> {combined_file}")

    return combined_df


if __name__ == "__main__":
    generate_all_patient_streams()
