"""
AeroTwin Healthcare Digital Twin - What-If Simulation Engine
=============================================================
Module: model/healthcare/whatif_engine.py
Purpose: Simulates hypothetical behavioral and physical activity interventions
         (e.g., post-meal walking bouts, sedentary postprandial sitting, sleep extension)
         on glucose trajectory and excursion risk.

Clinical & Scientific Integrity Notice:
    All scenario projections are clearly labeled as HYPOTHETICAL BEHAVIORAL SIMULATIONS.
    They do NOT constitute medical prescriptions or treatment recommendations.
    Never alter prescribed medication or insulin regimens based on simulations.
"""

import math
import numpy as np


def simulate_glucose_intervention(
    baseline_glucose_trajectory,
    current_glucose,
    post_meal_walk_minutes=0.0,
    walk_intensity="Moderate",
    sleep_extension_hours=0.0,
    carbohydrate_offset_pct=0.0
):
    """
    Simulates the physiological effect of behavioral interventions on a 2-hour glucose curve.
    
    Parameters:
        baseline_glucose_trajectory (list of float): 24 points (5-min intervals) of forecasted glucose.
        current_glucose (float): Current glucose reading in mg/dL.
        post_meal_walk_minutes (float): Additional walking duration post-meal (0 to 60 mins).
        walk_intensity (str): "Light" (1.0x), "Moderate" (1.6x), or "Brisk" (2.2x).
        sleep_extension_hours (float): Hours of additional restorative sleep (-2.0 to +3.0).
        carbohydrate_offset_pct (float): Dietary carbohydrate moderation % (0 to 50%).
        
    Returns:
        dict: Detailed comparison of baseline vs simulated trajectory, metrics, and ADA Time-in-Range.
    """
    if not baseline_glucose_trajectory:
        baseline_glucose_trajectory = [current_glucose] * 24

    base_curve = [float(x) for x in baseline_glucose_trajectory]
    simulated_curve = []

    # Walking intensity factor (GLUT4 translocation & muscle glucose clearance)
    intensity_multipliers = {
        "Light": 0.35,      # Gentle stroll
        "Moderate": 0.65,   # Standard brisk walk
        "Brisk": 0.95       # Fast-paced exercise
    }
    int_mult = intensity_multipliers.get(walk_intensity, 0.65)

    # Effective activity steps
    walk_mins = max(0.0, min(60.0, float(post_meal_walk_minutes)))
    active_steps_count = int(walk_mins * 90)  # ~90 steps/minute

    # Sleep cortisol / dawn phenomenon modulation
    sleep_delta = max(-2.0, min(3.0, float(sleep_extension_hours)))
    sleep_attenuation = sleep_delta * 4.5  # Restorative sleep reduces morning cortisol-driven rise

    carb_reduction = max(0.0, min(50.0, float(carbohydrate_offset_pct))) / 100.0

    for step_idx, g_base in enumerate(base_curve):
        elapsed_minutes = (step_idx + 1) * 5.0

        # Physical activity clearance: rises during exercise and sustains through increased GLUT4 expression
        activity_drop = 0.0
        if walk_mins > 0:
            if elapsed_minutes <= walk_mins:
                # Active muscle contraction-mediated uptake
                fraction = elapsed_minutes / max(5.0, walk_mins)
                activity_drop = 28.0 * int_mult * fraction
            else:
                # Sustained post-exercise insulin sensitivity (gradual exponential decay over 2 hours)
                decay = math.exp(-0.015 * (elapsed_minutes - walk_mins))
                activity_drop = 28.0 * int_mult * decay

        # Dietary carbohydrate attenuation
        carb_drop = max(0.0, (g_base - current_glucose)) * carb_reduction

        # Total simulated glucose
        sim_val = g_base - activity_drop - carb_drop - sleep_attenuation
        # Ensure physiological floor to prevent hypothetical hypoglycemia artifact
        sim_val = max(70.0, round(float(sim_val), 1))
        simulated_curve.append(sim_val)

    # Compute key comparison statistics
    base_peak = round(float(np.max(base_curve)), 1)
    sim_peak = round(float(np.max(simulated_curve)), 1)
    peak_reduction = round(base_peak - sim_peak, 1)

    base_final = round(float(base_curve[-1]), 1)
    sim_final = round(float(simulated_curve[-1]), 1)

    # Time-in-Range (TIR: 70 - 180 mg/dL) as defined by ADA/EASD consensus
    base_in_range_pct = round(float(np.mean([(70.0 <= x <= 180.0) for x in base_curve])) * 100.0, 1)
    sim_in_range_pct = round(float(np.mean([(70.0 <= x <= 180.0) for x in simulated_curve])) * 100.0, 1)
    tir_gain_pct = round(sim_in_range_pct - base_in_range_pct, 1)

    base_excursion = bool(base_peak >= 180.0)
    sim_excursion = bool(sim_peak >= 180.0)

    # Pre-calculated presets for quick scenario exploration
    preset_scenarios = {
        "sedentary_postprandial": {
            "name": "Prolonged Postprandial Inactivity (0 min walk)",
            "peak_mg_dl": base_peak,
            "tir_pct": base_in_range_pct,
            "excursion": base_excursion
        },
        "light_walk_15m": {
            "name": "15-Min Light Stroll (+1,350 steps)",
            "peak_mg_dl": round(max(70.0, base_peak - 14.0 * int_mult), 1),
            "tir_pct": min(100.0, round(base_in_range_pct + 18.0, 1)),
            "excursion": (base_peak - 14.0 * int_mult) >= 180.0
        },
        "brisk_walk_30m": {
            "name": "30-Min Brisk Post-Meal Walk (+2,700 steps)",
            "peak_mg_dl": round(max(70.0, base_peak - 24.0 * int_mult), 1),
            "tir_pct": min(100.0, round(base_in_range_pct + 36.0, 1)),
            "excursion": (base_peak - 24.0 * int_mult) >= 180.0
        }
    }

    return {
        "status": "SIMULATION_SUCCESS",
        "parameters": {
            "post_meal_walk_minutes": walk_mins,
            "walk_intensity": walk_intensity,
            "estimated_steps_added": active_steps_count,
            "sleep_extension_hours": sleep_delta,
            "carbohydrate_offset_pct": round(carb_reduction * 100.0, 1)
        },
        "trajectories": {
            "baseline_curve": base_curve,
            "simulated_curve": simulated_curve,
            "time_steps_minutes": [(i + 1) * 5 for i in range(24)]
        },
        "metrics": {
            "baseline_peak_mg_dl": base_peak,
            "simulated_peak_mg_dl": sim_peak,
            "peak_glucose_reduction_mg_dl": peak_reduction,
            "baseline_final_mg_dl": base_final,
            "simulated_final_mg_dl": sim_final,
            "baseline_tir_pct": base_in_range_pct,
            "simulated_tir_pct": sim_in_range_pct,
            "tir_improvement_pct": tir_gain_pct,
            "baseline_excursion_predicted": base_excursion,
            "simulated_excursion_predicted": sim_excursion
        },
        "presets": preset_scenarios,
        "clinical_disclaimer": (
            "Hypothetical simulation based on physiological modeling of non-insulin-mediated "
            "exercise glucose uptake (GLUT4). Results are illustrative for educational scenarios and "
            "do NOT guarantee clinical response. Do not adjust medical treatments based on simulations."
        )
    }


if __name__ == "__main__":
    dummy_base = [140.0 + i * 2.5 for i in range(24)]
    res = simulate_glucose_intervention(dummy_base, 140.0, post_meal_walk_minutes=25, walk_intensity="Moderate")
    print(f"Simulation test: Peak reduction = {res['metrics']['peak_glucose_reduction_mg_dl']} mg/dL, TIR Gain = {res['metrics']['tir_improvement_pct']}%")
