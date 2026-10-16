"""
AeroTwin - What-If Exposure Simulation Engine (Step 7)
======================================================
Module: model/simulation.py
Target Audience: B.Tech Engineering / Student Level & Digital Twin Engineers

Purpose:
    Implements a transparent, illustrative personal micro-environmental
    exposure simulation model. Computes time-weighted personal inhaled
    PM2.5 exposure and equivalent CPCB advisory tiers across baseline and
    mitigated interventions (indoor HEPA purifiers, N95 masks, and time budgets).

Micro-Environmental Formulation:
    C_daily = (T_indoor * C_indoor + T_transit * C_transit + T_outdoor * C_outdoor) / 24.0
    where:
      C_indoor  = C_ambient * (1 - purifier_efficiency)
      C_transit = C_ambient * (1 - mask_efficiency)
      C_outdoor = C_ambient

Clinical Integrity Notice:
    All scenarios are clearly labeled as HYPOTHETICAL SIMULATIONS based on
    user-defined parameters. They do NOT represent clinically measured health outcomes.
"""

import numpy as np


CPCB_AQI_BREAKPOINTS = [
    (0.0, 30.0, "Good", "#00F5D4", "Minimal impact on health"),
    (30.1, 60.0, "Satisfactory", "#00CFFF", "Minor breathing discomfort to sensitive people"),
    (60.1, 90.0, "Moderate", "#FFD166", "Breathing discomfort to people with asthma and heart disease"),
    (90.1, 120.0, "Poor", "#FF9F1C", "Breathing discomfort to most people on prolonged exposure"),
    (120.1, 250.0, "Very Poor", "#FF6FAE", "Respiratory illness on prolonged exposure"),
    (250.1, 9999.0, "Severe", "#E63946", "Affects healthy people and seriously impacts those with existing diseases")
]


def get_cpcb_tier(pm25_value):
    """
    Maps a PM2.5 concentration (ug/m3) to its official CPCB category,
    hex color, and public advisory description.
    """
    if pm25_value is None or np.isnan(pm25_value):
        return {
            "tier": "Unknown",
            "color": "#9CB0C0",
            "advisory": "Measurement unavailable"
        }

    val = max(0.0, float(pm25_value))
    for low, high, tier, color, advisory in CPCB_AQI_BREAKPOINTS:
        if low <= val <= high:
            return {"tier": tier, "color": color, "advisory": advisory}

    return {"tier": "Severe", "color": "#E63946", "advisory": "Hazardous extreme pollution"}


def simulate_personal_exposure(
    ambient_pm25,
    indoor_hours=18.0,
    transit_hours=2.0,
    purifier_reduction_pct=75.0,
    mask_reduction_pct=80.0,
    custom_reduction_pct=50.0
):
    """
    Simulates time-weighted personal PM2.5 exposure based on user-defined assumptions.
    
    Parameters:
        ambient_pm25 (float): Outdoor ambient PM2.5 concentration in ug/m3.
        indoor_hours (float): Hours spent indoors per day (0 to 24).
        transit_hours (float): Hours spent in transit / outdoors with mask (0 to 24).
        purifier_reduction_pct (float): Assumed indoor particle reduction efficiency % (0 to 100).
        mask_reduction_pct (float): Assumed mask filtration efficiency % (0 to 100).
        custom_reduction_pct (float): User-defined direct exposure reduction % when specific parameters are unavailable (0 to 100).
        
    Returns:
        dict: Detailed comparison between unmitigated baseline and simulated scenarios.
    """
    # Validation & Safe Bounding
    if ambient_pm25 is None or np.isnan(ambient_pm25):
        ambient_pm25 = 100.0
    ambient_pm25 = max(0.0, float(ambient_pm25))

    indoor_hours = max(0.0, min(24.0, float(indoor_hours)))
    transit_hours = max(0.0, min(24.0 - indoor_hours, float(transit_hours)))
    outdoor_hours = max(0.0, 24.0 - (indoor_hours + transit_hours))

    purifier_eff = max(0.0, min(100.0, float(purifier_reduction_pct))) / 100.0
    mask_eff = max(0.0, min(100.0, float(mask_reduction_pct))) / 100.0
    custom_eff = max(0.0, min(100.0, float(custom_reduction_pct))) / 100.0

    # Micro-environmental concentrations
    c_indoor_purified = ambient_pm25 * (1.0 - purifier_eff)
    c_transit_masked = ambient_pm25 * (1.0 - mask_eff)
    c_outdoor_unprotected = ambient_pm25

    # 1. Baseline: Unmitigated outdoor ambient exposure for 24 hours
    baseline_exposure = ambient_pm25
    baseline_tier = get_cpcb_tier(baseline_exposure)

    # 2. Scenario A: Continuous Indoor Purifier (24h indoors with purifier)
    scenario_a_exposure = c_indoor_purified
    scenario_a_tier = get_cpcb_tier(scenario_a_exposure)

    # 3. Scenario B: Integrated Time-Activity Budget (Indoor + Masked Transit + Unprotected Outdoor)
    simulated_daily_exposure = (
        (indoor_hours * c_indoor_purified) +
        (transit_hours * c_transit_masked) +
        (outdoor_hours * c_outdoor_unprotected)
    ) / 24.0
    simulated_tier = get_cpcb_tier(simulated_daily_exposure)

    # 4. Scenario C: User-Defined Direct Reduction Scenario (custom % reduction)
    scenario_c_exposure = ambient_pm25 * (1.0 - custom_eff)
    scenario_c_tier = get_cpcb_tier(scenario_c_exposure)

    # Exposure reductions
    reduction_pct = (
        round(((baseline_exposure - simulated_daily_exposure) / baseline_exposure * 100.0), 1)
        if baseline_exposure > 0 else 0.0
    )
    reduction_ug = round(baseline_exposure - simulated_daily_exposure, 2)

    return {
        "status": "SIMULATION_SUCCESS",
        "is_simulated": True,
        "parameters": {
            "ambient_pm25": round(ambient_pm25, 2),
            "indoor_hours": round(indoor_hours, 1),
            "transit_hours": round(transit_hours, 1),
            "outdoor_hours": round(outdoor_hours, 1),
            "purifier_reduction_pct": round(purifier_eff * 100, 1),
            "purifier_reduction_percent": round(purifier_eff * 100, 1),
            "mask_reduction_pct": round(mask_eff * 100, 1),
            "mask_reduction_percent": round(mask_eff * 100, 1),
            "custom_reduction_pct": round(custom_eff * 100, 1),
            "custom_reduction_percent": round(custom_eff * 100, 1)
        },
        "baseline": {
            "name": "Unmitigated Baseline (100% Outdoor Ambient)",
            "pm25": round(baseline_exposure, 2),
            "tier": baseline_tier["tier"],
            "color": baseline_tier["color"],
            "advisory": baseline_tier["advisory"]
        },
        "scenario_indoor_only": {
            "name": "Continuous Indoor Purifier (24h)",
            "pm25": round(scenario_a_exposure, 2),
            "tier": scenario_a_tier["tier"],
            "color": scenario_a_tier["color"],
            "reduction_pct": round((1.0 - scenario_a_exposure / max(1e-5, baseline_exposure)) * 100, 1),
            "reduction_percent": round((1.0 - scenario_a_exposure / max(1e-5, baseline_exposure)) * 100, 1),
            "advisory": scenario_a_tier["advisory"]
        },
        "scenario_time_weighted": {
            "name": "Personal Time-Activity Daily Routine",
            "pm25": round(simulated_daily_exposure, 2),
            "tier": simulated_tier["tier"],
            "color": simulated_tier["color"],
            "advisory": simulated_tier["advisory"],
            "net_reduction_ug": reduction_ug,
            "net_reduction_pct": reduction_pct,
            "reduction_pct": reduction_pct,
            "reduction_percent": reduction_pct,
            "tier_transition": f"{baseline_tier['tier']} -> {simulated_tier['tier']}"
        },
        "scenario_custom": {
            "name": f"User-Defined Mitigation ({round(custom_eff * 100, 1)}%)",
            "pm25": round(scenario_c_exposure, 2),
            "tier": scenario_c_tier["tier"],
            "color": scenario_c_tier["color"],
            "advisory": scenario_c_tier["advisory"],
            "net_reduction_ug": round(baseline_exposure - scenario_c_exposure, 2),
            "reduction_pct": round(custom_eff * 100, 1),
            "reduction_percent": round(custom_eff * 100, 1),
            "tier_transition": f"{baseline_tier['tier']} -> {scenario_c_tier['tier']}"
        },
        "disclaimer": (
            "Illustrative personal exposure simulation based on user-defined time-activity budgets. "
            "Does not account for indoor source emissions (cooking, smoking) or individual respiratory physiology."
        )
    }


if __name__ == "__main__":
    print("Testing AeroTwin Exposure Simulation Engine...")
    test_run = simulate_personal_exposure(
        ambient_pm25=145.0,
        indoor_hours=18.0,
        transit_hours=2.0,
        purifier_reduction_pct=75.0,
        mask_reduction_pct=80.0
    )
    print("Baseline:", test_run["baseline"])
    print("Simulated Routine:", test_run["scenario_time_weighted"])
