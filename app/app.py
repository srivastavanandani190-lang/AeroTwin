"""
AeroTwin - Vibrant Black Crystal Air Quality & Health Analytics Dashboard
==========================================================================
Author: AeroTwin Engineering Team
Visual Direction: Vibrant Black Crystal (Obsidian Glass + Neon Lighting)

Palette:
    - Main background: #030609 (Deep Obsidian Black)
    - Sidebar and main panels: #07131C (Dark Crystal Navy)
    - Elevated cards: #0B1822 (Obsidian Card Surface)
    - Borders: #173744 (Subtle Teal Crystal Border)
    - Neon aqua: #00F5D4
    - Electric cyan: #00CFFF
    - Electric purple: #A78BFA
    - Neon pink: #FF6FAE
    - Crystal gold: #FFD166
    - Primary text: #F3F8FC
    - Secondary text: #9CB0C0
"""

import os
import json
import pandas as pd
import numpy as np

try:
    from flask import Flask, render_template, request, send_file, abort
except ImportError:
    Flask = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.utils
except ImportError:
    go = None


import sys

# Setup Paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Lightweight .env loader (zero-dependency standard library)
def _load_env_file():
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Import Step 7 Simulation & Prediction Engines
try:
    from model.simulation import simulate_personal_exposure, get_cpcb_tier
    from model.predict import predict_next_day_pm25, load_model_artifact
except ImportError:
    simulate_personal_exposure = None
    predict_next_day_pm25 = None

# Import Healthcare Digital Twin Services & Engines
try:
    from model.healthcare.ehr_generator import load_synthetic_patients
    from model.healthcare.twin_service import (
        get_digital_twin_state_at_step,
        get_patient_profile,
        get_model_evaluation_metrics_summary,
        get_environmental_metabolic_context
    )
    from model.healthcare.whatif_engine import simulate_glucose_intervention
    from model.healthcare.twin_charts import (
        create_cgm_history_chart,
        create_forecast_trajectory_chart,
        create_whatif_comparison_chart,
        create_tir_donut_chart,
        create_confusion_matrix_chart
    )
except ImportError:
    load_synthetic_patients = None
    get_digital_twin_state_at_step = None
    get_patient_profile = None
    get_model_evaluation_metrics_summary = None
    get_environmental_metabolic_context = None
    simulate_glucose_intervention = None
    create_cgm_history_chart = None
    create_forecast_trajectory_chart = None
    create_whatif_comparison_chart = None
    create_tir_donut_chart = None
    create_confusion_matrix_chart = None

# Color Palette Constants (Premium Black & Deep Forest Theme)
THEME_COLORS = {
    # Backgrounds
    "bg_main": "#020405",           # Main page background (pure near-black)
    "bg_content": "#030608",        # Main content background
    "surface_sidebar": "#04090B",   # Sidebar background
    "surface_filter": "#07120F",    # Filter panel background
    "surface_card": "#091510",      # Standard card background (forest green surface tint)
    "surface_elevated": "#0D1D16",  # Elevated card background
    "surface_hover": "#12271D",     # Hovered panel / active nav background
    "input_bg": "#050B0C",          # Input & chart plot background
    "plot_bg": "#050B0C",           # Chart plotting background
    "surface_secondary": "#0D1D16",

    # Accents
    "accent_jade": "#54E6A5",       # Primary jade green
    "accent_emerald": "#27C98B",    # Bright emerald
    "accent_ice": "#72C9F0",        # Ice blue
    "accent_cyan": "#25D9E8",       # Cyan
    "accent_gold": "#F5C96A",       # Soft gold
    "accent_purple": "#A78BFA",     # Purple (PM10 / Category)
    "accent_pink": "#FF7198",       # Pink (Duplicate / Category)

    # Backwards compatibility aliases
    "accent_aqua": "#54E6A5",

    # Text
    "text_primary": "#F2F7F5",      # Primary text
    "text_secondary": "#A5B7AF",    # Secondary text
    "text_muted": "#758A80",        # Muted text
    "text_main": "#F2F7F5",

    # Borders & Dividers
    "border_standard": "#193328",   # Standard border
    "border_highlight": "#245640",  # Highlight border
    "border_subtle": "#12241C",     # Subtle divider
    "border_color": "#193328",
    "gridline_color": "#172A22",
    "axis_color": "#244136",
}

# In-memory dataframe cache for high-speed responsiveness
_DATA_CACHE = {}


def get_available_datasets():
    """
    Scans `data/processed/` for ready-to-use cleaned CSV files.
    """
    if not os.path.exists(PROCESSED_DIR):
        return []
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]
    return sorted(files)


def load_dataset(dataset_name):
    """
    Loads and caches a cleaned dataset from `data/processed/`.
    
    Parameters:
        dataset_name (str): CSV filename
        
    Returns:
        pd.DataFrame or None
    """
    if dataset_name in _DATA_CACHE:
        return _DATA_CACHE[dataset_name].copy()

    filepath = os.path.join(PROCESSED_DIR, dataset_name)
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, low_memory=False)

    # Standardize Date columns if present
    for date_col in ["Date", "Datetime"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    _DATA_CACHE[dataset_name] = df
    return df.copy()


def calculate_metrics(df):
    """
    Calculates summary metrics dynamically from the active dataframe.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    date_range_str = "N/A"
    for date_col in ["Date", "Datetime"]:
        if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            valid_dates = df[date_col].dropna()
            if not valid_dates.empty:
                min_d = valid_dates.min().strftime("%b %Y")
                max_d = valid_dates.max().strftime("%b %Y")
                date_range_str = f"{min_d} — {max_d}"
            break

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "date_range_str": date_range_str,
    }


def get_column_profiles(df):
    """
    Computes data quality and datatype profile for each column.
    """
    profiles = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        unique_count = int(df[col].nunique())

        stat_summary = ""
        if pd.api.types.is_numeric_dtype(df[col]):
            valid_series = df[col].dropna()
            if not valid_series.empty:
                stat_summary = f"Min: {valid_series.min():.1f} | Avg: {valid_series.mean():.1f} | Max: {valid_series.max():.1f}"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            valid_dates = df[col].dropna()
            if not valid_dates.empty:
                stat_summary = f"{valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}"
        else:
            sample_vals = df[col].dropna().unique()[:3]
            stat_summary = ", ".join(map(str, sample_vals))
            if len(df[col].dropna().unique()) > 3:
                stat_summary += "..."

        profiles.append({
            "name": col,
            "dtype": dtype_str,
            "non_null": len(df) - null_count,
            "missing": null_count,
            "unique": unique_count,
            "stats": stat_summary
        })
    return profiles


def create_plotly_theme_layout(title=""):
    """
    Returns a unified Plotly layout matching the Premium Black & Deep Forest aesthetic.
    Paper: #030608, Plot: #050B0C, Text: #EAF4EF, Grid: #172A22, Axis: #244136
    """
    return dict(
        title=dict(
            text=title,
            font=dict(family="Inter, -apple-system, sans-serif", size=13, color="#EAF4EF"),
            x=0.02,
            y=0.96
        ),
        paper_bgcolor="#030608",
        plot_bgcolor="#050B0C",
        font=dict(family="Inter, -apple-system, sans-serif", color="#A5B7AF", size=11),
        margin=dict(l=45, r=25, t=35, b=35),
        xaxis=dict(
            showgrid=True,
            gridcolor="#172A22",
            linecolor="#244136",
            zeroline=False,
            color="#A5B7AF",
            tickfont=dict(size=10, color="#758A80")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#172A22",
            linecolor="#244136",
            zeroline=False,
            color="#A5B7AF",
            tickfont=dict(size=10, color="#758A80")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#EAF4EF"),
            bgcolor="rgba(0,0,0,0)"
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#07120F",
            bordercolor="#245640",
            font=dict(color="#F2F7F5", size=12)
        )
    )


def generate_dashboard_charts(df, custom_type="scatter", custom_x="PM2.5", custom_y="PM10"):
    """
    Generates JSON specifications for rich, interactive Plotly charts supported by available columns.
    """
    charts = {}

    if df.empty or go is None:
        return charts

    date_col = next((c for c in ["Date", "Datetime"] if c in df.columns), None)

    # ----------------------------------------------------
    # Special Handling for Stations Metadata (cleaned_stations.csv)
    # ----------------------------------------------------
    if "StationId" in df.columns and "AQI" not in df.columns and "PM2.5" not in df.columns:
        if "State" in df.columns:
            state_counts = df["State"].value_counts().head(12).reset_index()
            state_counts.columns = ["State", "Count"]
            fig_state = go.Figure(data=[go.Bar(
                x=state_counts["State"],
                y=state_counts["Count"],
                marker=dict(
                    color=THEME_COLORS["accent_aqua"],
                    line=dict(color=THEME_COLORS["border_color"], width=1)
                ),
                text=state_counts["Count"],
                textposition="auto",
                textfont=dict(color=THEME_COLORS["bg_main"], size=11, family="JetBrains Mono, monospace")
            )])
            layout_state = create_plotly_theme_layout()
            layout_state["xaxis"]["tickangle"] = -45
            layout_state["yaxis"]["title"] = "Station Count"
            fig_state.update_layout(layout_state)
            charts["chart_json_stations_state"] = json.dumps(fig_state, cls=plotly.utils.PlotlyJSONEncoder)

        if "Status" in df.columns:
            status_counts = df["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            status_colors = [THEME_COLORS["accent_aqua"], THEME_COLORS["accent_cyan"], THEME_COLORS["accent_purple"]]
            fig_status = go.Figure(data=[go.Pie(
                labels=status_counts["Status"],
                values=status_counts["Count"],
                hole=0.55,
                marker=dict(colors=status_colors, line=dict(color=THEME_COLORS["surface_card"], width=2)),
                textinfo="percent+label",
                textfont=dict(size=11, color=THEME_COLORS["text_main"])
            )])
            layout_status = create_plotly_theme_layout()
            layout_status["showlegend"] = False
            fig_status.update_layout(layout_status)
            charts["chart_json_distribution"] = json.dumps(fig_status, cls=plotly.utils.PlotlyJSONEncoder)

        return charts

    # ----------------------------------------------------
    # Charts for Time-Series Air Quality Data (City Day / Station Day)
    # ----------------------------------------------------

    # 1. Time-series Line Chart: Mean AQI, PM2.5, PM10 over time
    if date_col and "AQI" in df.columns:
        ts_df = df.groupby(df[date_col].dt.date).agg({
            "AQI": "mean",
            "PM2.5": "mean" if "PM2.5" in df.columns else lambda x: np.nan,
            "PM10": "mean" if "PM10" in df.columns else lambda x: np.nan,
        }).reset_index()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts_df[date_col],
            y=ts_df["AQI"],
            mode="lines",
            name="AQI",
            line=dict(color="#54E6A5", width=2.2)
        ))
        if "PM2.5" in ts_df.columns:
            fig_ts.add_trace(go.Scatter(
                x=ts_df[date_col],
                y=ts_df["PM2.5"],
                mode="lines",
                name="PM2.5 (\u03bcg/m\u00b3)",
                line=dict(color="#72C9F0", width=1.8, dash="dot")
            ))
        if "PM10" in ts_df.columns:
            fig_ts.add_trace(go.Scatter(
                x=ts_df[date_col],
                y=ts_df["PM10"],
                mode="lines",
                name="PM10 (\u03bcg/m\u00b3)",
                line=dict(color="#C3A6FF", width=1.5, dash="dash")
            ))

        layout_ts = create_plotly_theme_layout()
        layout_ts["xaxis"]["title"] = "Date"
        layout_ts["yaxis"]["title"] = "Concentration (\u03bcg/m\u00b3)"
        fig_ts.update_layout(layout_ts)
        charts["chart_json_timeseries"] = json.dumps(fig_ts, cls=plotly.utils.PlotlyJSONEncoder)

    # 2. Donut Chart: CPCB AQI Category Breakdown
    if "AQI_Bucket" in df.columns:
        bucket_counts = df["AQI_Bucket"].value_counts().reset_index()
        bucket_counts.columns = ["Category", "Count"]

        category_colors = {
            "Good": "#54E6A5",
            "Satisfactory": "#25D9E8",
            "Moderate": "#F5C96A",
            "Poor": "#FF9F1C",
            "Very Poor": "#A78BFA",
            "Severe": "#FF7198",
            "Unknown": "#758A80"
        }
        colors = [category_colors.get(cat, "#25D9E8") for cat in bucket_counts["Category"]]

        fig_pie = go.Figure(data=[go.Pie(
            labels=bucket_counts["Category"],
            values=bucket_counts["Count"],
            hole=0.60,
            marker=dict(colors=colors, line=dict(color="#091510", width=2)),
            textinfo="percent",
            textposition="inside",
            insidetextorientation="radial",
            textfont=dict(size=10, color="#F2F7F5", family="Inter, sans-serif"),
            hoverinfo="label+value+percent"
        )])
        layout_pie = create_plotly_theme_layout()
        layout_pie["showlegend"] = True
        layout_pie["legend"] = dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11, color="#EAF4EF"),
            bgcolor="rgba(0,0,0,0)"
        )
        layout_pie["margin"] = dict(l=15, r=100, t=30, b=30)
        fig_pie.update_layout(layout_pie)
        charts["chart_json_distribution"] = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)

    # 3. Bar Chart: City-Wise or Station-Wise Mean AQI Comparison
    group_col = "City" if "City" in df.columns else ("StationId" if "StationId" in df.columns else None)
    if group_col and "AQI" in df.columns:
        grouped_aqi = df.groupby(group_col)["AQI"].mean().sort_values(ascending=False).head(15).reset_index()
        fig_bar = go.Figure(data=[go.Bar(
            x=grouped_aqi[group_col],
            y=grouped_aqi["AQI"].round(1),
            marker=dict(
                color=grouped_aqi["AQI"],
                colorscale=[[0, "#25D9E8"], [0.5, "#54E6A5"], [1, "#F5C96A"]],
                line=dict(color="#193328", width=1)
            ),
            text=grouped_aqi["AQI"].round(1),
            textposition="auto",
            textfont=dict(color="#020405", size=10, family="JetBrains Mono, monospace")
        )])
        layout_bar = create_plotly_theme_layout()
        layout_bar["xaxis"]["tickangle"] = -45
        layout_bar["yaxis"]["title"] = "Mean AQI"
        fig_bar.update_layout(layout_bar)
        charts["chart_json_city_bar"] = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

    # 4. Distribution Histogram: PM2.5 or Primary Pollutant
    dist_col = "PM2.5" if "PM2.5" in df.columns else ("AQI" if "AQI" in df.columns else None)
    if dist_col:
        fig_hist = go.Figure(data=[go.Histogram(
            x=df[dist_col].dropna(),
            nbinsx=35,
            marker=dict(
                color="#54E6A5",
                line=dict(color="#091510", width=1)
            ),
            opacity=0.85
        )])
        layout_hist = create_plotly_theme_layout()
        layout_hist["xaxis"]["title"] = f"{dist_col} Concentration / Score"
        layout_hist["yaxis"]["title"] = "Observation Frequency"
        fig_hist.update_layout(layout_hist)
        charts["chart_json_histogram"] = json.dumps(fig_hist, cls=plotly.utils.PlotlyJSONEncoder)

    # 5. Scatter Plot: PM2.5 vs PM10
    if "PM2.5" in df.columns and "PM10" in df.columns:
        scatter_sample = df.sample(min(len(df), 800), random_state=42)
        fig_scat = go.Figure(data=[go.Scatter(
            x=scatter_sample["PM2.5"],
            y=scatter_sample["PM10"],
            mode="markers",
            marker=dict(
                size=6,
                color="#72C9F0",
                opacity=0.7,
                line=dict(color="#091510", width=0.5)
            ),
            text=scatter_sample["City"] if "City" in scatter_sample.columns else scatter_sample.index,
            hovertemplate="PM2.5: %{x:.1f} <br>PM10: %{y:.1f}<extra></extra>"
        )])
        layout_scat = create_plotly_theme_layout()
        layout_scat["xaxis"]["title"] = "PM2.5 (\u03bcg/m\u00b3)"
        layout_scat["yaxis"]["title"] = "PM10 (\u03bcg/m\u00b3)"
        fig_scat.update_layout(layout_scat)
        charts["chart_json_scatter"] = json.dumps(fig_scat, cls=plotly.utils.PlotlyJSONEncoder)

    # 6. Correlation Heatmap: Pearson Cross-Correlation Matrix
    pollutant_cols = [c for c in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "AQI"] if c in df.columns]
    if len(pollutant_cols) >= 3:
        corr_matrix = df[pollutant_cols].corr().round(2)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[[0, "#07120F"], [0.5, "#193328"], [1, "#54E6A5"]],
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate="%{text}",
            textfont=dict(size=10, color="#F2F7F5", family="JetBrains Mono, monospace")
        ))
        layout_corr = create_plotly_theme_layout()
        layout_corr["xaxis"]["title"] = ""
        layout_corr["yaxis"]["title"] = ""
        fig_corr.update_layout(layout_corr)
        charts["chart_json_correlation"] = json.dumps(fig_corr, cls=plotly.utils.PlotlyJSONEncoder)

    # 7. Interactive Custom Chart Studio
    if custom_x in df.columns:
        fig_custom = go.Figure()
        sample_sub = df.sample(min(len(df), 600), random_state=42) if len(df) > 600 else df

        if custom_type == "scatter" and custom_y in df.columns:
            fig_custom.add_trace(go.Scatter(
                x=sample_sub[custom_x],
                y=sample_sub[custom_y],
                mode="markers",
                marker=dict(size=6, color="#FF7198", opacity=0.7)
            ))
            y_title = custom_y
        elif custom_type == "bar" and custom_y in df.columns:
            grouped_custom = df.groupby(custom_x)[custom_y].mean().head(15).reset_index()
            fig_custom.add_trace(go.Bar(
                x=grouped_custom[custom_x],
                y=grouped_custom[custom_y].round(1),
                marker=dict(color="#54E6A5")
            ))
            y_title = f"Mean {custom_y}"
        elif custom_type == "line" and custom_y in df.columns:
            if date_col and custom_x == date_col:
                line_grouped = df.groupby(df[date_col].dt.date)[custom_y].mean().reset_index()
                fig_custom.add_trace(go.Scatter(
                    x=line_grouped[date_col],
                    y=line_grouped[custom_y],
                    mode="lines",
                    line=dict(color=THEME_COLORS["accent_cyan"], width=2.5)
                ))
            else:
                fig_custom.add_trace(go.Scatter(
                    x=sample_sub[custom_x],
                    y=sample_sub[custom_y],
                    mode="lines+markers",
                    line=dict(color=THEME_COLORS["accent_cyan"], width=1.5)
                ))
            y_title = custom_y
        else:  # histogram
            fig_custom.add_trace(go.Histogram(
                x=df[custom_x].dropna(),
                marker=dict(color=THEME_COLORS["accent_gold"]),
                opacity=0.85
            ))
            y_title = "Frequency"

        layout_custom = create_plotly_theme_layout()
        layout_custom["xaxis"]["title"] = custom_x
        layout_custom["yaxis"]["title"] = y_title
        fig_custom.update_layout(layout_custom)
        charts["chart_json_custom"] = json.dumps(fig_custom, cls=plotly.utils.PlotlyJSONEncoder)

    return charts


def parse_float(val, default, min_val=None, max_val=None):
    """
    Safely parses a number from any user input (string, int, float)
    with bounds checking and graceful fallback.
    """
    try:
        if val is None or str(val).strip() == "":
            return default
        num = float(val)
        if np.isnan(num) or np.isinf(num):
            return default
        if min_val is not None:
            num = max(min_val, num)
        if max_val is not None:
            num = min(max_val, num)
        return num
    except (ValueError, TypeError):
        return default


def get_model_evaluation_metrics():
    """
    Loads saved evaluation metrics from model/artifacts/model_metrics.json
    and model/artifacts/evaluation_summary.json if present.
    """
    metrics_path = os.path.join(BASE_DIR, "model", "artifacts", "model_metrics.json")
    summary_path = os.path.join(BASE_DIR, "model", "artifacts", "evaluation_summary.json")
    data = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data.update(json.load(f))
        except Exception as e:
            print(f"[Warning] Failed to read model_metrics.json: {e}")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                sum_data = json.load(f)
                data["evaluation_summary"] = sum_data
                if "classification_agreement" in sum_data:
                    data["cpcb_category_accuracy"] = sum_data["classification_agreement"].get("cpcb_category_accuracy")
        except Exception as e:
            print(f"[Warning] Failed to read evaluation_summary.json: {e}")
    return data if data else None


def generate_simulation_charts(sim_result, forecast_result=None):
    """
    Generates three dedicated Plotly charts for the AeroTwin Studio:
    1. Comparison Bar Chart: Ambient Baseline vs 24h Purifier vs Personal Routine vs User-Defined Mitigation.
    2. Time-Activity Donut Chart: 24h micro-environmental schedule.
    3. Digital Twin Holistic Overview: Observed Today vs 24h Ambient Forecast vs What-If Simulated Inhaled Exposure.
    """
    if not sim_result or go is None:
        return {
            "chart_json_sim_bar": None,
            "chart_json_sim_donut": None,
            "chart_json_twin_overview": None
        }

    # 1. Bar Chart (Side-by-side scenario exposure comparison)
    base_val = sim_result["baseline"]["pm25"]
    purifier_val = sim_result["scenario_indoor_only"]["pm25"]
    routine_val = sim_result["scenario_time_weighted"]["pm25"]
    custom_val = sim_result.get("scenario_custom", {}).get("pm25")

    bar_x = ["Ambient Baseline", "24h Continuous Purifier", "Daily Personal Routine"]
    bar_y = [base_val, purifier_val, routine_val]
    bar_colors = [sim_result["baseline"]["color"], "#54E6A5", "#72C9F0"]
    bar_texts = [
        f"{base_val} ug/m3<br>({sim_result['baseline']['tier']})",
        f"{purifier_val} ug/m3<br>({sim_result['scenario_indoor_only']['tier']})",
        f"{routine_val} ug/m3<br>({sim_result['scenario_time_weighted']['tier']})"
    ]

    if custom_val is not None:
        custom_pct = sim_result["parameters"].get("custom_reduction_percent", 50.0)
        bar_x.append(f"Custom Mitigation ({custom_pct}%)")
        bar_y.append(custom_val)
        bar_colors.append("#C3A6FF")
        bar_texts.append(f"{custom_val} ug/m3<br>({sim_result['scenario_custom']['tier']})")

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=bar_x,
        y=bar_y,
        marker_color=bar_colors,
        marker_line=dict(color="#193328", width=1.5),
        text=bar_texts,
        textposition="auto",
        textfont=dict(color="#F2F7F5", size=11),
        hoverinfo="x+y+text"
    ))
    layout_bar = create_plotly_theme_layout()
    layout_bar["yaxis"]["title"] = "Inhaled PM2.5 (ug/m3)"
    layout_bar["margin"] = dict(l=45, r=20, t=30, b=40)
    fig_bar.update_layout(layout_bar)
    bar_json = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

    # 2. Donut Chart (Time Budget)
    params = sim_result["parameters"]
    indoor_h = params["indoor_hours"]
    transit_h = params["transit_hours"]
    outdoor_h = params["outdoor_hours"]

    fig_donut = go.Figure(data=[go.Pie(
        labels=["Indoor (Purified)", "Transit (Masked)", "Outdoor (Unmitigated)"],
        values=[indoor_h, transit_h, outdoor_h],
        hole=0.62,
        marker=dict(colors=["#54E6A5", "#C3A6FF", "#FF7198"], line=dict(color="#091510", width=2)),
        textinfo="label+value",
        texttemplate="%{label}<br>%{value}h (%{percent})",
        hoverinfo="label+value+percent"
    )])
    layout_donut = create_plotly_theme_layout()
    layout_donut["margin"] = dict(l=20, r=20, t=20, b=20)
    layout_donut["legend"] = dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(color="#A5B7AF", size=10))
    layout_donut["annotations"] = [dict(text="24h Budget", x=0.5, y=0.5, font_size=13, font_color="#F2F7F5", showarrow=False)]
    fig_donut.update_layout(layout_donut)
    donut_json = json.dumps(fig_donut, cls=plotly.utils.PlotlyJSONEncoder)

    # 3. Digital Twin Holistic Overview (Observed vs Forecast vs Simulated)
    obs_pm25 = forecast_result.get("current_pm25") if (forecast_result and forecast_result.get("current_pm25") is not None) else base_val
    pred_pm25 = forecast_result.get("forecast_pm25") if (forecast_result and forecast_result.get("forecast_pm25") is not None) else None
    sim_pm25 = sim_result["scenario_time_weighted"]["pm25"]

    overview_x = ["1. Observed Ambient [OBSERVED]"]
    overview_y = [obs_pm25]
    overview_colors = ["#F5C96A"]
    overview_texts = [f"{obs_pm25} ug/m3<br>(Ground Telemetry)"]

    if pred_pm25 is not None:
        overview_x.append("2. Next-Day Ambient [FORECAST]")
        overview_y.append(pred_pm25)
        overview_colors.append("#54E6A5")
        overview_texts.append(f"{pred_pm25} ug/m3<br>(ML Forecaster)")

    overview_x.append("3. Personal Daily Routine [SIMULATED]")
    overview_y.append(sim_pm25)
    overview_colors.append("#72C9F0")
    overview_texts.append(f"{sim_pm25} ug/m3<br>(What-If Routine)")

    fig_overview = go.Figure()
    fig_overview.add_trace(go.Bar(
        x=overview_x,
        y=overview_y,
        marker_color=overview_colors,
        marker_line=dict(color="#193328", width=1.5),
        text=overview_texts,
        textposition="auto",
        textfont=dict(color="#F2F7F5", size=11),
        hoverinfo="x+y+text"
    ))
    layout_overview = create_plotly_theme_layout()
    layout_overview["yaxis"]["title"] = "PM2.5 (ug/m3)"
    layout_overview["margin"] = dict(l=45, r=20, t=30, b=40)
    fig_overview.update_layout(layout_overview)
    twin_overview_json = json.dumps(fig_overview, cls=plotly.utils.PlotlyJSONEncoder)

    return {
        "chart_json_sim_bar": bar_json,
        "chart_json_sim_donut": donut_json,
        "chart_json_twin_overview": twin_overview_json
    }


def create_flask_app():
    """
    Application factory for the AeroTwin Flask web application.
    """
    if Flask is None:
        return None

    app = Flask(__name__, template_folder=TEMPLATES_DIR)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "aerotwin-vibrant-crystal-secret-key-2026")

    @app.route("/")
    def index():
        available_datasets = get_available_datasets()
        if not available_datasets:
            return (
                "<div style='font-family:sans-serif; background:#030609; color:#F3F8FC; padding:40px; text-align:center;'>"
                "<h2 style='color:#00F5D4;'>AeroTwin — Data Pipeline Notice</h2>"
                "<p style='color:#9CB0C0;'>No cleaned datasets were found in <code>data/processed/</code>.</p>"
                "<p>Please execute the data cleaning pipeline first:</p>"
                "<pre style='background:#07131C; padding:12px; display:inline-block; border-radius:6px; color:#00F5D4; border:1px solid #173744;'>python cleaning/clean_data.py</pre>"
                "</div>"
            )

        # 1. Dataset selection
        default_ds = "cleaned_city_day.csv" if "cleaned_city_day.csv" in available_datasets else available_datasets[0]
        selected_ds_name = request.args.get("dataset", default_ds)
        if selected_ds_name not in available_datasets:
            selected_ds_name = default_ds

        df = load_dataset(selected_ds_name)
        if df is None or df.empty:
            return (
                "<div style='font-family:sans-serif; background:#030609; color:#F3F8FC; padding:40px; text-align:center;'>"
                "<h2 style='color:#FF6FAE;'>AeroTwin — Dataset Notice</h2>"
                f"<p style='color:#9CB0C0;'>Dataset <code>{selected_ds_name}</code> could not be loaded or contains no records.</p>"
                "<p><a href='/' style='color:#00F5D4; text-decoration:none;'>← Return to Primary Dashboard</a></p>"
                "</div>"
            )

        total_rows_unfiltered = len(df)
        total_cols = len(df.columns)

        # 2. Extract Cities for filtering if present
        cities = sorted(df["City"].dropna().unique().tolist()) if "City" in df.columns else []
        selected_city = request.args.get("city", "All")

        # 3. Filter by City
        filtered_df = df.copy()
        if selected_city != "All" and "City" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["City"] == selected_city]

        # 4. Extract Date range and filter if present
        date_col = next((c for c in ["Date", "Datetime"] if c in filtered_df.columns), None)
        has_dates = date_col is not None and pd.api.types.is_datetime64_any_dtype(filtered_df[date_col])

        min_date, max_date = "", ""
        start_date = request.args.get("start_date", "")
        end_date = request.args.get("end_date", "")

        if has_dates:
            valid_dates = df[date_col].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().strftime("%Y-%m-%d")
                max_date = valid_dates.max().strftime("%Y-%m-%d")

            if start_date:
                filtered_df = filtered_df[filtered_df[date_col] >= pd.to_datetime(start_date)]
            if end_date:
                filtered_df = filtered_df[filtered_df[date_col] <= pd.to_datetime(end_date)]

        # 5. Calculate Metrics
        metrics = calculate_metrics(filtered_df)

        # 6. Column Profiles & Numeric Options for Custom Chart Studio
        column_profiles = get_column_profiles(df)
        numeric_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["Year", "Month", "Day"]]
        all_columns = df.columns.tolist()

        # Custom chart studio parameters
        custom_chart_type = request.args.get("chart_type", "scatter")
        default_x = "PM2.5" if "PM2.5" in all_columns else all_columns[0]
        default_y = "PM10" if "PM10" in all_columns else (all_columns[1] if len(all_columns) > 1 else all_columns[0])
        custom_x = request.args.get("custom_x", default_x)
        custom_y = request.args.get("custom_y", default_y)

        # 7. Generate Plotly Charts for Visualizations Section
        charts = generate_dashboard_charts(
            filtered_df,
            custom_type=custom_chart_type,
            custom_x=custom_x,
            custom_y=custom_y
        )

        # 8. Environmental Overview Extraction (Observed Ground Telemetry)
        latest_pollutants = {}
        latest_date_str = "N/A"
        if not filtered_df.empty:
            if date_col and pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
                valid_dates = filtered_df[date_col].dropna()
                if not valid_dates.empty:
                    latest_date_str = valid_dates.iloc[-1].strftime("%Y-%m-%d")

            pollutant_units = {
                "PM2.5": "ug/m3",
                "PM10": "ug/m3",
                "NO2": "ug/m3",
                "SO2": "ug/m3",
                "CO": "mg/m3",
                "O3": "ug/m3",
                "AQI": "Index"
            }
            for pol in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]:
                if pol in filtered_df.columns:
                    valid_vals = filtered_df[pol].dropna()
                    if not valid_vals.empty:
                        val = round(float(valid_vals.iloc[-1]), 1)
                        tier_info = get_cpcb_tier(val) if (pol == "PM2.5" and get_cpcb_tier) else None
                        latest_pollutants[pol] = {
                            "val": val,
                            "unit": pollutant_units.get(pol, "ug/m3"),
                            "tier": tier_info["tier"] if tier_info else "Monitored",
                            "color": tier_info["color"] if tier_info else THEME_COLORS["accent_cyan"]
                        }

        environmental_overview = {
            "dataset_name": selected_ds_name,
            "observation_date": latest_date_str,
            "location": selected_city if selected_city != "All" else "All Monitored Regions",
            "source": "CPCB Ground Monitoring Stations" if ("city" in selected_ds_name or "station" in selected_ds_name) else "Processed Environmental Telemetry",
            "pollutants": latest_pollutants,
            "is_observed": True,
            "has_pollutants": bool(latest_pollutants),
            "primary_pm25": latest_pollutants.get("PM2.5", {}).get("val") if "PM2.5" in latest_pollutants else None,
            "primary_aqi": latest_pollutants.get("AQI", {}).get("val") if "AQI" in latest_pollutants else None
        }

        # 9. Environmental Forecasting Inference (ML Next-Day Forecaster)
        model_metrics = get_model_evaluation_metrics()
        forecast_result = None
        if predict_next_day_pm25 and not filtered_df.empty:
            if "PM2.5" in filtered_df.columns:
                sample_features = {}
                for col in filtered_df.columns:
                    if pd.api.types.is_numeric_dtype(filtered_df[col]):
                        valid_vals = filtered_df[col].dropna()
                        if not valid_vals.empty:
                            sample_features[col] = float(valid_vals.iloc[-1])
                forecast_result = predict_next_day_pm25(sample_features)
            else:
                forecast_result = {
                    "status": "NO_POLLUTANT_DATA",
                    "message": "Atmospheric PM2.5 forecasting requires pollutant time-series records.",
                    "is_forecast": True
                }

        # 10. What-If Personal Exposure Simulation (Mass Balance Simulation Engine)
        default_ambient = environmental_overview["primary_pm25"]
        if default_ambient is None:
            if "PM2.5" in filtered_df.columns:
                valid_pm = filtered_df["PM2.5"].dropna()
                default_ambient = round(float(valid_pm.mean()), 1) if not valid_pm.empty else 100.0
            else:
                default_ambient = 100.0

        sim_ambient = parse_float(request.args.get("sim_ambient"), default_ambient, min_val=0.0)
        sim_indoor_h = parse_float(request.args.get("sim_indoor_h"), 18.0, min_val=0.0, max_val=24.0)
        sim_transit_h = parse_float(request.args.get("sim_transit_h"), 2.0, min_val=0.0, max_val=24.0)
        sim_purifier = parse_float(request.args.get("sim_purifier"), 75.0, min_val=0.0, max_val=100.0)
        sim_mask = parse_float(request.args.get("sim_mask"), 80.0, min_val=0.0, max_val=100.0)
        sim_custom = parse_float(request.args.get("sim_custom"), 50.0, min_val=0.0, max_val=100.0)

        sim_result = {}
        sim_charts = {}
        if simulate_personal_exposure:
            sim_result = simulate_personal_exposure(
                ambient_pm25=sim_ambient,
                indoor_hours=sim_indoor_h,
                transit_hours=sim_transit_h,
                purifier_reduction_pct=sim_purifier,
                mask_reduction_pct=sim_mask,
                custom_reduction_pct=sim_custom
            )
            sim_charts = generate_simulation_charts(sim_result, forecast_result)

        # 11. Format Table Sample
        limit = int(request.args.get("limit", 15))
        sample_df = filtered_df.head(limit).copy()

        for c in sample_df.columns:
            if pd.api.types.is_datetime64_any_dtype(sample_df[c]):
                sample_df[c] = sample_df[c].dt.strftime("%Y-%m-%d")
            elif pd.api.types.is_float_dtype(sample_df[c]):
                sample_df[c] = sample_df[c].round(2)

        table_columns = sample_df.columns.tolist()
        table_data = sample_df.to_dict(orient="records")

        return render_template(
            "index.html",
            available_datasets=available_datasets,
            active_dataset_name=selected_ds_name,
            total_rows=total_rows_unfiltered,
            filtered_rows=len(filtered_df),
            total_cols=total_cols,
            missing_values=metrics["missing_values"],
            duplicate_rows=metrics["duplicate_rows"],
            date_range_str=metrics["date_range_str"],
            cities=cities,
            selected_city=selected_city,
            has_dates=has_dates,
            min_date=min_date,
            max_date=max_date,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            table_columns=table_columns,
            table_data=table_data,
            column_profiles=column_profiles,
            numeric_columns=numeric_columns,
            all_columns=all_columns,
            custom_chart_type=custom_chart_type,
            custom_x=custom_x,
            custom_y=custom_y,
            charts=charts,
            chart_json_timeseries=charts.get("chart_json_timeseries"),
            chart_json_distribution=charts.get("chart_json_distribution"),
            chart_json_city_bar=charts.get("chart_json_city_bar"),
            chart_json_correlation=charts.get("chart_json_correlation"),
            chart_json_histogram=charts.get("chart_json_histogram"),
            chart_json_scatter=charts.get("chart_json_scatter"),
            chart_json_custom=charts.get("chart_json_custom"),
            chart_json_stations_state=charts.get("chart_json_stations_state"),
            environmental_overview=environmental_overview,
            sim_result=sim_result,
            chart_json_sim_bar=sim_charts.get("chart_json_sim_bar"),
            chart_json_sim_donut=sim_charts.get("chart_json_sim_donut"),
            chart_json_twin_overview=sim_charts.get("chart_json_twin_overview"),
            model_metrics=model_metrics,
            forecast_result=forecast_result,
            sim_ambient=sim_ambient,
            sim_purifier=sim_purifier,
            sim_mask=sim_mask,
            sim_indoor_h=sim_indoor_h,
            sim_transit_h=sim_transit_h,
            sim_custom=sim_custom
        )

    @app.route("/api/simulate", methods=["GET", "POST"])
    def api_simulate():
        """
        API endpoint for What-If Micro-Environmental Personal Exposure Simulation.
        Accepts GET query parameters or POST JSON/form data.
        """
        if not simulate_personal_exposure:
            return {"error": "Simulation engine not loaded"}, 500

        data = {}
        if request.method == "POST":
            if request.is_json:
                data = request.get_json(silent=True) or {}
            else:
                data = request.form.to_dict()
        else:
            data = request.args.to_dict()

        ambient = parse_float(data.get("ambient_pm25", data.get("ambient")), 100.0, min_val=0.0)
        indoor_h = parse_float(data.get("indoor_hours", data.get("indoor_h")), 18.0, min_val=0.0, max_val=24.0)
        transit_h = parse_float(data.get("transit_hours", data.get("transit_h")), 2.0, min_val=0.0, max_val=24.0)
        purifier = parse_float(data.get("purifier_reduction", data.get("purifier_reduction_pct")), 75.0, min_val=0.0, max_val=100.0)
        mask = parse_float(data.get("mask_reduction", data.get("mask_reduction_pct")), 80.0, min_val=0.0, max_val=100.0)
        custom = parse_float(data.get("custom_reduction", data.get("custom_reduction_pct")), 50.0, min_val=0.0, max_val=100.0)

        res = simulate_personal_exposure(ambient, indoor_h, transit_h, purifier, mask, custom)
        return res

    @app.route("/api/forecast", methods=["GET", "POST"])
    def api_forecast():
        """
        API endpoint for 24-Hour Next-Day Atmospheric PM2.5 Forecast.
        Accepts GET query parameters or POST JSON/form data.
        """
        if not predict_next_day_pm25:
            return {"error": "Prediction engine not loaded"}, 500

        data = {}
        if request.method == "POST":
            if request.is_json:
                data = request.get_json(silent=True) or {}
            else:
                data = request.form.to_dict()
        else:
            data = request.args.to_dict()

        features = {}
        for k, v in data.items():
            if k in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI", "Month", "Year", "Day", "Is_Weekend"]:
                features[k] = parse_float(v, 0.0)
            else:
                features[k] = v

        if "PM2.5" not in features and "pm25" in data:
            features["PM2.5"] = parse_float(data["pm25"], 100.0)
        if "AQI" not in features and "aqi" in data:
            features["AQI"] = parse_float(data["aqi"], 200.0)

        res = predict_next_day_pm25(features if features else {"PM2.5": 100.0})
        return res

    @app.route("/api/model-info", methods=["GET"])
    def api_model_info():
        """
        Returns machine learning model evaluation metrics, test split metadata,
        and scientific limitations.
        """
        metrics = get_model_evaluation_metrics()
        if not metrics:
            return {"status": "UNAVAILABLE", "message": "Model evaluation artifacts not found"}, 404
        return metrics

    @app.route("/download/<dataset_name>")
    def download_csv(dataset_name):
        available_datasets = get_available_datasets()
        if dataset_name not in available_datasets:
            abort(404)
        file_path = os.path.join(PROCESSED_DIR, dataset_name)
        return send_file(file_path, as_attachment=True, download_name=dataset_name)

    @app.route("/api/metrics")
    def api_metrics():
        ds = request.args.get("dataset", "cleaned_city_day.csv")
        df = load_dataset(ds)
        if df is None:
            return {"error": "Dataset not found"}, 404
        return calculate_metrics(df)

    # ========================================================
    # HEALTHCARE DIGITAL TWIN ROUTES (TYPE 2 DIABETES)
    # ========================================================

    @app.route("/twin")
    def healthcare_twin():
        """
        Primary Doctor-Facing Healthcare Digital Twin Dashboard for Type 2 Diabetes.
        Aligned with Happiest Health Summit 2026.
        """
        patients = load_synthetic_patients() if load_synthetic_patients else []
        default_pid = patients[0]["patient_id"] if patients else "SYN-T2D-001"
        selected_pid = request.args.get("patient_id", default_pid)

        # Replay step (default to step 120 = 10:00 AM on Day 1, postprandial window)
        step = parse_float(request.args.get("step"), 120.0, min_val=24.0)

        # Retrieve virtual patient synchronized state
        state = {}
        charts = {}
        eval_summary = {}
        env_context = {}

        if get_digital_twin_state_at_step:
            state = get_digital_twin_state_at_step(selected_pid, int(step))

            # Generate Plotly charts for the initial page load
            if create_cgm_history_chart:
                charts["chart_cgm_history"] = create_cgm_history_chart(
                    state.get("recent_history"),
                    state["live_vitals"]["glucose_mg_dl"],
                    state["stream_meta"]["timestamp"]
                )
            if create_forecast_trajectory_chart:
                charts["chart_forecast_trajectory"] = create_forecast_trajectory_chart(
                    state.get("recent_history"),
                    state.get("forecast"),
                    state.get("ground_truth_validation")
                )
            if create_whatif_comparison_chart:
                charts["chart_whatif_comparison"] = create_whatif_comparison_chart(
                    state.get("whatif_preview")
                )
            if create_tir_donut_chart:
                charts["chart_tir_donut"] = create_tir_donut_chart(
                    state.get("glycemic_summary")
                )

        if get_model_evaluation_metrics_summary:
            eval_summary = get_model_evaluation_metrics_summary()
            if create_confusion_matrix_chart:
                charts["chart_confusion_matrix"] = create_confusion_matrix_chart(eval_summary)

        if get_environmental_metabolic_context:
            env_context = get_environmental_metabolic_context()

        return render_template(
            "twin.html",
            patients=patients,
            selected_patient_id=selected_pid,
            state=state,
            charts=charts,
            eval_summary=eval_summary,
            env_context=env_context
        )

    @app.route("/api/healthcare/patients", methods=["GET"])
    def api_healthcare_patients():
        """Returns synthetic patient directory."""
        if not load_synthetic_patients:
            return {"error": "Healthcare service unavailable"}, 500
        return {"patients": load_synthetic_patients()}

    @app.route("/api/healthcare/patient/<patient_id>", methods=["GET"])
    def api_healthcare_patient(patient_id):
        """Returns detailed synthetic EHR for a specific patient."""
        if not get_patient_profile:
            return {"error": "Healthcare service unavailable"}, 500
        profile = get_patient_profile(patient_id)
        if not profile:
            return {"error": "Patient not found"}, 404
        return profile

    @app.route("/api/healthcare/state", methods=["GET", "POST"])
    def api_healthcare_state():
        """
        Dynamic replay synchronization endpoint.
        Returns synchronized patient state and updated Plotly chart JSONs at a specific replay step.
        """
        if not get_digital_twin_state_at_step:
            return {"error": "Twin service unavailable"}, 500

        data = request.get_json(silent=True) or request.args.to_dict()
        patient_id = data.get("patient_id", "SYN-T2D-001")
        step = int(parse_float(data.get("step"), 120.0, min_val=24.0))

        state = get_digital_twin_state_at_step(patient_id, step)
        if "error" in state:
            return state, 404

        charts = {}
        if create_cgm_history_chart:
            charts["chart_cgm_history"] = create_cgm_history_chart(
                state.get("recent_history"),
                state["live_vitals"]["glucose_mg_dl"],
                state["stream_meta"]["timestamp"]
            )
        if create_forecast_trajectory_chart:
            charts["chart_forecast_trajectory"] = create_forecast_trajectory_chart(
                state.get("recent_history"),
                state.get("forecast"),
                state.get("ground_truth_validation")
            )
        if create_tir_donut_chart:
            charts["chart_tir_donut"] = create_tir_donut_chart(
                state.get("glycemic_summary")
            )

        return {
            "status": "SYNCHRONIZED",
            "state": state,
            "charts": charts
        }

    @app.route("/api/healthcare/predict", methods=["GET", "POST"])
    def api_healthcare_predict():
        """
        API endpoint for 120-minute hyperglycemic excursion prediction and trajectory forecasting.
        """
        if not get_digital_twin_state_at_step:
            return {"error": "Twin service unavailable"}, 500

        data = request.get_json(silent=True) or request.args.to_dict()
        patient_id = data.get("patient_id", "SYN-T2D-001")
        step = int(parse_float(data.get("step"), 120.0, min_val=24.0))

        state = get_digital_twin_state_at_step(patient_id, step)
        if "error" in state:
            return state, 404

        return {
            "patient_id": patient_id,
            "timestamp": state["stream_meta"]["timestamp"],
            "features_used": state["features"],
            "forecast": state["forecast"],
            "ground_truth_validation": state.get("ground_truth_validation")
        }

    @app.route("/api/healthcare/whatif", methods=["GET", "POST"])
    def api_healthcare_whatif():
        """
        API endpoint for What-If physiological intervention simulation.
        """
        if not simulate_glucose_intervention or not get_digital_twin_state_at_step:
            return {"error": "Simulation engine unavailable"}, 500

        data = request.get_json(silent=True) or request.args.to_dict()
        patient_id = data.get("patient_id", "SYN-T2D-001")
        step = int(parse_float(data.get("step"), 120.0, min_val=24.0))

        walk_mins = parse_float(data.get("walk_minutes", data.get("post_meal_walk_minutes")), 20.0, min_val=0.0, max_val=60.0)
        intensity = str(data.get("walk_intensity", "Moderate"))
        sleep_ext = parse_float(data.get("sleep_extension_hours"), 0.0, min_val=-2.0, max_val=3.0)
        carb_offset = parse_float(data.get("carbohydrate_offset_pct"), 0.0, min_val=0.0, max_val=50.0)

        state = get_digital_twin_state_at_step(patient_id, step)
        curr_g = state["live_vitals"]["glucose_mg_dl"]
        base_curve = state["forecast"]["projected_trajectory"]

        sim_res = simulate_glucose_intervention(
            baseline_glucose_trajectory=base_curve,
            current_glucose=curr_g,
            post_meal_walk_minutes=walk_mins,
            walk_intensity=intensity,
            sleep_extension_hours=sleep_ext,
            carbohydrate_offset_pct=carb_offset
        )

        chart_json = None
        if create_whatif_comparison_chart:
            chart_json = create_whatif_comparison_chart(sim_res)

        return {
            "simulation": sim_res,
            "chart_whatif_comparison": chart_json
        }

    @app.route("/api/healthcare/model-evaluation", methods=["GET"])
    def api_healthcare_model_eval():
        """Returns verified machine learning evaluation metrics and test confusion matrix."""
        if not get_model_evaluation_metrics_summary:
            return {"error": "Evaluation service unavailable"}, 500
        return get_model_evaluation_metrics_summary()

    @app.route("/api/healthcare/environmental-context", methods=["GET"])
    def api_healthcare_environmental_context():
        """Returns ambient air quality data and metabolic correlation context."""
        if not get_environmental_metabolic_context:
            return {"error": "Environmental context service unavailable"}, 500
        return get_environmental_metabolic_context()

    return app


def main():
    """
    Entry point to launch the Flask development server.
    Automatically detects open port (trying 5000, 5001, 8080) for Windows network compatibility.
    """
    if Flask is None:
        print("[AeroTwin Error]: Flask is not installed.")
        print("Please install Flask using: pip install flask")
        return

    app = create_flask_app()
    if app:
        port = int(os.environ.get("PORT", 0))
        if port == 0:
            import socket
            for p in [5000, 5001, 8080]:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.bind(("127.0.0.1", p))
                    s.close()
                    port = p
                    break
                except OSError:
                    continue
            if port == 0:
                port = 5000

        print("=" * 60)
        print("    AEROTWIN - VIBRANT BLACK CRYSTAL ANALYTICS DASHBOARD    ")
        print("=" * 60)
        print(f"Server running locally at: http://127.0.0.1:{port}/")
        debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
        app.run(host="127.0.0.1", port=port, debug=debug_mode)


if __name__ == "__main__":
    main()
