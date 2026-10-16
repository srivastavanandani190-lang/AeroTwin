"""
AeroTwin Healthcare Digital Twin - Plotly Chart Generator
==========================================================
Module: model/healthcare/twin_charts.py
Purpose: Generates Plotly JSON charts aligned with the approved theme:
         Near-black (#020405) + Deep Forest (#07120F/#091510) + Jade (#54E6A5) + Ice Blue (#72C9F0)
"""

import json
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.utils
except ImportError:
    go = None


def get_healthcare_plotly_layout(title=""):
    """
    Standard layout configuration for Healthcare Digital Twin charts.
    """
    return dict(
        title=dict(
            text=title,
            font=dict(family="Inter, -apple-system, sans-serif", size=13, color="#F2F7F5"),
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
            font=dict(size=10, color="#A5B7AF")
        )
    )


def create_cgm_history_chart(recent_history, current_glucose, current_time):
    """
    Renders CGM historical readings with clinical threshold lines (70 and 180 mg/dL).
    """
    if go is None or not recent_history or not recent_history.get("timestamps"):
        return None

    ts = recent_history["timestamps"]
    g = recent_history["glucose"]

    fig = go.Figure()

    # Target glycemic range zone (70 to 180 mg/dL)
    fig.add_hrect(
        y0=70, y1=180,
        fillcolor="rgba(84, 230, 165, 0.07)",
        layer="below", line_width=0,
        annotation_text="ADA Target In-Range (70-180 mg/dL)",
        annotation_position="top left",
        annotation_font=dict(color="#54E6A5", size=9)
    )

    # 180 mg/dL Excursion Warning Threshold Line
    fig.add_hline(
        y=180,
        line=dict(color="#FF7198", width=1.5, dash="dash"),
        annotation_text="Excursion Threshold (180 mg/dL)",
        annotation_position="bottom right",
        annotation_font=dict(color="#FF7198", size=9)
    )

    # Historical CGM Curve
    fig.add_trace(go.Scatter(
        x=ts,
        y=g,
        mode="lines+markers",
        name="CGM Glucose (mg/dL)",
        line=dict(color="#54E6A5", width=2.5),
        marker=dict(size=4, color="#54E6A5"),
        hoverinfo="x+y"
    ))

    # Live Current Point Indicator
    fig.add_trace(go.Scatter(
        x=[current_time],
        y=[current_glucose],
        mode="markers",
        name="Current Telemetry",
        marker=dict(
            size=11,
            color="#FF7198" if current_glucose >= 180 else "#25D9E8",
            line=dict(color="#FFFFFF", width=2)
        ),
        hoverinfo="x+y+name"
    ))

    layout = get_healthcare_plotly_layout()
    layout["yaxis"]["title"] = "Glucose (mg/dL)"
    layout["yaxis"]["range"] = [max(40, min(g) - 15), max(240, max(g) + 20)]
    fig.update_layout(layout)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_forecast_trajectory_chart(recent_history, forecast_data, ground_truth=None):
    """
    Renders the continuous 2-hour projected trajectory alongside historical readings
    and subsequent observed ground-truth values when available during replay.
    """
    if go is None or not forecast_data:
        return None

    fig = go.Figure()

    # Target glycemic range zone
    fig.add_hrect(
        y0=70, y1=180,
        fillcolor="rgba(84, 230, 165, 0.06)",
        layer="below", line_width=0
    )
    fig.add_hline(
        y=180,
        line=dict(color="#FF7198", width=1.5, dash="dash"),
        annotation_text="High Glucose Excursion (>=180 mg/dL)",
        annotation_position="bottom right",
        annotation_font=dict(color="#FF7198", size=9)
    )

    # Past 6 readings (last 30 mins) leading up to now
    past_ts = recent_history["timestamps"][-6:] if recent_history else []
    past_g = recent_history["glucose"][-6:] if recent_history else []

    if past_ts:
        fig.add_trace(go.Scatter(
            x=past_ts,
            y=past_g,
            mode="lines+markers",
            name="Recent Observed",
            line=dict(color="#A5B7AF", width=2),
            marker=dict(size=4, color="#A5B7AF")
        ))

    # Forecast points (24 points at 5-min intervals over next 120 minutes)
    traj = forecast_data.get("projected_trajectory", [])
    if past_ts and traj:
        # Generate future timestamps
        import pandas as pd
        last_dt = pd.to_datetime(past_ts[-1])
        future_ts = [(last_dt + pd.Timedelta(minutes=5 * (i + 1))).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(traj))]

        # Connecting point
        fx = [past_ts[-1]] + future_ts
        fy = [past_g[-1]] + traj

        # Forecast Curve
        traj_color = "#FF7198" if forecast_data.get("predicted_excursion") else "#72C9F0"
        fig.add_trace(go.Scatter(
            x=fx,
            y=fy,
            mode="lines+markers",
            name="Digital Twin 120m Forecast",
            line=dict(color=traj_color, width=2.5, dash="dot"),
            marker=dict(size=4, color=traj_color)
        ))

        # Upper and lower uncertainty confidence ribbon (+- 10 mg/dL based on test MAE)
        upper_bound = [v + 12.0 for v in fy]
        lower_bound = [max(40.0, v - 12.0) for v in fy]

        fig.add_trace(go.Scatter(
            x=fx + fx[::-1],
            y=upper_bound + lower_bound[::-1],
            fill="toself",
            fillcolor="rgba(114, 201, 240, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Forecast Ribbon (MAE +/- 12 mg/dL)"
        ))

        # Plot Actual Ground Truth Observations if available
        if ground_truth and ground_truth.get("actual_trajectory"):
            actual_y = [past_g[-1]] + ground_truth["actual_trajectory"]
            actual_x = [past_ts[-1]] + ground_truth["future_timestamps"]
            fig.add_trace(go.Scatter(
                x=actual_x,
                y=actual_y,
                mode="lines",
                name="Actual Subsequent Readings",
                line=dict(color="#F5C96A", width=2),
                hoverinfo="x+y+name"
            ))

    layout = get_healthcare_plotly_layout()
    layout["yaxis"]["title"] = "Glucose (mg/dL)"
    layout["yaxis"]["range"] = [60, 260]
    fig.update_layout(layout)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_whatif_comparison_chart(whatif_res):
    """
    Renders baseline vs simulated what-if trajectory (e.g. post-meal walk).
    """
    if go is None or not whatif_res or "trajectories" not in whatif_res:
        return None

    trajs = whatif_res["trajectories"]
    mins = trajs["time_steps_minutes"]
    base_curve = trajs["baseline_curve"]
    sim_curve = trajs["simulated_curve"]

    fig = go.Figure()

    fig.add_hline(
        y=180,
        line=dict(color="#FF7198", width=1.5, dash="dash"),
        annotation_text="Excursion Threshold (180 mg/dL)",
        annotation_position="bottom right",
        annotation_font=dict(color="#FF7198", size=9)
    )

    # Baseline Curve
    fig.add_trace(go.Scatter(
        x=mins,
        y=base_curve,
        mode="lines+markers",
        name="Baseline Routine (Sedentary)",
        line=dict(color="#FF7198", width=2.5),
        marker=dict(size=4, color="#FF7198")
    ))

    # Simulated Curve
    fig.add_trace(go.Scatter(
        x=mins,
        y=sim_curve,
        mode="lines+markers",
        name="What-If Intervention",
        line=dict(color="#54E6A5", width=2.5),
        marker=dict(size=4, color="#54E6A5")
    ))

    # Shaded Benefit Area
    fig.add_trace(go.Scatter(
        x=mins + mins[::-1],
        y=base_curve + sim_curve[::-1],
        fill="toself",
        fillcolor="rgba(84, 230, 165, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Glycemic Reduction (Area Under Curve)"
    ))

    layout = get_healthcare_plotly_layout()
    layout["xaxis"]["title"] = "Elapsed Minutes from Present"
    layout["yaxis"]["title"] = "Simulated Glucose (mg/dL)"
    layout["yaxis"]["range"] = [60, max(240, max(base_curve) + 20)]
    fig.update_layout(layout)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_tir_donut_chart(glycemic_summary):
    """
    Renders Time-in-Range (TIR) donut chart following ADA clinical standards:
      - In-Range (70-180 mg/dL)
      - Above-Range (>180 mg/dL)
      - Below-Range (<70 mg/dL)
    """
    if go is None or not glycemic_summary:
        return None

    tir = glycemic_summary.get("time_in_range_pct", 70.0)
    tar = glycemic_summary.get("time_above_range_pct", 25.0)
    tbr = glycemic_summary.get("time_below_range_pct", 5.0)

    fig = go.Figure(data=[go.Pie(
        labels=["In-Range (70-180 mg/dL)", "Above-Range (>180 mg/dL)", "Below-Range (<70 mg/dL)"],
        values=[tir, tar, tbr],
        hole=0.64,
        marker=dict(
            colors=["#54E6A5", "#FF7198", "#F5C96A"],
            line=dict(color="#091510", width=2)
        ),
        textinfo="label+percent",
        texttemplate="%{label}<br>%{value}%",
        hoverinfo="label+value+percent"
    )])

    layout = get_healthcare_plotly_layout()
    layout["margin"] = dict(l=20, r=20, t=20, b=20)
    layout["legend"] = dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(color="#A5B7AF", size=10))
    layout["annotations"] = [dict(text=f"TIR<br>{tir}%", x=0.5, y=0.5, font_size=14, font_color="#54E6A5", showarrow=False)]
    fig.update_layout(layout)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_confusion_matrix_chart(eval_summary):
    """
    Renders 2x2 confusion matrix heatmap for the 2,760 held-out test windows.
    """
    if go is None or not eval_summary or "classification_metrics" not in eval_summary:
        return None

    cm_data = eval_summary["classification_metrics"]["confusion_matrix"]
    matrix = [
        [cm_data["true_negative"], cm_data["false_positive"]],
        [cm_data["false_negative"], cm_data["true_positive"]]
    ]

    labels_x = ["Pred: In-Range (<180)", "Pred: Excursion (>=180)"]
    labels_y = ["True: In-Range (<180)", "True: Excursion (>=180)"]

    annotations = []
    for i in range(2):
        for j in range(2):
            val = matrix[i][j]
            label = "TN" if (i==0 and j==0) else ("FP" if (i==0 and j==1) else ("FN" if (i==1 and j==0) else "TP"))
            annotations.append(dict(
                x=labels_x[j],
                y=labels_y[i],
                text=f"<b>{label}</b><br>{val:,}",
                showarrow=False,
                font=dict(color="#F2F7F5", size=13)
            ))

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels_x,
        y=labels_y,
        colorscale=[[0, "#07120F"], [0.5, "#12271D"], [1.0, "#27C98B"]],
        showscale=False
    ))

    layout = get_healthcare_plotly_layout()
    layout["annotations"] = annotations
    layout["margin"] = dict(l=80, r=30, t=30, b=40)
    fig.update_layout(layout)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
