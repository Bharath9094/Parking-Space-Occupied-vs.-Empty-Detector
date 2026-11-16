from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CHART_TEMPLATE = "plotly_dark"
CHART_COLORS = {"EMPTY": "#22c55e", "OCCUPIED": "#ef4444"}


def history_frame(history: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(history)
    if not frame.empty:
        frame["detected_at"] = pd.to_datetime(frame["detected_at"], errors="coerce")
    return frame


def occupancy_pie(occupied: int, empty: int) -> go.Figure:
    figure = px.pie(
        names=["Occupied", "Empty"],
        values=[occupied, empty],
        color=["Occupied", "Empty"],
        color_discrete_map={
            "Occupied": CHART_COLORS["OCCUPIED"],
            "Empty": CHART_COLORS["EMPTY"],
        },
        hole=0.62,
    )
    figure.update_layout(template=CHART_TEMPLATE, margin=dict(t=20, b=20, l=12, r=12))
    return figure


def occupancy_bar(occupied: int, empty: int) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=["Occupied", "Empty"],
            y=[occupied, empty],
            marker_color=[CHART_COLORS["OCCUPIED"], CHART_COLORS["EMPTY"]],
            text=[occupied, empty],
            textposition="outside",
        )
    )
    figure.update_layout(
        template=CHART_TEMPLATE,
        yaxis_title="Parking spaces",
        xaxis_title="",
        margin=dict(t=25, b=20, l=20, r=20),
    )
    return figure


def history_chart(history: list[dict[str, Any]]) -> go.Figure | None:
    frame = history_frame(history)
    if frame.empty:
        return None
    frame = frame.sort_values("detected_at")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["detected_at"],
            y=frame["occupancy_percentage"],
            mode="lines+markers",
            name="Occupancy",
            line=dict(color="#22d3ee", width=3),
            customdata=frame[["image_name", "total_spaces"]],
            hovertemplate=(
                "%{x|%b %d, %Y %H:%M}<br>Occupancy: %{y:.1f}%"
                "<br>Image: %{customdata[0]}<br>Spaces: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        template=CHART_TEMPLATE,
        yaxis_title="Occupancy (%)",
        xaxis_title="Detection time",
        yaxis=dict(range=[0, 100]),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return figure
