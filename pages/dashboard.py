from pathlib import Path
from typing import Any

import cv2
import streamlit as st

from modules.analytics import history_chart, occupancy_pie
from modules.database import get_history


def _metric_card(label: str, value: str, accent: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{accent}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render() -> None:
    st.markdown('<div class="eyebrow">OVERVIEW</div>', unsafe_allow_html=True)
    st.title("Parking Dashboard")
    st.caption("Live insights from your uploaded-image detection sessions.")
    history: list[dict[str, Any]] = get_history()
    latest = history[0] if history else None
    if latest:
        values = [
            ("Total Parking Spaces", str(latest["total_spaces"]), "#67e8f9"),
            ("Occupied Spaces", str(latest["occupied_spaces"]), "#f87171"),
            ("Empty Spaces", str(latest["empty_spaces"]), "#4ade80"),
            ("Occupancy Rate", f"{latest['occupancy_percentage']:.1f}%", "#60a5fa"),
            ("Availability Rate", f"{latest['availability_percentage']:.1f}%", "#22d3ee"),
        ]
    else:
        values = [(name, "—", "#67e8f9") for name in (
            "Total Parking Spaces", "Occupied Spaces", "Empty Spaces",
            "Occupancy Rate", "Availability Rate"
        )]
    columns = st.columns(5)
    for column, (label, value, color) in zip(columns, values):
        with column:
            _metric_card(label, value, color)

    left, right = st.columns([1, 1.5])
    with left:
        st.markdown("#### Latest occupancy")
        if latest:
            st.plotly_chart(
                occupancy_pie(latest["occupied_spaces"], latest["empty_spaces"]),
                width="stretch",
            )
        else:
            st.info("Upload a parking image to see the first detection.")
    with right:
        st.markdown("#### Last analyzed image")
        if latest:
            image_path = Path(latest["output_path"])
            if image_path.is_file():
                image = cv2.imread(str(image_path))
                if image is not None:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
                else:
                    st.warning("The saved analysis image could not be read.")
            else:
                st.info("The annotated image is no longer available on this device.")
            st.caption(
                f"**{latest['image_name']}** · "
                f"{latest['detected_at'].replace('T', ' ')}"
            )
        else:
            st.markdown(
                '<div class="empty-state">Your latest annotated parking image will appear here.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Occupancy over time")
    chart = history_chart(history)
    if chart is not None:
        st.plotly_chart(chart, width="stretch")
    else:
        st.info("Detection history will appear here after you analyze an image.")

    if history:
        st.markdown("#### Recent detections")
        st.dataframe(
            [
                {
                    "Image": item["image_name"],
                    "Date & time": item["detected_at"].replace("T", " "),
                    "Spaces": item["total_spaces"],
                    "Occupied": item["occupied_spaces"],
                    "Empty": item["empty_spaces"],
                    "Occupancy": f"{item['occupancy_percentage']:.1f}%",
                }
                for item in history[:5]
            ],
            width="stretch",
            hide_index=True,
        )
