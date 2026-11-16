import streamlit as st

from modules.analytics import history_chart, occupancy_bar, occupancy_pie
from modules.database import get_history


def render() -> None:
    st.markdown('<div class="eyebrow">PARKING INSIGHTS</div>', unsafe_allow_html=True)
    st.title("Parking Analytics")
    history = get_history()
    latest = history[0] if history else None
    if latest is None:
        st.info("Analytics will be available after your first image analysis.")
        return
    st.caption(
        f"Metrics for the latest detection: {latest['image_name']} · "
        f"{latest['detected_at'].replace('T', ' ')}"
    )
    total, occupied, empty = (
        latest["total_spaces"],
        latest["occupied_spaces"],
        latest["empty_spaces"],
    )
    cards = st.columns(5)
    cards[0].metric("Total spaces", total)
    cards[1].metric("Occupied", occupied)
    cards[2].metric("Empty", empty)
    cards[3].metric("Occupancy", f"{latest['occupancy_percentage']:.1f}%")
    cards[4].metric("Availability", f"{latest['availability_percentage']:.1f}%")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Occupied vs empty")
        st.plotly_chart(occupancy_pie(occupied, empty), width="stretch")
    with right:
        st.markdown("#### Parking spaces")
        st.plotly_chart(occupancy_bar(occupied, empty), width="stretch")
    st.markdown("#### Detection history")
    chart = history_chart(history)
    if chart is not None:
        st.plotly_chart(chart, width="stretch")
