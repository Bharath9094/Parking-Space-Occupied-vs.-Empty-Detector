import streamlit as st

from modules.database import get_history
from modules.report_generator import history_csv


def render() -> None:
    st.markdown('<div class="eyebrow">PAST SESSIONS</div>', unsafe_allow_html=True)
    st.title("Detection History")
    history = get_history()
    if not history:
        st.info("No detection sessions have been saved yet.")
        return
    st.download_button(
        "Download history as CSV",
        data=history_csv(history),
        file_name="parking_detection_history.csv",
        mime="text/csv",
    )
    st.dataframe(
        [
            {
                "Detection ID": item["id"],
                "Uploaded image": item["image_name"],
                "Date & time": item["detected_at"].replace("T", " "),
                "Total spaces": item["total_spaces"],
                "Occupied": item["occupied_spaces"],
                "Empty": item["empty_spaces"],
                "Occupancy": f"{item['occupancy_percentage']:.1f}%",
                "Availability": f"{item['availability_percentage']:.1f}%",
            }
            for item in history
        ],
        width="stretch",
        hide_index=True,
    )
