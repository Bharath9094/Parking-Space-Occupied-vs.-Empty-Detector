import cv2
import streamlit as st


def render() -> None:
    st.markdown('<div class="eyebrow">ANALYSIS OUTPUT</div>', unsafe_allow_html=True)
    st.title("Detection Results")
    result = st.session_state.get("detection_result")
    if not result:
        st.info("Analyze an uploaded parking image to view its detection results here.")
        return
    st.caption(f"Source image: {result['image_name']}")
    original, analyzed = st.columns(2)
    with original:
        st.markdown("#### Original")
        st.image(cv2.cvtColor(result["source_image"], cv2.COLOR_BGR2RGB), width="stretch")
    with analyzed:
        st.markdown("#### AI-analyzed")
        st.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB), width="stretch")
    st.markdown(
        '<div class="legend"><span><i class="legend-dot green"></i> GREEN — EMPTY</span>'
        '<span><i class="legend-dot red"></i> RED — OCCUPIED</span>'
        '<span><i class="legend-dot blue"></i> BLUE — VEHICLE</span></div>',
        unsafe_allow_html=True,
    )
    first, second, third, fourth = st.columns(4)
    first.metric("Total spaces", result["total_spaces"])
    second.metric("Occupied", result["occupied_spaces"])
    third.metric("Empty", result["empty_spaces"])
    fourth.metric("Occupancy", f"{result['occupancy_percentage']:.1f}%")
    st.download_button(
        "Download annotated image",
        data=cv2.imencode(".png", result["annotated"])[1].tobytes(),
        file_name=f"{result['image_name'].rsplit('.', 1)[0]}_analyzed.png",
        mime="image/png",
    )
    if result["spaces"]:
        st.markdown("#### Per-space classification")
        st.dataframe(
            [
                {
                    "Parking space": space["label"],
                    "Status": space["status"],
                    "Vehicle": space["vehicle_class"] or "—",
                    "Vehicle confidence": (
                        f"{space['vehicle_confidence']:.0%}"
                        if space["vehicle_confidence"] is not None
                        else "—"
                    ),
                    "Space confidence": (
                        f"{space['confidence']:.0%}"
                        if space.get("confidence") is not None
                        else "Manual region"
                    ),
                }
                for space in result["spaces"]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("No parking spaces were identified in this detection.")
