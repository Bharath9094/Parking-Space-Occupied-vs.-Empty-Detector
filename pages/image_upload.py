import hashlib
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from streamlit.elements import image as streamlit_image

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

if st_canvas is not None and not hasattr(streamlit_image, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def _canvas_image_to_url(
        image: Any,
        width: int,
        clamp: bool,
        channels: str,
        output_format: str,
        image_id: str,
    ) -> str:
        return image_to_url(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )

    streamlit_image.image_to_url = _canvas_image_to_url

from config import (
    DEFAULT_OVERLAP_THRESHOLD,
    MAX_CANVAS_WIDTH,
    MAX_IMAGE_DIMENSION,
    OUTPUTS_DIR,
)
from modules.database import get_layout, save_detection, save_layout
from modules.image_processing import (
    annotate_image,
    decode_image,
    image_quality_message,
    prepare_image,
    to_pil_rgb,
)
from modules.parking_detection import (
    classify_regions,
    detect_parking_regions,
    normalize_regions,
    regions_from_normalized,
)
from modules.vehicle_detection import detect_vehicles


def _clear_image_state(image_hash: str) -> None:
    for key in (
        "detection_result",
        "upload_error",
        "vehicles",
        "proposals",
        "source_image",
        "layout_check_key",
    ):
        st.session_state.pop(key, None)
    st.session_state.upload_hash = image_hash
    st.session_state.detection_hash = None


def _persist_result(
    image: np.ndarray,
    image_name: str,
    regions: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    spaces = classify_regions(regions, vehicles, threshold)
    annotated = annotate_image(image, spaces, vehicles)
    output_path = OUTPUTS_DIR / f"{uuid.uuid4().hex}.jpg"
    if not cv2.imwrite(str(output_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError("The annotated image could not be saved.")
    occupied = sum(space["status"] == "OCCUPIED" for space in spaces)
    record = save_detection(image_name, len(spaces), occupied, output_path)
    result = {
        **record,
        "spaces": spaces,
        "vehicles": vehicles,
        "annotated": annotated,
        "source_image": image.copy(),
        "overlap_threshold": threshold,
    }
    st.session_state.detection_result = result
    return result


def _try_persist_result(
    image: np.ndarray,
    image_name: str,
    regions: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    threshold: float,
) -> bool:
    try:
        _persist_result(image, image_name, regions, vehicles, threshold)
        return True
    except Exception as exc:
        st.error(f"The detection result could not be saved: {exc}")
        return False


def _image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get_fabric_rectangles(canvas_data: dict[str, Any] | None) -> list[dict[str, float]]:
    if not canvas_data:
        return []
    rectangles = []
    for item in canvas_data.get("objects", []):
        if item.get("type") != "rect":
            continue
        left = float(item.get("left", 0))
        top = float(item.get("top", 0))
        scale_x = float(item.get("scaleX", 1))
        scale_y = float(item.get("scaleY", 1))
        rect_width = float(item.get("width", 0)) * scale_x
        rect_height = float(item.get("height", 0)) * scale_y
        if rect_width > 8 and rect_height > 8:
            rectangles.append(
                {
                    "x1": max(0.0, left),
                    "y1": max(0.0, top),
                    "x2": left + rect_width,
                    "y2": top + rect_height,
                }
            )
    return rectangles


def _canvas_regions(
    rectangles: list[dict[str, float]],
    canvas_width: int,
    canvas_height: int,
    original_width: int,
    original_height: int,
) -> list[dict[str, Any]]:
    regions = []
    for rectangle in rectangles:
        x1 = round(rectangle["x1"] / canvas_width * original_width)
        x2 = round(rectangle["x2"] / canvas_width * original_width)
        y1 = round(rectangle["y1"] / canvas_height * original_height)
        y2 = round(rectangle["y2"] / canvas_height * original_height)
        x1, x2 = sorted((max(0, min(original_width, x1)), max(0, min(original_width, x2))))
        y1, y2 = sorted((max(0, min(original_height, y1)), max(0, min(original_height, y2))))
        if x2 > x1 and y2 > y1:
            regions.append({"bbox": (x1, y1, x2, y2), "confidence": None})
    return regions


def _show_detection_result(result: dict[str, Any]) -> None:
    total = result["total_spaces"]
    occupied = result["occupied_spaces"]
    empty = result["empty_spaces"]
    left, right = st.columns([1.3, 1])
    with left:
        st.image(
            cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB),
            caption="Analyzed image — green: empty · red: occupied",
            width="stretch",
        )
    with right:
        st.markdown("#### Detection summary")
        metrics = st.columns(2)
        metrics[0].metric("Total spaces", total)
        metrics[1].metric("Occupied", occupied)
        metrics[0].metric("Empty", empty)
        metrics[1].metric("Occupancy", f"{result['occupancy_percentage']:.1f}%")
        st.metric("Availability", f"{result['availability_percentage']:.1f}%")
        st.caption(
            f"Detected {len(result['vehicles'])} vehicle(s) · "
            f"{result['detected_at'].replace('T', ' ')}"
        )
        st.download_button(
            "Download processed image (PNG)",
            data=cv2.imencode(".png", result["annotated"])[1].tobytes(),
            file_name=f"{Path(result['image_name']).stem}_analyzed.png",
            mime="image/png",
            width="stretch",
        )


def render() -> None:
    st.markdown('<div class="eyebrow">IMAGE ANALYSIS</div>', unsafe_allow_html=True)
    st.title("Upload Parking Image")
    st.caption("Upload a still image only. No camera, video, or streaming input is used.")
    viewpoint = st.text_input(
        "Parking view name",
        placeholder="For example: North lot entrance",
        help="Name this fixed viewpoint to reuse manually marked spaces on later images.",
    )
    threshold = st.slider(
        "Vehicle overlap threshold",
        min_value=0.05,
        max_value=0.80,
        value=DEFAULT_OVERLAP_THRESHOLD,
        step=0.05,
        help="A slot is occupied when a vehicle box covers at least this fraction of its area.",
    )
    uploaded_file = st.file_uploader(
        "Choose a parking-lot image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG.",
    )
    if uploaded_file is None:
        st.markdown(
            '<div class="upload-hint">Drop a JPG or PNG parking image here, or browse to select one.'
            '<br><span>For best results, use a clear, elevated view of the parking spaces.</span></div>',
            unsafe_allow_html=True,
        )
        return

    image_bytes = uploaded_file.getvalue()
    current_hash = _image_hash(image_bytes)
    if st.session_state.get("upload_hash") != current_hash:
        _clear_image_state(current_hash)
        try:
            image = decode_image(image_bytes)
            image = prepare_image(image, MAX_IMAGE_DIMENSION)
            st.session_state.source_image = image
            st.session_state.vehicles = detect_vehicles(image)
            st.session_state.proposals = detect_parking_regions(image)
            st.session_state.upload_error = None
        except (ValueError, RuntimeError) as exc:
            st.session_state.upload_error = str(exc)
        except Exception as exc:
            st.session_state.upload_error = f"Image analysis could not be completed: {exc}"

    if st.session_state.get("upload_error"):
        st.error(st.session_state.upload_error)
        return

    image = st.session_state.source_image
    layout_key = (current_hash, viewpoint.strip())
    if st.session_state.get("layout_check_key") != layout_key:
        st.session_state.layout_check_key = layout_key
        try:
            saved_layout = get_layout(viewpoint)
            if saved_layout:
                height, width = image.shape[:2]
                saved_regions = regions_from_normalized(saved_layout, width, height)
                if saved_regions:
                    _persist_result(
                        image,
                        uploaded_file.name,
                        saved_regions,
                        st.session_state.vehicles,
                        threshold,
                    )
                    st.session_state.detection_hash = current_hash
                    st.success(
                        f"Analyzed using the saved '{viewpoint.strip()}' parking layout."
                    )
        except Exception as exc:
            st.error(f"The saved parking layout could not be applied: {exc}")
            return

    warning = image_quality_message(image)
    if warning:
        st.warning(warning)
    st.markdown("#### Original image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
    st.caption(
        f"Image size: {image.shape[1]} × {image.shape[0]} · "
        f"Vehicle detections: {len(st.session_state.vehicles)}"
    )
    if not st.session_state.vehicles:
        st.info(
            "No supported vehicles were detected. This can happen when vehicles are "
            "small, obscured, or absent; spaces will be classified using the detections available."
        )
    result = st.session_state.get("detection_result")
    if result and st.session_state.get("detection_hash") == current_hash:
        if result["overlap_threshold"] != threshold:
            st.info("The overlap threshold has changed since this image was analyzed.")
            if st.button("Re-analyze with selected threshold"):
                if _try_persist_result(
                    image,
                    uploaded_file.name,
                    result["spaces"],
                    st.session_state.vehicles,
                    threshold,
                ):
                    st.session_state.detection_hash = current_hash
                    st.rerun()
        st.markdown("#### Analyzed result")
        _show_detection_result(result)
        return

    st.markdown("#### Identify parking spaces")
    st.info(
        "The uploaded image has been checked for vehicles and possible parking-space outlines. "
        "Automatic space outlines are proposals and should be reviewed; draw spaces manually "
        "when the proposals do not match the painted stalls."
    )
    proposals = st.session_state.proposals
    if proposals:
        preview = image.copy()
        for index, region in enumerate(proposals, 1):
            x1, y1, x2, y2 = region["bbox"]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 210, 255), 2)
            cv2.putText(
                preview,
                f"Proposal {index}",
                (x1, max(20, y1 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 210, 255),
                2,
                cv2.LINE_AA,
            )
        st.image(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            caption=f"{len(proposals)} proposed parking region(s) — review before accepting",
            width="stretch",
        )
        if st.button("Analyze proposed parking spaces", type="primary"):
            if _try_persist_result(
                image, uploaded_file.name, proposals, st.session_state.vehicles, threshold
            ):
                st.session_state.detection_hash = current_hash
                st.rerun()

    st.markdown("##### Or draw the parking spaces")
    st.caption(
        "Draw one rectangle around each parking stall. Select and delete an incorrect "
        "rectangle before submitting. If you named this viewpoint, you can save the layout."
    )
    if st_canvas is None:
        st.error(
            "Manual parking-space drawing requires streamlit-drawable-canvas. "
            "Install the project dependencies with 'pip install -r requirements.txt'."
        )
        return
    height, width = image.shape[:2]
    scale = min(1.0, MAX_CANVAS_WIDTH / width)
    canvas_width = max(1, round(width * scale))
    canvas_height = max(1, round(height * scale))
    canvas_image = to_pil_rgb(
        cv2.resize(image, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)
        if scale < 1
        else image
    )
    canvas = st_canvas(
        fill_color="rgba(34, 211, 238, 0.15)",
        stroke_width=3,
        stroke_color="#22d3ee",
        background_image=canvas_image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
        key=f"parking_canvas_{current_hash[:12]}",
    )
    save_for_view = st.checkbox(
        "Save drawn spaces for this parking view",
        value=bool(viewpoint.strip()),
        disabled=not viewpoint.strip(),
    )
    if st.button("Analyze drawn parking spaces", type="primary"):
        rectangles = _get_fabric_rectangles(canvas.json_data)
        regions = _canvas_regions(
            rectangles, canvas_width, canvas_height, width, height
        )
        if not regions:
            st.error("Draw at least one valid rectangular parking space before analyzing.")
        else:
            if save_for_view:
                save_layout(viewpoint, normalize_regions(regions, width, height))
            if _try_persist_result(
                image, uploaded_file.name, regions, st.session_state.vehicles, threshold
            ):
                st.session_state.detection_hash = current_hash
                st.rerun()
