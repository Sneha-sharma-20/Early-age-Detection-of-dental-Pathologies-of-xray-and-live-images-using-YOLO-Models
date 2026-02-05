import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Dental Pathology Detection",
    layout="wide"
)

st.title("🦷 Dental Pathology Detection")

# --------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------

XRAY_CLASSES = {
    0: "Healthy Teeth",
    1: "Caries",
    2: "Impacted Teeth",
    3: "Broken Down Crown/Root",
    4: "Infection",
    5: "Fractured Teeth"
}

CAMERA_CLASSES = {
    0: "Caries",
    1: "Ulcer",
    2: "Tooth Discoloration",
    3: "Gingivitis"
}

# Colors (BGR)
COLORS = {
    "Healthy Teeth": (0, 255, 0),
    "Caries": (255, 0, 0),
    "Impacted Teeth": (0, 255, 255),
    "Broken Down Crown/Root": (255, 165, 0),
    "Infection": (0, 0, 255),
    "Fractured Teeth": (255, 0, 255),
    "Ulcer": (128, 0, 128),
    "Tooth Discoloration": (0, 128, 255),
    "Gingivitis": (0, 255, 128),
}

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_xray_model():
    return YOLO("models/xray_best.pt")

@st.cache_resource
def load_camera_model():
    return YOLO("models/camera_best.pt")

# --------------------------------------------------
# INPUT SELECTION
# --------------------------------------------------
input_type = st.radio(
    "Select Input Type",
    ["X-ray Image", "Camera Image"],
    horizontal=True
)

conf_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05
)

# --------------------------------------------------
# IMAGE INPUT
# --------------------------------------------------
uploaded = st.file_uploader(
    "Upload Dental Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    col1.image(image, caption="Input Image", use_container_width=True)

    if input_type == "X-ray Image":
        model = load_xray_model()
        class_map = XRAY_CLASSES
    else:
        model = load_camera_model()
        class_map = CAMERA_CLASSES

    with st.spinner("Running detection..."):
        results = model(img_bgr, conf=conf_threshold)

    output = img.copy()
    detections = results[0].boxes

    detected_labels = []

    if detections is not None and len(detections) > 0:
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = class_map.get(cls_id, f"class{cls_id}")
            detected_labels.append((label, conf))

            color = COLORS.get(label, (255, 255, 255))

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                f"{label} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        col2.image(output, caption="Detected Output", use_container_width=True)

    else:
        col2.image(output, caption="Detected Output", use_container_width=True)
        st.warning("No anomalies detected.")

    # --------------------------------------------------
    # CONFIDENCE SCORES
    # --------------------------------------------------
    st.subheader("📊 Confidence Scores")

    if len(detected_labels) == 0:
        st.write("No anomalies detected.")
    else:
        for lbl, cf in detected_labels:
            st.write(f"**{lbl}** → {cf:.2f}")

    # --------------------------------------------------
    # LEGEND
    # --------------------------------------------------
    st.subheader("🎨 Legend")

    legend_cols = st.columns(3)
    for i, (name, color) in enumerate(COLORS.items()):
        if input_type == "X-ray Image" and name not in XRAY_CLASSES.values():
            continue
        if input_type == "Camera Image" and name not in CAMERA_CLASSES.values():
            continue

        legend_cols[i % 3].markdown(
            f"<span style='color:rgb{color}; font-weight:bold;'>■</span> {name}",
            unsafe_allow_html=True
        )
