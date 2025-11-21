# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import requests
import io
import time

st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo")

st.title("Microscopy Detector (ONNX via Ultralytics)")

# ------------- Settings -------------
MODEL_LOCAL_PATH = "best.onnx"   # local fallback
# If you host on Google Drive, paste the file id here (optional)
GDRIVE_FILE_ID = ""  # e.g. "1a2B3cD4eF..." (leave empty if you put best.onnx beside app.py)
MODEL_IMG_SIZE = 1024  # set to the size you exported with (you exported with 1024)
CONF_THRESH = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25)
# ------------------------------------

def download_from_gdrive(file_id, dest):
    if os.path.exists(dest):
        return dest
    # direct download helper (works for small files)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download model from Drive: status {r.status_code}")
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    return dest

@st.cache_resource
def load_model(model_path):
    # This will accept both .pt and .onnx exported by Ultralytics
    model = YOLO(model_path)
    return model

def draw_predictions(pil_img, results, conf_thresh=0.25):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    counts = {}
    for r in results:
        boxes = r.boxes  # ultralytics Boxes object
        if boxes is None:
            continue
        for box in boxes:
            score = float(box.conf[0]) if hasattr(box, 'conf') else float(box.confidence)
            cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
            if score < conf_thresh:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # xyxy as floats
            label = model.names[cls] if cls < len(model.names) else str(cls)
            counts[label] = counts.get(label, 0) + 1
            # draw box
            draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
            text = f"{label} {score:.2f}"
            tw, th = draw.textsize(text, font=font)
            draw.rectangle([x1, y1-th, x1+tw, y1], fill=(255,0,0))
            draw.text((x1, y1-th), text, fill=(255,255,255), font=font)
    return pil_img, counts

# Model preparation: download if needed and load
if GDRIVE_FILE_ID:
    try:
        st.info("Downloading model from Google Drive...")
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
        st.success("Downloaded model.")
    except Exception as e:
        st.error(f"Failed to download model: {e}")

# Load model (cached)
with st.spinner("Loading model..."):
    model = load_model(MODEL_LOCAL_PATH)
st.success("Model loaded.")

# UI - image upload or camera
uploaded = st.file_uploader("Upload microscope image", type=["png","jpg","jpeg","tif","tiff"])
camera = st.camera_input("Or take a picture (Chromium browsers)")

if uploaded is None and camera is None:
    st.info("Upload an image or use the camera.")
else:
    img_bytes = uploaded.read() if uploaded else camera.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(pil_img, caption="Input image", width=400)

    if st.button("Run inference"):
        start = time.time()
        # ultralytics accepts PIL image or path; we pass image bytes
        # Use model.predict with appropriate imgsz (you exported with 1024)
        results = model.predict(source=np.array(pil_img), imgsz=MODEL_IMG_SIZE, conf=CONF_THRESH, verbose=False)
        # results is a list (one entry per image) of Results objects
        pil_out, counts = draw_predictions(pil_img.copy(), results, conf_thresh=CONF_THRESH)
        st.image(pil_out, caption="Detections", use_column_width=True)
        st.write("Counts:", counts)
        st.success(f"Inference done in {time.time()-start:.2f}s")
