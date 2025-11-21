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
import base64
from datetime import datetime
from pymongo import MongoClient, errors

st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo")

st.title("Microscopy Detector (ONNX via Ultralytics)")

# ------------- Settings -------------
MODEL_LOCAL_PATH = "best.onnx"
GDRIVE_FILE_ID = ""  # optional
MODEL_IMG_SIZE = 1024
CONF_THRESH = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25)
DB_NAME = "microscopy_db"
COLLECTION_NAME = "detections"
# ------------------------------------

# --- Get Mongo URI safely ---
# Try st.secrets first; fallback to environment variable
mongo_uri = None
try:
    mongo_uri = st.secrets.get("mongo", {}).get("uri")
except Exception:
    mongo_uri = None

if not mongo_uri:
    # also support an environment variable fallback (for local dev)
    mongo_uri = os.environ.get("MONGO_URI")

# Provide clear user message if missing
if not mongo_uri:
    st.warning("MongoDB URI not found. Add it to Streamlit Secrets (Manage app → Settings → Secrets) as [mongo] uri = \"...\" or set MONGO_URI env var for local dev.")
    db_client = None
else:
    # create client with short serverSelectionTimeoutMS to fail fast if network/auth fails
    try:
        db_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # quick ping to test connectivity & auth
        db_client.admin.command('ping')
        st.success("Connected to MongoDB Atlas.")
    except errors.ServerSelectionTimeoutError as e:
        st.error("Cannot connect to MongoDB. Possible reasons: IP not whitelisted, network blocked, or wrong URI/credentials.")
        st.write("Atlas error (short):", str(e))
        db_client = None
    except errors.OperationFailure as e:
        st.error("MongoDB authentication failed (OperationFailure). Check username/password and user privileges.")
        st.write("Atlas error (short):", str(e))
        db_client = None
    except Exception as e:
        st.error("Unexpected error while connecting to MongoDB.")
        st.write(str(e))
        db_client = None

# Prepare DB handles if connection ok
collection = None
if db_client:
    db = db_client[DB_NAME]
    collection = db[COLLECTION_NAME]

# --- model loading helpers ---
def download_from_gdrive(file_id, dest):
    if os.path.exists(dest):
        return dest
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download model from Drive (status {r.status_code})")
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    return dest

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def draw_predictions(pil_img, results, conf_thresh=0.25):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    counts = {}
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            # compat with ultralytics Boxes structure
            score = float(box.conf[0]) if hasattr(box, 'conf') else float(box.confidence)
            cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
            if score < conf_thresh:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = model.names[cls] if cls < len(model.names) else str(cls)
            counts[label] = counts.get(label, 0) + 1
            draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
            text = f"{label} {score:.2f}"
            bbox = draw.textbbox((0,0), text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            draw.rectangle([x1, y1-th, x1+tw, y1], fill=(255,0,0))
            draw.text((x1, y1-th), text, fill=(255,255,255), font=font)
    return pil_img, counts

# Model prep
if GDRIVE_FILE_ID:
    try:
        st.info("Downloading model from Google Drive...")
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
        st.success("Downloaded model.")
    except Exception as e:
        st.error("Failed to download model from Drive: " + str(e))

with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        st.success("Model loaded.")
    except Exception as e:
        st.error("Failed to load model (check model file).")
        st.write(str(e))
        model = None

# UI
uploaded = st.file_uploader("Upload microscope image", type=["png","jpg","jpeg","tif","tiff"])
camera = st.camera_input("Or take a picture (Chromium browsers)")

if uploaded is None and camera is None:
    st.info("Upload an image or use the camera.")
else:
    img_bytes = uploaded.read() if uploaded else camera.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(pil_img, caption="Input image", width=400)

    if st.button("Run inference"):
        if model is None:
            st.error("Model not loaded. Fix model path or GDRIVE_FILE_ID.")
        else:
            start = time.time()
            try:
                results = model.predict(source=np.array(pil_img), imgsz=MODEL_IMG_SIZE, conf=CONF_THRESH, verbose=False)
            except Exception as e:
                st.error("Model inference failed: " + str(e))
                results = []
            pil_out, counts = draw_predictions(pil_img.copy(), results, conf_thresh=CONF_THRESH)
            st.image(pil_out, caption="Detections", use_column_width=True)
            st.write("Counts:", counts)
            st.success(f"Inference done in {time.time()-start:.2f}s")

            # --- Convert output image to base64 for storing ---
            try:
                buffer = io.BytesIO()
                pil_out.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            except Exception as e:
                st.error("Failed to convert image to base64: " + str(e))
                img_base64 = None

            # --- Insert into MongoDB if available ---
            if collection is None:
                st.info("MongoDB not configured or connection failed. Skipping DB save.")
            else:
                document = {
                    "timestamp": datetime.utcnow(),
                    "counts": counts,
                    "image_base64": img_base64,
                    "model": os.path.basename(MODEL_LOCAL_PATH),
                    "img_size": MODEL_IMG_SIZE
                }
                try:
                    insertion_result = collection.insert_one(document)
                    st.success(f"Saved detection to DB, id: {insertion_result.inserted_id}")
                except errors.OperationFailure as e:
                    st.error("MongoDB OperationFailure: likely auth or IP whitelist issue.")
                    st.write("Atlas error (short):", str(e))
                except Exception as e:
                    st.error("Failed to insert document to MongoDB: " + str(e))
