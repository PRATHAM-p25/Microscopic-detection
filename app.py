# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, os, time, base64
import requests
from pymongo import MongoClient, errors
import gridfs
from datetime import datetime

st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo")

st.title("Microscopy Detector (ONNX via Ultralytics + MongoDB storage)")

# ---------------- Settings ----------------
MODEL_LOCAL_PATH = "best.onnx"   # change if your model is in a subfolder, e.g. "models/best.onnx"
GDRIVE_FILE_ID = ""             # optional: Google Drive file id if model hosted on Drive
MODEL_IMG_SIZE = 1024
DEFAULT_CONF = 0.25
# ------------------------------------------

# Helper to safely get Mongo URI from secrets or env
def get_mongo_uri():
    # First try Streamlit secrets
    try:
        mongo_conf = st.secrets.get("mongo")
        if mongo_conf and "uri" in mongo_conf:
            return mongo_conf["uri"]
    except Exception:
        pass
    # Next, environment variable
    return os.environ.get("MONGO_URI")

MONGO_URI = get_mongo_uri()
USE_DB = bool(MONGO_URI)

# Download helper (small files)
def download_from_gdrive(file_id, dest):
    if os.path.exists(dest):
        return dest
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    return dest

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# draw text utility (works across PIL versions)
def get_text_size(draw, text, font):
    # prefer textbbox
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            try:
                return font.getsize(text)
            except Exception:
                return (len(text)*6, 11)

def draw_predictions(pil_img, results, conf_thresh=0.25, model_names=None):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    counts = {}
    # results: list of ultralytics Results
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        # ultralytics Boxes object supports iteration
        for box in boxes:
            # get confidence & class robustly
            try:
                score = float(box.conf[0]) if hasattr(box, "conf") else float(box.confidence)
            except Exception:
                score = float(getattr(box, "confidence", 0.0))
            try:
                cls = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            except Exception:
                cls = int(getattr(box, "class_id", 0))
            if score < conf_thresh:
                continue
            # xyxy extraction
            try:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
            except Exception:
                # fallback if attributes different
                coords = getattr(box, "xyxy", None)
                if coords is not None:
                    x1, y1, x2, y2 = coords[0].tolist()
                else:
                    continue
            label = (model_names[cls] if model_names and cls < len(model_names) else str(cls))
            counts[label] = counts.get(label, 0) + 1
            # draw
            draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
            text = f"{label} {score:.2f}"
            tw, th = get_text_size(draw, text, font)
            # ensure we don't draw outside image
            ty1 = max(0, y1 - th)
            draw.rectangle([x1, ty1, x1 + tw, y1], fill=(255,0,0))
            draw.text((x1, ty1), text, fill=(255,255,255), font=font)
    return pil_img, counts

# ------- Try to initialize DB (GridFS) -------
client = None
db = None
fs = None
collection = None
db_error_msg = None
if USE_DB:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # trigger server selection to detect auth/whitelist issues early
        client.server_info()
        db = client["microscopy_db"]
        fs = gridfs.GridFS(db)
        collection = db["detections"]
    except errors.OperationFailure as e:
        db_error_msg = ("MongoDB auth failure. Check username/password and user privileges. "
                        "Also ensure the user has write rights to the target DB/collection.")
    except errors.ServerSelectionTimeoutError as e:
        db_error_msg = ("Could not connect to MongoDB Atlas. This often means your IP is not whitelisted. "
                        "For testing add 0.0.0.0/0 to Network Access (temporarily) or add Streamlit Cloud IPs.")
    except Exception as e:
        db_error_msg = f"MongoDB connection error: {e}"

# ---- Model download if needed and load ----
if GDRIVE_FILE_ID:
    try:
        st.info("Downloading model from Google Drive...")
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
        st.success("Downloaded model.")
    except Exception as e:
        st.error(f"Downloading model failed: {e}")

with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        model_names = getattr(model, "names", None)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# --------------- UI ---------------
col1, col2 = st.columns([1, 1.2])
with col1:
    st.header("Run detection")
    conf = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONF)
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
            try:
                results = model.predict(source=np.array(pil_img), imgsz=MODEL_IMG_SIZE, conf=conf, verbose=False)
            except Exception as e:
                st.error(f"Model inference failed: {e}")
                st.stop()

            pil_out, counts = draw_predictions(pil_img.copy(), results, conf_thresh=conf, model_names=model_names)
            st.image(pil_out, caption="Detections", use_column_width=True)
            st.write("Counts:", counts)
            st.success(f"Inference done in {time.time()-start:.2f}s")

            # ---- Save to DB ----
            if not USE_DB:
                st.info("Mongo URI not provided. Skipping DB save. To enable DB storage, add your URI to Streamlit secrets or MONGO_URI env var.")
            else:
                if db_error_msg:
                    st.error(db_error_msg)
                else:
                    try:
                        # Save image bytes to GridFS
                        buf = io.BytesIO()
                        pil_out.save(buf, format="PNG")
                        img_bytes_out = buf.getvalue()

                        file_id = fs.put(img_bytes_out, filename=f"det_{int(time.time())}.png", contentType="image/png")
                        document = {
                            "timestamp": datetime.utcnow(),
                            "counts": counts,
                            "model": MODEL_LOCAL_PATH,
                            "img_gridfs_id": file_id,
                        }
                        insertion_result = collection.insert_one(document)
                        st.success(f"Saved detection to DB. doc_id: {insertion_result.inserted_id}")
                    except Exception as e:
                        st.error(f"Failed to save to DB: {e}")

with col2:
    st.header("Recent saved detections (latest 10)")
    if not USE_DB:
        st.info("DB not configured. Set your MongoDB URI in Streamlit secrets to enable saved history.")
    elif db_error_msg:
        st.error(db_error_msg)
    else:
        try:
            docs = list(collection.find().sort("timestamp", -1).limit(10))
            if not docs:
                st.info("No saved detections yet.")
            else:
                for doc in docs:
                    st.write(f"ID: {doc.get('_id')}, Time: {doc.get('timestamp')}, Counts: {doc.get('counts')}")
                    gfid = doc.get("img_gridfs_id")
                    if gfid:
                        try:
                            grid_out = fs.get(gfid)
                            data = grid_out.read()
                            img = Image.open(io.BytesIO(data))
                            st.image(img, width=300)
                        except Exception as e:
                            st.text(f"Could not read GridFS file {gfid}: {e}")
        except Exception as e:
            st.error(f"Failed to load recent docs: {e}")

# Footer: helpful tips for DB
st.markdown("---")
st.markdown("**MongoDB notes / troubleshooting**")
st.markdown("""
- If you see `auth failure` -> double-check username/password and that the user has access to the `microscopy_db` database (or use admin DB).
- If you see connection timeout -> whitelist your IP or use `0.0.0.0/0` temporarily under Network Access in Atlas.
- To store URI securely in Streamlit Cloud: go to *Manage app → Settings → Secrets* and add: """)
