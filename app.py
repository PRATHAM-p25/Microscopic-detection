# app.py
"""
Microscopy Detector (ONNX via Ultralytics) with Sign-up / Sign-in and MongoDB storage.
- Sign-up / Sign-in stores user records in MongoDB (collection: users).
- Uses bcrypt for password hashing.
- Saves detection images to GridFS and an entry to `detections` collection (only for signed-in users).
- If you want to enable DB, set Streamlit secret: [mongo] uri = "<your mongodb+srv uri>"
  or set environment variable MONGO_URI.
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, os, time, base64, requests
from pymongo import MongoClient, errors
import gridfs
from datetime import datetime
import bcrypt

# ---------------------------
# Configuration / Constants
# ---------------------------
st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo (Auth + MongoDB)")
st.title("Microscopy Detector (ONNX via Ultralytics + MongoDB storage)")

MODEL_LOCAL_PATH = "best.onnx"   # put best.onnx next to app.py or provide a Google Drive id below
GDRIVE_FILE_ID = ""              # optional: public google drive file id if you want app to download model at start
MODEL_IMG_SIZE = 1024
DEFAULT_CONF = 0.25

# ---------------------------
# Helper: get Mongo URI
# ---------------------------
def get_mongo_uri():
    # Try Streamlit secrets first, then environment variable fallback
    try:
        mongo_conf = st.secrets.get("mongo")
        if mongo_conf and "uri" in mongo_conf:
            return mongo_conf["uri"]
    except Exception:
        pass
    return os.environ.get("MONGO_URI")

MONGO_URI = get_mongo_uri()
USE_DB = bool(MONGO_URI)

# ---------------------------
# Model download helper
# ---------------------------
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

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# ---------------------------
# Text size helper (robust)
# ---------------------------
def get_text_size(draw, text, font):
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

# ---------------------------
# Draw detections
# ---------------------------
def draw_predictions(pil_img, results, conf_thresh=0.25, model_names=None):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    counts = {}
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
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
            try:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
            except Exception:
                coords = getattr(box, "xyxy", None)
                if coords is not None:
                    x1, y1, x2, y2 = coords[0].tolist()
                else:
                    continue
            label = (model_names[cls] if model_names and cls < len(model_names) else str(cls))
            counts[label] = counts.get(label, 0) + 1
            draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
            text = f"{label} {score:.2f}"
            tw, th = get_text_size(draw, text, font)
            ty1 = max(0, y1 - th)
            draw.rectangle([x1, ty1, x1 + tw, y1], fill=(255,0,0))
            draw.text((x1, ty1), text, fill=(255,255,255), font=font)
    return pil_img, counts

# ---------------------------
# DB Setup (GridFS + collections)
# ---------------------------
client = None
db = None
fs = None
collection = None
users_collection = None
db_error_msg = None

if USE_DB:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # will throw if cannot connect/auth
        db = client["microscopy_db"]
        fs = gridfs.GridFS(db)
        collection = db["detections"]
        users_collection = db["users"]
    except errors.OperationFailure as e:
        db_error_msg = ("MongoDB auth failure. Check username/password and user privileges. "
                        "Ensure the user has write rights to the DB/collection.")
    except errors.ServerSelectionTimeoutError as e:
        db_error_msg = ("Could not connect to MongoDB Atlas. This often means your IP is not whitelisted. "
                        "For testing add 0.0.0.0/0 to Network Access (temporarily) or add Streamlit Cloud IPs.")
    except Exception as e:
        db_error_msg = f"MongoDB connection error: {e}"

# ---------------------------
# Authentication helpers
# ---------------------------
def hash_password(plain_text_password: str) -> bytes:
    return bcrypt.hashpw(plain_text_password.encode("utf-8"), bcrypt.gensalt())

def check_password(plain_text_password: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(plain_text_password.encode("utf-8"), hashed)
    except Exception:
        return False

def sign_up_user(username: str, password: str):
    if not USE_DB:
        return False, "Database not configured."
    if db_error_msg:
        return False, db_error_msg
    if users_collection.find_one({"username": username}):
        return False, "Username already exists."
    hashed = hash_password(password)
    user_doc = {
        "username": username,
        "password": hashed,  # stored as bytes; PyMongo handles binary
        "created_at": datetime.utcnow()
    }
    users_collection.insert_one(user_doc)
    return True, "User created."

def authenticate_user(username: str, password: str):
    if not USE_DB:
        return False, "Database not configured."
    if db_error_msg:
        return False, db_error_msg
    user = users_collection.find_one({"username": username})
    if not user:
        return False, "User not found."
    stored = user.get("password")
    if isinstance(stored, str):
        # If password stored as hex/str by mistake, convert attempt
        try:
            stored = stored.encode("utf-8")
        except Exception:
            pass
    ok = check_password(password, stored)
    if ok:
        return True, user
    return False, "Invalid credentials."

# ---------------------------
# Download model if requested
# ---------------------------
if GDRIVE_FILE_ID:
    try:
        st.info("Downloading model from Google Drive...")
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
        st.success("Downloaded model.")
    except Exception as e:
        st.error(f"Downloading model failed: {e}")

# ---------------------------
# Load model and names
# ---------------------------
with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        model_names = getattr(model, "names", None)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ---------------------------
# Authentication UI (sidebar)
# ---------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None

st.sidebar.header("Account")
auth_mode = st.sidebar.radio("Choose", ["Sign In", "Sign Up", "Guest"], index=0)

if auth_mode == "Sign Up":
    su_username = st.sidebar.text_input("Create username", key="su_user")
    su_password = st.sidebar.text_input("Create password", type="password", key="su_pass")
    if st.sidebar.button("Sign Up"):
        ok, msg = sign_up_user(su_username, su_password)
        if ok:
            st.sidebar.success("Account created. You can now sign in.")
        else:
            st.sidebar.error(msg)

elif auth_mode == "Sign In":
    si_username = st.sidebar.text_input("Username", key="si_user")
    si_password = st.sidebar.text_input("Password", type="password", key="si_pass")
    if st.sidebar.button("Sign In"):
        ok, res = authenticate_user(si_username, si_password)
        if ok:
            st.session_state["user"] = {"username": si_username, "_id": res.get("_id")}
            st.sidebar.success(f"Signed in as {si_username}")
        else:
            st.sidebar.error(res)
elif auth_mode == "Guest":
    if st.sidebar.button("Continue as Guest"):
        st.session_state["user"] = {"username": "guest"}

# Show logged-in user and sign out
if st.session_state.get("user"):
    st.sidebar.write("Signed in as:", st.session_state["user"].get("username"))
    if st.sidebar.button("Sign Out"):
        st.session_state["user"] = None
        st.sidebar.success("Signed out.")

# ---------------------------
# Main UI: Detection
# ---------------------------
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

            # Save to DB only if logged in and DB available
            if not st.session_state.get("user"):
                st.info("You are not signed in. Sign in to save this detection to DB.")
            else:
                if not USE_DB:
                    st.error("MongoDB URI not configured. Add to Streamlit secrets or set MONGO_URI env var.")
                elif db_error_msg:
                    st.error(db_error_msg)
                else:
                    try:
                        buf = io.BytesIO()
                        pil_out.save(buf, format="PNG")
                        img_bytes_out = buf.getvalue()
                        file_id = fs.put(img_bytes_out, filename=f"det_{int(time.time())}.png", contentType="image/png")

                        document = {
                            "timestamp": datetime.utcnow(),
                            "counts": counts,
                            "model": MODEL_LOCAL_PATH,
                            "img_gridfs_id": file_id,
                            "user": st.session_state["user"].get("username")
                        }
                        insertion_result = collection.insert_one(document)
                        st.success(f"Saved detection to DB. doc_id: {insertion_result.inserted_id}")
                    except Exception as e:
                        st.error(f"Failed to save to DB: {e}")

with col2:
    st.header("Info / DB status")
    if not USE_DB:
        st.info("MongoDB not configured. To enable DB, add your URI to Streamlit secrets under [mongo] uri or set MONGO_URI env var.")
    elif db_error_msg:
        st.error(db_error_msg)
    else:
        st.success("MongoDB connected.")

    st.markdown("**Notes**")
    st.markdown("""
    - Sign up stores username and a bcrypt-hashed password.
    - Only signed-in users can save detections to the DB.
    - Image bytes are saved in GridFS and metadata in `microscopy_db.detections`.
    - To configure MongoDB: in Streamlit Cloud -> Settings -> Secrets, add:
      ```
      [mongo]
      uri = "mongodb+srv://<user>:<pass>@.../yourDB?retryWrites=true&w=majority"
      ```
      or set MONGO_URI environment variable on your host.
    """)

# EOF
