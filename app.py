import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, os, time, base64
import requests
from pymongo import MongoClient, errors
import gridfs
from datetime import datetime

# ---------- AUTHENTICATION (Sign up / Sign in) ----------
import bcrypt
from pymongo.errors import DuplicateKeyError

# Make sure a 'users' collection exists and has a unique index on username
users_coll = db.get_collection("users")
try:
    # create unique index (safe to call repeatedly)
    users_coll.create_index("username", unique=True)
except Exception:
    pass

# Helper: hash password
def hash_password(plain_password: str) -> bytes:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())

# Helper: verify password
def check_password(plain_password: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed)
    except Exception:
        return False

# Initialize session state
if "user" not in st.session_state:
    st.session_state["user"] = None  # stores username when logged in

# Auth UI: small panel at top-right (or adjust placement)
with st.sidebar.expander("Account"):
    # if logged in: show username + sign out
    if st.session_state.get("user"):
        st.write(f"Logged in as: **{st.session_state['user']}**")
        if st.button("Sign out"):
            st.session_state["user"] = None
            st.success("Signed out.")
    else:
        auth_tab = st.radio("Choose action", ["Sign In", "Sign Up"])

        if auth_tab == "Sign Up":
            new_user = st.text_input("Username (signup)", key="su_user")
            new_pwd = st.text_input("Password (signup)", type="password", key="su_pwd")
            confirm_pwd = st.text_input("Confirm password", type="password", key="su_pwd2")

            if st.button("Create account"):
                if not new_user or not new_pwd:
                    st.error("Enter username and password.")
                elif new_pwd != confirm_pwd:
                    st.error("Passwords do not match.")
                else:
                    # hash and store
                    hashed = hash_password(new_pwd)
                    user_doc = {
                        "username": new_user,
                        "password": hashed,          # stored as bytes/BSON Binary
                        "created_at": datetime.utcnow()
                    }
                    try:
                        users_coll.insert_one(user_doc)
                        st.success("Account created. You can now sign in.")
                    except DuplicateKeyError:
                        st.error("Username already exists. Choose another.")
                    except Exception as e:
                        st.error(f"Failed to create account: {e}")

        else:  # Sign In
            user_in = st.text_input("Username (signin)", key="si_user")
            pwd_in = st.text_input("Password (signin)", type="password", key="si_pwd")
            if st.button("Sign in"):
                if not user_in or not pwd_in:
                    st.error("Enter username and password.")
                else:
                    user_doc = users_coll.find_one({"username": user_in})
                    if not user_doc:
                        st.error("User not found.")
                    else:
                        stored_hash = user_doc.get("password")
                        # pymongo may return stored_hash as Binary; ensure bytes
                        if isinstance(stored_hash, (bytes, bytearray)):
                            ok = check_password(pwd_in, stored_hash)
                        else:
                            # if stored as str for some reason, convert
                            try:
                                ok = check_password(pwd_in, bytes(stored_hash))
                            except Exception:
                                ok = False
                        if ok:
                            st.session_state["user"] = user_in
                            st.success(f"Signed in as {user_in}")
                        else:
                            st.error("Incorrect password.")
# ---------- END AUTH ----------


st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo")

st.title("Microscopy Detector (ONNX via Ultralytics + MongoDB storage)")

MODEL_LOCAL_PATH = "best.onnx"
GDRIVE_FILE_ID = ""
MODEL_IMG_SIZE = 1024
DEFAULT_CONF = 0.25


def get_mongo_uri():
    try:
        mongo_conf = st.secrets.get("mongo")
        if mongo_conf and "uri" in mongo_conf:
            return mongo_conf["uri"]
    except Exception:
        pass
    return os.environ.get("MONGO_URI")


MONGO_URI = get_mongo_uri()
USE_DB = bool(MONGO_URI)


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


def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            return (len(text) * 6, 11)


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
                score = float(box.conf[0])
            except:
                score = float(getattr(box, "confidence", 0.0))

            if score < conf_thresh:
                continue

            try:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
            except:
                continue

            try:
                cls = int(box.cls[0])
            except:
                cls = 0

            label = model_names[cls] if model_names and cls < len(model_names) else str(cls)
            counts[label] = counts.get(label, 0) + 1

            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            text = f"{label} {score:.2f}"
            tw, th = get_text_size(draw, text, font)
            ty = max(0, y1 - th)
            draw.rectangle([x1, ty, x1 + tw, y1], fill=(255, 0, 0))
            draw.text((x1, ty), text, fill="white", font=font)

    return pil_img, counts


client = None
db = None
fs = None
collection = None
db_error_msg = None

if USE_DB:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client["microscopy_db"]
        fs = gridfs.GridFS(db)
        collection = db["detections"]
    except errors.OperationFailure:
        db_error_msg = "MongoDB auth failed. Check username/password & user permissions."
    except errors.ServerSelectionTimeoutError:
        db_error_msg = "Cannot connect to MongoDB Atlas. Add IP 0.0.0.0/0 to Network Access."
    except Exception as e:
        db_error_msg = f"MongoDB connection error: {e}"


if GDRIVE_FILE_ID:
    try:
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
    except Exception as e:
        st.error(f"Google Drive download failed: {e}")

with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        model_names = getattr(model, "names", None)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

st.success("Model loaded successfully!")


st.header("Run Detection")

conf = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONF)
uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"])
camera = st.camera_input("Or take a picture")


if uploaded or camera:
    img_bytes = uploaded.read() if uploaded else camera.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(pil_img, caption="Input Image", width=450)

    if st.button("Run inference"):
        start = time.time()

        try:
            results = model.predict(np.array(pil_img), imgsz=MODEL_IMG_SIZE, conf=conf, verbose=False)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

        pil_out, counts = draw_predictions(pil_img.copy(), results, conf_thresh=conf, model_names=model_names)
        st.image(pil_out, caption="Detections", width=650)
        st.write("Counts:", counts)
        st.success(f"Done in {time.time() - start:.2f}s")

        if USE_DB:
            if db_error_msg:
                st.error(db_error_msg)
            else:
                try:
                    buf = io.BytesIO()
                    pil_out.save(buf, format="PNG")
                    img_bytes_out = buf.getvalue()

                    file_id = fs.put(img_bytes_out, filename=f"det_{int(time.time())}.png")

                    doc = {
                        "timestamp": datetime.utcnow(),
                        "counts": counts,
                        "img_gridfs_id": file_id,
                    }

                    inserted = collection.insert_one(doc)
                    st.success(f"Saved to MongoDB (doc_id: {inserted.inserted_id})")
                except Exception as e:
                    st.error(f"Failed to save to DB: {e}")

else:
    st.info("Upload an image to begin.")
