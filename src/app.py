import os
import time
import json
import base64
import random
import shutil
import logging
from threading import Lock

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image
import wikipedia

from flask import Flask, request, render_template, url_for, session, redirect
from werkzeug.utils import secure_filename

# ✅ Translator
from deep_translator import GoogleTranslator

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BreedSnap")

# ✅ FIXED Flask init (templates + static dono root me accessible honge)
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
    static_url_path="/static"
)

# session secret for language setting
app.secret_key = os.environ.get("APP_SECRET", "change_this_secret_in_prod")

# ----------------
# Config
# ----------------
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# upload limits & allowed types
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

# ----------------
# Load classes (prefer classes.json; fallback to train folder)
# ----------------
CLASSES_JSON = os.path.join("data", "classes.json")
classes = None
if os.path.exists(CLASSES_JSON):
    try:
        with open(CLASSES_JSON, "r", encoding="utf-8") as f:
            classes = json.load(f)
        log.info(f"Loaded {len(classes)} classes from {CLASSES_JSON}")
    except Exception as e:
        log.warning(f"Failed to load {CLASSES_JSON}: {e}")

if classes is None:
    train_folder = os.path.join("data", "images", "train")
    if not os.path.exists(train_folder):
        raise FileNotFoundError(f"Train folder not found: {train_folder} (or provide data/classes.json)")
    train_data = datasets.ImageFolder(train_folder, transform=transform)
    classes = train_data.classes
    log.info(f"Found {len(classes)} classes from train folder.")

# ----------------
# Load model & weights
# ----------------
try:
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
except Exception:
    model = models.resnet18(pretrained=False)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))

MODEL_PATH = "breed_classifier.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

state = torch.load(MODEL_PATH, map_location=device)

if isinstance(state, dict) and "state_dict" in state:
    state_dict = state["state_dict"]
else:
    state_dict = state

fixed_state = {}
for k, v in state_dict.items():
    nk = k[len("module."):] if k.startswith("module.") else k
    fixed_state[nk] = v

model.load_state_dict(fixed_state)
model = model.to(device)
model.eval()
log.info("Model loaded and ready.")

# ----------------
# Breed JSON + Wiki cache
# ----------------
BREEDS_JSON_PATH = os.path.join("data", "breeds.json")
WIKI_CACHE_PATH = os.path.join("data", "wiki_cache.json")

if os.path.exists(BREEDS_JSON_PATH):
    try:
        with open(BREEDS_JSON_PATH, "r", encoding="utf-8") as f:
            raw_breed_data = json.load(f)
    except Exception:
        raw_breed_data = {}
else:
    raw_breed_data = {}

breed_data = {k.replace(" ", "_").lower(): v for k, v in raw_breed_data.items()}

wiki_cache = {}
if os.path.exists(WIKI_CACHE_PATH):
    try:
        with open(WIKI_CACHE_PATH, "r", encoding="utf-8") as f:
            wiki_cache = json.load(f)
    except Exception:
        wiki_cache = {}

_wiki_lock = Lock()

def _atomic_write_json(path, data):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log.warning(f"Atomic write failed for {path}: {e}")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except Exception:
            log.exception("Failed to persist wiki cache")

def get_breed_info(breed):
    breed_key = breed.replace(" ", "_").lower()
    if breed_key in wiki_cache:
        entry = wiki_cache[breed_key]
        return entry.get("summary"), entry.get("type", "Unknown"), entry.get("source", "Cache")

    search_queries = [breed + " cattle", breed + " cow", breed + " buffalo", breed]
    for query in search_queries:
        try:
            summary = wikipedia.summary(query, sentences=2, auto_suggest=False, redirect=True)
            if any(word in summary.lower() for word in ["city", "district", "state", "village", "municipality"]):
                continue
            animal_type = "Buffalo" if "buffalo" in summary.lower() else "Cow"
            entry = {"summary": summary, "type": animal_type, "source": "Wikipedia"}
            with _wiki_lock:
                wiki_cache[breed_key] = entry
                _atomic_write_json(WIKI_CACHE_PATH, wiki_cache)
            return summary, animal_type, "Wikipedia"
        except Exception as e:
            log.debug(f"Wikipedia lookup failed for '{query}': {e}")
            continue

    if breed_key in breed_data:
        entry = breed_data[breed_key]
        summary = entry.get("summary", "No reliable info found for this breed.")
        animal_type = entry.get("type", "Unknown")
        entry_out = {"summary": summary, "type": animal_type, "source": "Local JSON"}
        with _wiki_lock:
            wiki_cache[breed_key] = entry_out
            _atomic_write_json(WIKI_CACHE_PATH, wiki_cache)
        return summary, animal_type, "Local JSON"

    entry_out = {"summary": "No reliable info found for this breed.", "type": "Unknown", "source": "Local JSON"}
    with _wiki_lock:
        wiki_cache[breed_key] = entry_out
        _atomic_write_json(WIKI_CACHE_PATH, wiki_cache)
    return entry_out["summary"], entry_out["type"], entry_out["source"]

# ----------------
# ✅ Translation Helper
# ----------------
def translate_text(text, target_lang):
    if not text or target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        log.warning(f"Translation failed: {e}")
        return text

# ----------------
# Voice helpers
# ----------------
def find_train_folder_for_breed(breed_key):
    base = os.path.join("data", "images", "train")
    if not os.path.exists(base):
        return None
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if not os.path.isdir(p):
            continue
        norm = d.replace(" ", "_").lower()
        if norm == breed_key:
            return p
    return None

def ensure_static_breed_images(breed_key, n=3):
    dest_dir = os.path.join(app.static_folder, "breeds", breed_key)
    os.makedirs(dest_dir, exist_ok=True)
    existing = [f for f in os.listdir(dest_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(existing) >= n:
        return existing[:n]

    train_folder_path = find_train_folder_for_breed(breed_key)
    if not train_folder_path:
        log.info(f"No train folder found for breed_key='{breed_key}'")
        return existing

    train_imgs = [f for f in os.listdir(train_folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not train_imgs:
        log.info(f"No images in train folder for breed_key='{breed_key}' ({train_folder_path})")
        return existing

    random.shuffle(train_imgs)
    for img in train_imgs:
        if len(existing) >= n:
            break
        src = os.path.join(train_folder_path, img)
        dst = os.path.join(dest_dir, img)
        if not os.path.exists(dst):
            try:
                shutil.copy(src, dst)
                log.info(f"Copied {src} -> {dst}")
            except Exception as e:
                log.warning(f"Failed to copy {src} -> {dst}: {e}")
        existing = [f for f in os.listdir(dest_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return existing[:n]

# ----------------
# Utilities
# ----------------
def _is_allowed_filename(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS

def display_name(breed_key):
    return breed_key.replace("_", " ").title()

# ----------------
# Routes
# ----------------
@app.route("/")
def home():
    lang = session.get("lang", "en")
    return render_template("index.html", lang=lang)

@app.route("/set_language", methods=["POST"])
def set_language():
    lang = request.form.get("lang", "en")
    session["lang"] = lang
    return redirect(url_for("home"))

@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("files")  
    if not files or len(files) == 0:
        return render_template("index.html", error="No files uploaded. Please choose at least one image.", lang=session.get("lang", "en"))

    lang = session.get("lang", "en")
    results = []
    top1_breeds = set()

    for idx, file in enumerate(files):
        if not file or file.filename == "":
            continue

        orig = secure_filename(file.filename)
        if not _is_allowed_filename(orig):
            app.logger.warning(f"Skipped file with disallowed extension: {orig}")
            continue

        filename = f"{int(time.time())}_{idx}_{orig}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        try:
            file.save(filepath)
        except Exception as e:
            app.logger.error(f"Failed to save uploaded file {orig}: {e}")
            continue

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            app.logger.error(f"Saved file missing or empty: {filepath}")
            continue

        data_uri = None
        try:
            with open(filepath, "rb") as fh:
                blob = fh.read()
            with Image.open(filepath) as im:
                fmt = (im.format or "JPEG").lower()
            mime = f"image/{fmt if fmt else 'jpeg'}"
            b64 = base64.b64encode(blob).decode("ascii")
            data_uri = f"data:{mime};base64,{b64}"
        except Exception as e:
            app.logger.error(f"Failed to read/encode image {filepath}: {e}")
            data_uri = None

        try:
            pil_img = Image.open(filepath).convert("RGB")
        except Exception as e:
            app.logger.error(f"Failed to open image for prediction {filepath}: {e}")
            continue

        img_t = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            probs = F.softmax(outputs, dim=1)[0]
            topk = torch.topk(probs, k=min(3, probs.shape[0]))
            top_indices = topk.indices.cpu().numpy()
            top_values = topk.values.cpu().numpy()

        top_results = []
        for i, p in zip(top_indices, top_values):
            top_results.append({"breed": classes[int(i)], "prob": float(p)})

        top1 = top_results[0]["breed"]
        top1_breeds.add(top1)

        results.append({
            "image_filename": filename,
            "image_data": data_uri,
            "top_results": top_results,
            "predicted": top1
        })

    breed_info_map = {}
    for breed in top1_breeds:
        summary, animal_type, source = get_breed_info(breed)
        summary = translate_text(summary, lang)  # ✅ translate summary
        breed_info_map[breed] = {"summary": summary, "type": animal_type, "source": source}

    for r in results:
        b = r["predicted"]
        r["breed_info"] = breed_info_map.get(b, {"summary": "No info", "type": "Unknown", "source": "Local JSON"})
        r["predicted_display"] = display_name(r["predicted"])
        for t in r["top_results"]:
            t["breed_display"] = display_name(t["breed"])

    ts = int(time.time())
    return render_template("result.html", results=results, ts=ts, lang=lang)

# ----------------
# Voice Search Route
# ----------------
@app.route("/voice_search", methods=["POST"])
def voice_search():
    spoken_text = request.form.get("query", "").strip().lower()
    log.info(f"Voice search received: '{spoken_text}'")
    if not spoken_text:
        return render_template("breed_search.html", query="", error="No voice input detected", lang=session.get("lang", "en"))

    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in spoken_text)
    words = [w.strip() for w in cleaned.split() if w.strip()]
    stop_words = {"cow", "buffalo", "cattle", "breed"}
    keywords = [w for w in words if w not in stop_words]

    breed_key = None
    if keywords:
        candidate_full = "_".join(keywords).lower()
        if candidate_full in breed_data:
            breed_key = candidate_full
        else:
            for w in keywords:
                if w.lower() in breed_data:
                    breed_key = w.lower()
                    break

    if not breed_key:
        for w in words:
            c = w.lower()
            if c in breed_data:
                breed_key = c
                break

    log.info(f"[DEBUG] spoken_text='{spoken_text}' -> cleaned='{cleaned}' -> keywords={keywords} -> breed_key={breed_key}")

    if not breed_key:
        return render_template("breed_search.html", query=spoken_text, error="Breed not found in database", lang=session.get("lang", "en"))

    lang = session.get("lang", "en")
    entry = breed_data.get(breed_key, {})
    summary, animal_type = entry.get("summary"), entry.get("type")
    summary = translate_text(summary, lang)  # ✅ translate summary

    sample_files = ensure_static_breed_images(breed_key, n=3)
    images_rel = [f"breeds/{breed_key}/{fn}" for fn in sample_files]

    log.info(f"For breed_key={breed_key}, found sample images: {images_rel}")

    return render_template("breed_search.html",
                           query=spoken_text,
                           breed=breed_key,
                           animal_type=animal_type,
                           summary=summary,
                           images=images_rel,
                           lang=lang)

def predict_breed(filepath):
    """Run model on single image and return breed, top_results, and breed_info."""
    lang = session.get("lang", "en")

    # Encode image to base64 (optional for preview)
    data_uri = None
    try:
        with open(filepath, "rb") as fh:
            blob = fh.read()
        with Image.open(filepath) as im:
            fmt = (im.format or "JPEG").lower()
        mime = f"image/{fmt if fmt else 'jpeg'}"
        b64 = base64.b64encode(blob).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
    except Exception as e:
        log.error(f"Failed to encode image {filepath}: {e}")

    # Open + transform
    try:
        pil_img = Image.open(filepath).convert("RGB")
    except Exception as e:
        log.error(f"Failed to open {filepath} for prediction: {e}")
        return "Unknown", [], {"summary": "Invalid image", "type": "Unknown"}

    img_t = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)[0]
        topk = torch.topk(probs, k=min(3, probs.shape[0]))
        top_indices = topk.indices.cpu().numpy()
        top_values = topk.values.cpu().numpy()

    top_results = []
    for i, p in zip(top_indices, top_values):
        top_results.append({"breed": classes[int(i)], "prob": float(p)})

    breed = top_results[0]["breed"] if top_results else "Unknown"

    # Get breed info
    summary, animal_type, source = get_breed_info(breed)
    summary = translate_text(summary, lang)  # ✅ translate summary

    breed_info = {"summary": summary, "type": animal_type, "source": source}

    return breed, top_results, breed_info

# ----------------
@app.route("/camera_capture", methods=["POST"])
def camera_capture():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # Save the captured image
    filename = f"camera_{int(time.time())}_{file.filename}"
    filepath = os.path.join("static", filename)
    file.save(filepath)

    # Run prediction
    breed, top_results, breed_info = predict_breed(filepath)

    # Base64 encode for preview
    data_uri = None
    try:
        with open(filepath, "rb") as fh:
            blob = fh.read()
        with Image.open(filepath) as im:
            fmt = (im.format or "JPEG").lower()
        mime = f"image/{fmt if fmt else 'jpeg'}"
        b64 = base64.b64encode(blob).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
    except Exception as e:
        log.error(f"Failed to encode image {filepath}: {e}")

    # ✅ Validation only for camera
    valid_types = ["Cow", "Buffalo"]
    if (
        not top_results
        or top_results[0]["prob"] < 0.70
        or (breed_info.get("type") not in valid_types)
    ):
        return render_template(
            "result.html",
            results=[],
            lang=session.get("lang", "en"),
            error="❌ Invalid image: Not a cow or buffalo. Please try again."
        )

    # Prepare result for result.html
    result = {
        "image_filename": filename,
        "image_data": data_uri,   # ✅ Show preview
        "predicted": breed,
        "top_results": top_results,
        "breed_info": breed_info,
    }

    return render_template(
        "result.html",
        results=[result],
        lang=session.get("lang", "en"),
        ts=int(time.time())
    )


if __name__ == "__main__":
    app.run(debug=True)
