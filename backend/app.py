"""
Road Surface Condition Detection — FastAPI backend.

Run from the project root:
    uvicorn backend.app:app --reload
then open http://127.0.0.1:8000
"""
import base64
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"
PREDICT_MODEL_PATH = MODELS_DIR / "road_surface_mobilenetv2.keras"
CLASSES_PATH = MODELS_DIR / "classes.json"
DEFAULT_CLASSES = ["normal", "potholes"]
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
OVERLAY_MAX_SIDE = 640

log = logging.getLogger("road-surface")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Filled in at startup
state = {
    "predict_model": None,
    "gradcam_fn": None,
    "class_names": DEFAULT_CLASSES,
    "img_size": (224, 224),
    "load_error": None,
}


def load_class_names():
    try:
        names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        if isinstance(names, list) and names:
            return [str(n) for n in names]
    except FileNotFoundError:
        log.warning("models/classes.json not found — using default classes %s", DEFAULT_CLASSES)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Could not read classes.json (%s) — using default classes", e)
    return DEFAULT_CLASSES


def last_feature_map_layer(model):
    """The layer Grad-CAM should explain: 'out_relu' (MobileNetV2) or 'last_conv' (custom CNN),
    otherwise the last layer with a 4-D (batch, h, w, channels) output."""
    for name in ("out_relu", "last_conv"):
        try:
            return model.get_layer(name)
        except ValueError:
            pass
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            return layer
    raise ValueError("no convolutional feature-map layer found")


def build_gradcam_fn(keras, model):
    """Return fn(batch) -> (feature maps, predictions) for Grad-CAM on the prediction model itself.

    MobileNetV2 is nested inside our model as one layer, so its inner conv output isn't reachable
    from the outer graph. We split the model: layers before the base, the base (returning its last
    feature maps as well), then the classifier head.
    """
    nested = [l for l in model.layers if isinstance(l, keras.Model)]
    if not nested:
        conv = last_feature_map_layer(model)
        grad_model = keras.Model(model.inputs, [conv.output, model.output])
        return (lambda x: grad_model(x, training=False)), conv.name  # e.g. a flat custom CNN

    base = nested[-1]
    conv = last_feature_map_layer(base)
    inner = keras.Model(base.inputs, [conv.output, base.output])
    pos = model.layers.index(base)
    before = [l for l in model.layers[:pos] if not isinstance(l, keras.layers.InputLayer)]
    after = model.layers[pos + 1:]

    def run(x):
        for layer in before:
            x = layer(x, training=False)
        conv_out, x = inner(x, training=False)
        for layer in after:
            x = layer(x, training=False)
        return conv_out, x

    return run, f"{base.name}/{conv.name}"


def check_gradcam_fn(fn, model, img_size):
    """The split model must reproduce the real model's output, otherwise the heatmap would be wrong."""
    x = np.random.default_rng(0).uniform(0, 255, (1, *img_size, 3)).astype("float32")
    _, preds = fn(x)
    if not np.allclose(np.asarray(preds), np.asarray(model(x, training=False)), atol=1e-4):
        raise ValueError("Grad-CAM model output doesn't match the prediction model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["class_names"] = load_class_names()

    if not PREDICT_MODEL_PATH.exists():
        state["load_error"] = (
            f"Model file not found: models/{PREDICT_MODEL_PATH.name}. "
            "Train the model in the Colab notebook, download road_surface_mobilenetv2.keras "
            "and classes.json, put them in the models/ folder, "
            "then restart the server."
        )
        log.error(state["load_error"])
        yield
        return

    import tensorflow as tf  # imported lazily so the server starts fast and errors are clear
    from tensorflow import keras

    log.info("TensorFlow %s — loading %s", tf.__version__, PREDICT_MODEL_PATH.name)
    try:
        model = keras.models.load_model(PREDICT_MODEL_PATH, compile=False)
    except Exception as e:
        state["load_error"] = (
            f"Could not load models/{PREDICT_MODEL_PATH.name}: {e}. "
            f"This is usually a TensorFlow/Keras version mismatch — local TensorFlow is {tf.__version__}; "
            "install the version printed at the end of the Colab notebook."
        )
        log.error(state["load_error"])
        yield
        return
    state["predict_model"] = model
    h, w = model.input_shape[1:3]
    state["img_size"] = (int(h or 224), int(w or 224))

    # Grad-CAM explains the same model that makes the prediction, so the heatmap always matches the label
    try:
        fn, layer_name = build_gradcam_fn(keras, model)
        check_gradcam_fn(fn, model, state["img_size"])
        state["gradcam_fn"] = fn
        log.info("Grad-CAM uses layer '%s'", layer_name)
    except Exception as e:
        log.warning("Grad-CAM disabled — %s", e)

    log.info("Ready. Classes: %s | Grad-CAM: %s", state["class_names"], state["gradcam_fn"] is not None)
    yield


app = FastAPI(title="Road Surface Condition Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- helpers ----------

def is_pothole(label: str) -> bool:
    return "pothole" in label.lower()


def advice_for(label: str, confidence: float) -> str:
    if is_pothole(label):
        if confidence >= 0.8:
            return ("Pothole detected. Drive slowly and steer around it if safe, "
                    "and report this location to the local road maintenance authority.")
        return ("Possible pothole. The model isn't fully sure — check the road in person "
                "or try a clearer, closer photo taken in daylight.")
    if confidence >= 0.8:
        return "Road surface looks normal. No visible potholes — keep driving safely."
    return ("Road looks mostly normal, but confidence is low. "
            "Try a clearer photo that shows more of the road surface.")


def jet_colormap(x: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB in [0, 1] using the classic 'jet' palette (no matplotlib needed)."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def gradcam_overlay_b64(batch: np.ndarray, original: Image.Image, class_index: int) -> str:
    import tensorflow as tf

    with tf.GradientTape() as tape:
        conv_out, preds = state["gradcam_fn"](batch)
        class_score = preds[:, class_index]
    grads = tape.gradient(class_score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Overlay on a (downsized) copy of the original photo so the result looks natural
    base = original.copy()
    base.thumbnail((OVERLAY_MAX_SIDE, OVERLAY_MAX_SIDE))
    heat_img = Image.fromarray((heatmap * 255).astype("uint8")).resize(base.size, Image.BILINEAR)
    colored = jet_colormap(np.asarray(heat_img, dtype="float32") / 255.0) * 255
    blended = np.asarray(base, dtype="float32") * 0.6 + colored * 0.4
    out = Image.fromarray(np.clip(blended, 0, 255).astype("uint8"))

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------- routes ----------

@app.get("/", include_in_schema=False)
def index():
    if not FRONTEND_FILE.exists():
        raise HTTPException(404, "frontend/index.html not found")
    return FileResponse(FRONTEND_FILE)


@app.get("/api/health")
def health():
    ok = state["predict_model"] is not None
    return {
        "status": "ok" if ok else "model_missing",
        "model_loaded": ok,
        "classes": state["class_names"],
        "gradcam_available": state["gradcam_fn"] is not None,
        "message": None if ok else state["load_error"],
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    if state["predict_model"] is None:
        raise HTTPException(503, state["load_error"] or "Model not loaded. Put the .keras files in models/.")

    if file.content_type and not (file.content_type.startswith("image/")
                                  or file.content_type == "application/octet-stream"):
        raise HTTPException(400, f"'{file.filename}' is not an image (got {file.content_type}). "
                                 "Please upload a JPG or PNG road photo.")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"Image is too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")  # respect phone camera rotation
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, f"'{file.filename}' could not be read as an image. "
                                 "Please upload a JPG or PNG road photo.")

    h, w = state["img_size"]
    batch = np.asarray(img.resize((w, h), Image.BILINEAR), dtype="float32")[None]  # raw 0-255, model rescales

    probs = np.asarray(state["predict_model"](batch, training=False))[0]
    names = state["class_names"]
    if len(names) != len(probs):
        names = [f"class_{i}" for i in range(len(probs))]
    idx = int(np.argmax(probs))
    label, confidence = names[idx], float(probs[idx])

    gradcam = None
    if state["gradcam_fn"] is not None:
        try:
            gradcam = gradcam_overlay_b64(batch, img, idx)
        except Exception as e:
            log.warning("Grad-CAM failed: %s", e)

    return {
        "label": label,
        "is_pothole": is_pothole(label),
        "confidence": confidence,
        "probabilities": {n: float(p) for n, p in zip(names, probs)},
        "advice": advice_for(label, confidence),
        "gradcam": gradcam,
        "gradcam_available": state["gradcam_fn"] is not None,
    }
