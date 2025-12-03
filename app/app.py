"""
app/app.py
-----------
Flask application for PneumoDetect
Includes correct model loading for Render + Docker.
"""

from pathlib import Path
import sys
import time
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from pydicom import dcmread
from torchvision import models, transforms
from werkzeug.utils import secure_filename

# -------------------------------------------------------------------
# Static directories (ensure they exist in Docker/Render)
# -------------------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "output").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "gradcam").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "uploads").mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gradcam import generate_cam, GradCAM  # noqa

# -------------------------------------------------------------------
# Flask Setup
# -------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder=str(STATIC_DIR)
)
BASE_DIR = Path(__file__).resolve().parent

# MODEL_PATH should resolve to /app/saved_models/resnet50_best.pt in Docker
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", "saved_models/resnet50_best.pt")
).resolve()

UPLOAD_FOLDER = STATIC_DIR / "uploads"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------
# Correct Model Loading (Fix: do NOT use weights_only=True)
# -------------------------------------------------------------------
def _load_model() -> torch.nn.Module:
    """Load trained model; fall back safely if missing."""
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    print(f"Resolved MODEL_PATH: {MODEL_PATH}")
    print(f"MODEL_PATH exists? {MODEL_PATH.exists()}")

    if MODEL_PATH.exists():
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Model loaded successfully: {MODEL_PATH.name} on {device}")
        except Exception as e:
            print(f"WARNING: Load failed ({e}); using random weights.")
    else:
        print("WARNING: MODEL_PATH does not exist! Using random weights.")

    return model.eval().to(device)


model = _load_model()


# -------------------------------------------------------------------
# Utility: Load image (supports .png/.jpg/.dcm)
# -------------------------------------------------------------------
def load_image(file_path: Path) -> Image.Image:
    ext = file_path.suffix.lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        return Image.open(file_path).convert("RGB")

    if ext == ".dcm":
        ds = dcmread(str(file_path))
        pixel_array = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        pixel_array = pixel_array * slope + intercept

        pixel_array -= pixel_array.min()
        max_val = pixel_array.max()
        if max_val > 0:
            pixel_array = pixel_array / max_val
        pixel_array = np.clip(pixel_array * 255.0, 0, 255).astype(np.uint8)

        if pixel_array.ndim == 2:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(pixel_array).convert("RGB")

    raise ValueError("Unsupported format.")


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        print(f"Rejected file type: {ext}")
        return redirect(url_for("index"))

    file_path = UPLOAD_FOLDER / filename
    file.save(file_path)

    try:
        return _perform_prediction(file_path, filename)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return redirect(url_for("index"))


def _perform_prediction(file_path: Path, filename: str):
    """
    Run preprocessing, inference, and Grad-CAM overlay generation.
    FIXED: Correct class ordering (PNEUMONIA = index 0, NORMAL = index 1)
    """
    try:
        threshold = float(request.form.get("threshold", 0.5))
        show_cam = "show_cam" in request.form

        # ------------------------------------------------------------
        # Load + preprocess the image
        # ------------------------------------------------------------
        ext = file_path.suffix.lower()
        if ext == ".dcm":
            ds = dcmread(str(file_path))
            img = ds.pixel_array
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_GRAY2RGB)
            img_pil = Image.fromarray(img)

            display_path = UPLOAD_FOLDER / f"{Path(filename).stem}.png"
            cv2.imwrite(str(display_path),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            img_pil = Image.open(file_path).convert("RGB")
            display_path = file_path

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        tensor = transform(img_pil).unsqueeze(0).to(device)

        # ------------------------------------------------------------
        # Inference
        # ------------------------------------------------------------
        start = time.time()
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
        elapsed = time.time() - start

        # ------------------------------------------------------------
        # FIXED CLASS ORDER (very important)
        # Your model outputs logits in this order:
        #   index 0 → Pneumonia
        #   index 1 → Normal
        # ------------------------------------------------------------
        pneumonia_prob = probs[0, 0].item()
        normal_prob = probs[0, 1].item()

        # Decision logic
        decision = "High Risk" if pneumonia_prob > threshold else "Low Risk"
        prediction_label = f"{decision} ({pneumonia_prob:.2f})"

        print(
            f"[PREDICTION] Pneumonia={pneumonia_prob:.3f}, "
            f"Normal={normal_prob:.3f}, Decision={prediction_label}"
        )

        # ------------------------------------------------------------
        # Grad-CAM overlay
        # ------------------------------------------------------------
        overlay_file = None
        if show_cam:
            cam = GradCAM(model, target_layer_name="layer4")
            heatmap = cam.generate(tensor)

            img_cv = cv2.imread(str(display_path))
            overlay = GradCAM.overlay_heatmap(img_cv, heatmap)

            overlay_name = f"{Path(filename).stem}_gradcam.png"
            overlay_path = STATIC_DIR / "output" / overlay_name
            cv2.imwrite(str(overlay_path), overlay)

            overlay_file = f"output/{overlay_name}"

        # ------------------------------------------------------------
        # Render HTML Results
        # ------------------------------------------------------------
        return render_template(
            "result.html",
            prediction=prediction_label,
            prob_pneumonia=pneumonia_prob,
            prob_normal=normal_prob,
            prob_raw_pneumonia=pneumonia_prob,
            prob_raw_normal=normal_prob,
            threshold=threshold,
            elapsed=f"{elapsed:.2f}s",
            image_file=f"uploads/{display_path.name}",
            overlay_file=overlay_file,
            show_cam=show_cam
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        return redirect(url_for("index"))


@app.route("/health")
def health():
    return {"status": "OK"}, 200


# -------------------------------------------------------------------
# Run Flask (local)
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
