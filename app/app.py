"""
app/app.py
-----------
Flask application for PneumoDetect (Week 3, Day 2)
Extends Day 1 by adding probability-based predictions, risk thresholding,
and inference timing. Supports .jpg/.png/.dcm uploads.
"""

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pydicom import dcmread
import time
import numpy as np
import cv2

# Project paths and import setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import GradCAM utilities (now that the project root is on sys.path)
from src.gradcam import generate_cam, GradCAM

# -------------------------------------------------------------------
# Flask Setup
# -------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
MODEL_PATH = PROJECT_ROOT / "saved_models" / "resnet50_best.pt"
UPLOAD_FOLDER = Path(app.static_folder) / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# Load model globally
# -------------------------------------------------------------------
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval().to(device)
print(f"Model loaded: {MODEL_PATH.name} on {device}")

# -------------------------------------------------------------------
# Image loader (handles .png/.jpg/.dcm)
# -------------------------------------------------------------------
def load_image(file_path: Path) -> Image.Image:
    ext = file_path.suffix.lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        return Image.open(file_path).convert("RGB")

    if ext == ".dcm":
        ds = dcmread(str(file_path))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply DICOM rescale parameters when present
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        pixel_array = pixel_array * slope + intercept

        # Normalize to 0–255 and convert to 3-channel RGB
        pixel_array -= pixel_array.min()
        max_val = pixel_array.max()
        if max_val > 0:
            pixel_array = pixel_array / max_val
        pixel_array = np.clip(pixel_array * 255.0, 0, 255).astype(np.uint8)
        if pixel_array.ndim == 2:  # grayscale
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
        elif pixel_array.shape[-1] == 1:  # single-channel with trailing dim
            pixel_array = cv2.cvtColor(pixel_array.squeeze(-1), cv2.COLOR_GRAY2RGB)

        return Image.fromarray(pixel_array).convert("RGB")

    raise ValueError("Unsupported image format. Please upload .jpg, .png, or .dcm.")

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Home page with upload form and threshold slider."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle image upload (.jpg/.png/.dcm), model inference, and Grad-CAM overlay generation.
    Combines Day 2 inference logic + Day 3 explainability + DICOM support.
    """
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    # Save uploaded file
    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        print(f"Rejected upload with unsupported extension: {ext}")
        return redirect(url_for("index"))

    file_path = UPLOAD_FOLDER / filename
    file.save(file_path)

    try:
        # Threshold and Grad-CAM toggle
        threshold = float(request.form.get("threshold", 0.5))
        show_cam = "show_cam" in request.form

        # ------------------------------------------------------------
        # Load image (supports DICOM and common image formats)
        # ------------------------------------------------------------
        ext = file_path.suffix.lower()
        display_path = file_path
        if ext == ".dcm":
            from pydicom import dcmread
            ds = dcmread(str(file_path))
            img = ds.pixel_array
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_GRAY2RGB)
            img_pil = Image.fromarray(img)

            # Save a PNG copy for browser display
            display_path = UPLOAD_FOLDER / f"{Path(filename).stem}.png"
            cv2.imwrite(str(display_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            img_pil = Image.open(file_path).convert("RGB")
            display_path = file_path

        # ------------------------------------------------------------
        # Preprocessing
        # ------------------------------------------------------------
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(img_pil).unsqueeze(0).to(device)

        # ------------------------------------------------------------
        # Inference
        # ------------------------------------------------------------
        start_time = time.time()
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
            pneumonia_prob = probs[0, 1].item()
        elapsed = time.time() - start_time

        # ------------------------------------------------------------
        # Decision
        # ------------------------------------------------------------
        decision = "High Risk" if pneumonia_prob > threshold else "Low Risk"
        label = f"{decision} ({pneumonia_prob:.2f} probability)"
        print(f"Prediction: {label} | Time: {elapsed:.2f}s")

        # ------------------------------------------------------------
        # Grad-CAM Overlay Generation
        # ------------------------------------------------------------
        overlay_path = None
        if show_cam:
            # Generate Grad-CAM heatmap
            heatmap = generate_cam(file_path, MODEL_PATH)

            # Use OpenCV array for overlay creation
            if ext == ".dcm":
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                img_cv = cv2.imread(str(file_path))

            overlay = GradCAM.overlay_heatmap(img_cv, heatmap)
            overlay_name = f"{Path(filename).stem}_gradcam.png"
            overlay_path = UPLOAD_FOLDER / overlay_name
            cv2.imwrite(str(overlay_path), overlay)
            print(f"Grad-CAM overlay saved: {overlay_path.name}")

        # ------------------------------------------------------------
        # Render HTML Result
        # ------------------------------------------------------------
        return render_template(
            "result.html",
            prediction=label,
            prob=f"{pneumonia_prob:.3f}",
            threshold=threshold,
            elapsed=f"{elapsed:.2f}s",
            image_file=f"uploads/{display_path.name}",
            overlay_file=f"uploads/{overlay_path.name}" if overlay_path else None,
            show_cam=show_cam
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        return redirect(url_for("index"))

# -------------------------------------------------------------------
# Run Flask App
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Running Flask server on http://127.0.0.1:5000")
    app.run(debug=True)
