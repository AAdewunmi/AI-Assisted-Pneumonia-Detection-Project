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
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pydicom import dcmread
import time
import numpy as np
import cv2

# -------------------------------------------------------------------
# Flask Setup
# -------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "saved_models" / "resnet50_best.pt"
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

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
    """Handle image upload, run inference, and return prediction."""
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    file_path = UPLOAD_FOLDER / filename
    file.save(file_path)

    try:
        # Get threshold (default 0.5)
        threshold = float(request.form.get("threshold", 0.5))

        # Preprocess image
        img = load_image(file_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0).to(device)

        # Run inference with timing
        start_time = time.time()
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
            pneumonia_prob = probs[0, 1].item()
        elapsed = time.time() - start_time

        # Decision logic
        decision = "High Risk" if pneumonia_prob > threshold else "Low Risk"
        label = f"{decision} ({pneumonia_prob:.2f} probability)"
        print(f"Prediction: {label} | Time: {elapsed:.2f}s")

        return render_template(
            "result.html",
            prediction=label,
            prob=f"{pneumonia_prob:.3f}",
            threshold=threshold,
            elapsed=f"{elapsed:.2f}s"
        )

    except Exception as e:
        print(f" Error during prediction: {e}")
        return redirect(url_for("index"))

# -------------------------------------------------------------------
# Run Flask App
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Running Flask server on http://127.0.0.1:5000")
    app.run(debug=True)
