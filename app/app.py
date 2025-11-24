"""
app/app.py
----------
Flask entrypoint for PneumoDetect (Week 3 - Day 1)
Adds DICOM + PNG upload support, routes for home and predict,
and renders results safely with cached ResNet-50 model.
"""

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import torch
from torchvision import models, transforms
from PIL import Image
from pydicom import dcmread
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
# Image loader that handles DICOM and PNG/JPG
# -------------------------------------------------------------------
def load_image(file_path: Path) -> Image.Image:
    ext = file_path.suffix.lower()
    if ext == ".dcm":
        ds = dcmread(str(file_path))
        img = ds.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img)
    else:
        return Image.open(file_path).convert("RGB")

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Home page with upload form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle file upload and run inference."""
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    file_path = UPLOAD_FOLDER / filename
    file.save(file_path)

    # Load image
    img = load_image(file_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, preds = torch.max(outputs, 1)
        label = "Pneumonia Detected" if preds.item() == 1 else "Normal"

    return render_template("result.html", prediction=label)

# -------------------------------------------------------------------
# Run Flask App
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Running Flask server on http://127.0.0.1:5000")
    app.run(debug=True)
