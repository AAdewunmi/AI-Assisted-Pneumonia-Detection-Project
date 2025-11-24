"""
app/app.py
-----------
Flask scaffolding for PneumoDetect (Week 3 - Day 1).
Implements home and prediction routes, loads model weights, and serves HTML templates.
"""

from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import models, transforms
from PIL import Image
import io

# ------------------------------------------------------------
# 1. App Initialization
# ------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ------------------------------------------------------------
# 2. Model Loading (cached globally)
# ------------------------------------------------------------
MODEL_PATH = "saved_models/resnet50_finetuned.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# ------------------------------------------------------------
# 3. Image Preprocessing
# ------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------------------------------------------
# 4. Routes
# ------------------------------------------------------------
@app.route("/")
def home():
    """Render the home upload form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle file upload and run inference."""
    if "image" not in request.files:
        return redirect(url_for("home"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("home"))

    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, preds = torch.max(outputs, 1)

    label = "Pneumonia" if preds.item() == 1 else "Normal"
    return render_template("result.html", prediction=label)

# ------------------------------------------------------------
# 5. Main Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Running Flask server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
