
# 🩺 PneumoDetect: Clinical Decision Support System for Pneumonia Detection

---

<!-- CI / Lint / Coverage Badges -->
[![CI Pipeline](https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project/actions/workflows/ci.yml)
[![Code Style: Flake8](https://img.shields.io/badge/code%20style-flake8-3572A5.svg)](https://flake8.pycqa.org/)
[![Test Coverage](https://img.shields.io/codecov/c/github/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project)](https://codecov.io/gh/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project)
[![Coverage Status](https://codecov.io/gh/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project/branch/main/graph/badge.svg)](https://codecov.io/gh/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project)
![Docker Build](https://img.shields.io/badge/Docker-Build-blue)
![Image Size](https://img.shields.io/docker/image-size/_/python?label=Base%20Image)
![Dockerized](https://img.shields.io/badge/Containerized-PneumoDetect-brightgreen)

---

````markdown
# PneumoDetect: Clinical Decision Support System for Pneumonia Detection

A research-grade, end-to-end deep learning project that detects pneumonia in chest X-rays, surfaces Grad-CAM explainability maps, and exposes a clinician-style triage dashboard through a Flask web app.

PneumoDetect has been developed as a structured 4-week development lab, moving from raw imaging data through model training and interpretability to Dockerised cloud deployment and bias analysis. 

---

## Live Demo & Repository

- **Live Deployment (Render):**  
  `https://pneumonia-detection-ai-51h5.onrender.com/`

- **GitHub Repository:**  
  `https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project`

- **Maintainer:**  
  **Adrian Adewunmi**

---

## Project Overview

PneumoDetect is a compact clinical AI prototype that:

- Uses a **ResNet-50** convolutional neural network to classify chest X-rays as *pneumonia* vs *normal*.
- Provides **Grad-CAM** heatmaps for clinician-style visual explanation.
- Runs as a **Flask** web application with upload, prediction, and explanation views.
- Ships with **Docker**, **GitHub Actions CI**, and **Render**-based deployment workflows.
- Includes **bias and error analysis** through image-derived slices and Grad-CAM inspection.

The project aims to demonstrate not only model performance, but also:

- Reproducibility and environment management.
- Testing and continuous integration.
- Fairness, robustness, and transparent limitations in a healthcare AI setting. 

---

## Tech Stack

**Core:**

- Python 3.11
- PyTorch & Torchvision (ResNet-50)
- NumPy, Pandas, Scikit-learn
- OpenCV, Pillow

**Web & Frontend:**

- Flask
- Jinja2 templates
- Bootstrap 5
- Chart.js

**DevOps & Tooling:**

- Docker
- GitHub Actions CI
- Render (Web Service deployment)
- pytest, pytest-cov

---

## Repository Structure

Approximate layout (key files):

```text
AI-Assisted-Pneumonia-Detection-Project/
├── app/
│   ├── app.py                  # Flask entrypoint (model loading, routes, Grad-CAM wiring)
│   ├── templates/
│   │   ├── index.html          # Upload form, threshold slider, options
│   │   └── result.html         # Results dashboard with Grad-CAM, charts, metrics
│   └── static/
│       ├── uploads/            # Uploaded images
│       ├── output/             # Grad-CAM overlays for web app
│       ├── gradcam/            # Analysis Grad-CAM images
│       ├── css/
│       └── js/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset, transforms, loaders (Week 1) :contentReference[oaicite:2]{index=2}
│   ├── model.py                # ResNet-50 definition and fine-tuning (Week 1–2) 
│   ├── train.py                # Training loop for baseline and fine-tuned models 
│   ├── losses.py               # Optional class imbalance loss (e.g., Focal Loss) :contentReference[oaicite:5]{index=5}
│   ├── gradcam.py              # Core Grad-CAM implementation + generate_cam() helper 
│   └── analysis_cam.py         # Grad-CAM helpers for bias/error analysis notebook (W4) :contentReference[oaicite:7]{index=7}
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb      # Dataset EDA & preprocessing (W1) :contentReference[oaicite:8]{index=8}
│   ├── 02_train_resnet50.ipynb         # Baseline + fine-tuned training analysis (W1–2) 
│   ├── 03_gradcam_explainability.ipynb # Grad-CAM exploration (W2–3) 
│   └── 04_bias_analysis.ipynb          # Bias and slice analysis (W4) :contentReference[oaicite:11]{index=11}
│
├── saved_models/
│   ├── resnet50_baseline.pt            # Initial transfer-learning model (W1) :contentReference[oaicite:12]{index=12}
│   └── resnet50_finetuned.pt           # Improved model with unfreezing (W2) :contentReference[oaicite:13]{index=13}
│
├── docs/
│   ├── week1_summary.md                # Data, preprocessing, baseline metrics :contentReference[oaicite:14]{index=14}
│   ├── performance_report_v1.md        # Evaluation report with Grad-CAM (W2) :contentReference[oaicite:15]{index=15}
│   ├── bias_analysis.md                # Bias and error analysis report (W4) :contentReference[oaicite:16]{index=16}
│   └── architecture.md                 # Optional: system and model architecture overview
│
├── reports/
│   ├── week1_metrics/                  # ROC, PR curves, confusion matrix plots :contentReference[oaicite:17]{index=17}
│   └── week2_gradcam_samples/          # Saved Grad-CAM overlays for examples :contentReference[oaicite:18]{index=18}
│
├── tests/
│   ├── test_preprocessing.py           # Data loader / preprocessing tests (W1) :contentReference[oaicite:19]{index=19}
│   ├── test_gradcam.py                 # Grad-CAM utility tests (W2) :contentReference[oaicite:20]{index=20}
│   ├── test_threshold_logic.py         # Threshold logic tests for Flask app (W3) :contentReference[oaicite:21]{index=21}
│   └── test_analysis_cam.py            # Tests for analysis_cam Grad-CAM overlays (W4) :contentReference[oaicite:22]{index=22}
│
├── Dockerfile                          # Container definition for Flask + PyTorch app :contentReference[oaicite:23]{index=23}
├── .dockerignore                       # Ignore data, reports, tests in image builds :contentReference[oaicite:24]{index=24}
├── .github/
│   └── workflows/ci.yml                # GitHub Actions CI for tests and coverage :contentReference[oaicite:25]{index=25}
│
├── requirements.txt
└── README.md                           # You are here
````

---

## 4-Week Lab Roadmap

This project doubles as a 4-week, postgraduate-style mini-course in medical imaging deep learning. Each week focuses on a different phase of the ML lifecycle.

### Week 1 — From Raw X-rays to a Trainable Dataset 

* Environment setup, repo hygiene, and requirements management.
* EDA on a chest X-ray dataset subset (label distribution, image statistics).
* Preprocessing and augmentation implemented in `src/data_loader.py`.
* Baseline ResNet-50 transfer learning model trained and evaluated.
* Week summary in `docs/week1_summary.md` with ROC, PR, confusion matrix.

### Week 2 — Model Refinement & Explainability (Grad-CAM) 

* Handling class imbalance with weighted sampling or specialised loss.
* Fine-tuning deeper ResNet blocks and experimenting with learning rates.
* Grad-CAM implemented in `src/gradcam.py` and explored in notebook 03.
* First performance report in `docs/performance_report_v1.md`.

### Week 3 — Flask Dashboard & Triage UI 

* Flask app scaffolding with `/` and `/predict` routes in `app/app.py`.
* File upload, preprocessing, inference, and thresholding integrated.
* Grad-CAM overlays served in a clinician-style triage dashboard.
* Bootstrap layout and Chart.js probability bar charts for class scores.
* Threshold logic and route tests added under `tests/`.

### Week 4 — Docker, CI, Cloud Deployment & Bias Analysis 

* GitHub Actions CI pipeline with pytest and coverage.
* Dockerfile and `.dockerignore` for reproducible builds.
* Render deployment with `/health` endpoint and environment variables.
* Bias and error analysis in `04_bias_analysis.ipynb` and `docs/bias_analysis.md`.
* Final wrap-up with deployment link, architecture notes, and reflection.

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project.git
cd AI-Assisted-Pneumonia-Detection-Project
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # on macOS/Linux
# .venv\Scripts\activate    # on Windows PowerShell
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. UI Screenshot

<img width="626" height="1029" alt="Image" src="https://github.com/user-attachments/assets/0109b675-b56f-456c-88a5-eeb5d8f23e4f" />

---

## Running the Web App Locally

From the project root:

```bash
export FLASK_APP=app/app.py
export FLASK_ENV=development
# Optional: configure model path and threshold
export MODEL_PATH="saved_models/resnet50_finetuned.pt"
export THRESHOLD=0.8

python app/app.py
```

The app will bind to the configured host and port (commonly `http://127.0.0.1:5001` for local development). The UI supports:

* Uploading `.jpg`, `.png`, or `.dcm` chest X-ray images.
* Selecting a risk threshold via slider.
* Viewing prediction label, probabilities, and inference time.
* Toggling Grad-CAM explainability overlays.

---

## Running with Docker

Build the image:

```bash
docker build -t pneumodetect .
```

Run the container:

```bash
docker run -p 5000:5000 \
  -e MODEL_PATH="saved_models/resnet50_finetuned.pt" \
  -e THRESHOLD=0.8 \
  pneumodetect
```

The app will be available at `http://127.0.0.1:5000`.

The Dockerfile:

* Uses `python:3.11-slim`.
* Installs required system libraries for OpenCV.
* Copies the repository into `/app`.
* Installs Python dependencies.
* Exposes port 5000 and runs `python app/app.py`. 

---

## Testing & Quality

Unit and integration tests live under `tests/`. The CI pipeline runs:

* Linting (optional, depending on your configuration).
* pytest with coverage thresholds.
* Docker build validation in some configurations.

Typical local test run:

```bash
pytest -q --disable-warnings --maxfail=1 \
  --cov=src --cov-report=term-missing
```

Examples:

* `tests/test_preprocessing.py` confirms transforms produce valid tensors. 
* `tests/test_gradcam.py` validates Grad-CAM output ranges and shapes. 
* `tests/test_threshold_logic.py` ensures the risk label matches probability and threshold rules. 
* `tests/test_analysis_cam.py` checks Grad-CAM overlay generation for analysis workflows. 

GitHub Actions CI workflow (`.github/workflows/ci.yml`) installs dependencies, runs tests, and can enforce coverage thresholds before merges. 

---

## Model & Training

### Data & Preprocessing

Week 1 focuses on preparing a research-grade pipeline:

* Downloading a manageable subset of a chest X-ray dataset.
* Exploring label distribution and image statistics in `01_eda_preprocessing.ipynb`.
* Implementing resizing (224×224), normalization with ImageNet mean/std, and augmentation in `src/data_loader.py`. 

The preprocessing step feeds into both model training and the web app inference path, supporting consistency between offline experiments and online predictions.

### Baseline & Fine-Tuned Models

The project uses a **ResNet-50** backbone:

* **Baseline model:**
  ResNet-50 with frozen base layers and a new 2-class head, trained for a small number of epochs to establish an initial ROC-AUC baseline. 
* **Balanced training and fine-tuning:**
  Later weeks introduce class balancing via weighted sampling or specialised loss, unfreezing deeper blocks, and differential learning rates, all tracked in `02_train_resnet50.ipynb`. 

Saved checkpoints:

* `saved_models/resnet50_baseline.pt`
* `saved_models/resnet50_finetuned.pt`

Performance metrics, including ROC, precision–recall, confusion matrices, and sensitivity/specificity, appear in `docs/week1_summary.md` and `docs/performance_report_v1.md`.

---

## Explainability with Grad-CAM

Grad-CAM is a central feature for clinician trust and model introspection.

Key components:

* `src/gradcam.py` registers hooks on the last convolutional layer (`layer4`) and produces Grad-CAM heatmaps with values normalised to `[0, 1]`.
* `generate_cam(image_path, model_path)` wraps end-to-end Grad-CAM generation for a given model checkpoint and image.
* `03_gradcam_explainability.ipynb` demonstrates Grad-CAM on curated examples, saving overlays under `reports/week2_gradcam_samples/`.

In the Flask app:

* The `/predict` route triggers Grad-CAM heatmap generation when the user selects the explainability option.
* The `result.html` template presents original and Grad-CAM overlay images side by side, along with class probabilities and threshold information. 

---

## Bias & Error Analysis

The Week 4 bias and error analysis recognises that demographic metadata is not always available in deployed systems. The notebook `04_bias_analysis.ipynb` therefore focuses on **image-derived slices** and **model-confidence analysis**. 

### Slices Analysed

* **Brightness bins:** low, mid, high brightness groups using quantile-based thresholds.
* **Resolution bins:** low, mid, high based on total pixel count.
* **Confidence bins:** partitions of the pneumonia probability, such as `[0–0.65]`, `[0.65–0.8]`, `[0.8–1.0]`.

For each slice:

* AUC is computed with respect to available or synthetic labels.
* Misclassified cases are identified, and Grad-CAM overlays are generated for closer inspection via `src/analysis_cam.py`. 

### Generated Outputs

* A summary table of AUC values per slice type in `docs/bias_analysis.md`.
* Grad-CAM overlays for misclassified or uncertain predictions under `static/gradcam/`.
* A short narrative about potential sources of bias and mitigation ideas, including:

  * Dataset composition and class imbalance.
  * Label noise and inconsistent imaging conditions.
  * Possible interventions such as reweighting, augmentation, and broader data collection. 

The notebook includes an ethical statement and reinforces the advisory role of the model.

---

## Deployment & Operations

### Render Deployment

The project targets Render as a simple hosting environment: 

* The Docker image builds from the same Dockerfile used locally.

* Environment variables configure:

  * `MODEL_PATH` for the loaded checkpoint.
  * `THRESHOLD` for decision thresholding.
  * `PORT` is managed by Render; the Flask app reads it from the environment.

* A `/health` endpoint returns JSON status for uptime checks.

* The live app surfaces Grad-CAM overlays and probability distributions for uploaded images.

### GitHub Actions CI

The CI pipeline (`.github/workflows/ci.yml`) typically executes:

* Python setup for specified versions.
* Dependency installation, including CPU builds of PyTorch.
* pytest with coverage enforcement.
* Docker build sanity checks in some configurations. 

A CI status badge can be added to the top of this README using the workflow badge URL from GitHub.

---

## Ethical Use & Limitations

This project is a **research and educational prototype**. It is not a medical device.

> **This model assists clinicians but is not a diagnostic device.** 

Key limitations:

* Trained on a subset of publicly available chest X-ray data; external validity may be limited.
* Potential for bias due to dataset composition, scanner variability, and class imbalance.
* No guarantee that Grad-CAM heatmaps always align with clinically meaningful regions.
* Performance metrics in notebooks are illustrative and may not generalise to real-world deployment conditions.

Any real clinical use would require rigorous validation, regulatory approval, and integration into clinical workflows.

---

## Roadmap & Ideas

Potential future directions:

* Integration with additional datasets that include demographic metadata, enabling age- and sex-stratified fairness analysis.
* Experimentation with lighter backbones (e.g., MobileNetV2) for lower-resource cloud environments.
* Calibration methods (temperature scaling, Platt scaling) and improved uncertainty quantification.
* Multi-label extensions for additional thoracic pathologies.
* Monitoring and logging for production MLOps scenarios.

---

## Acknowledgements

* Public chest X-ray datasets and associated research communities for making imaging data available for educational work.
* The PyTorch and Flask ecosystems for enabling rapid experimentation.
* Open-source contributors in the Python and MLOps community whose tools and patterns inform this project.

---





