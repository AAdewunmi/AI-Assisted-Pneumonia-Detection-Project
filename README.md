<!-- CI / Lint / Coverage Badges -->
[![CI Pipeline](https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project/actions/workflows/ci.yml)
[![Code Style: Flake8](https://img.shields.io/badge/code%20style-flake8-3572A5.svg)](https://flake8.pycqa.org/)
[![Test Coverage](https://img.shields.io/codecov/c/github/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project)](https://codecov.io/gh/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project)
[![Coverage Status](https://codecov.io/gh/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project/branch/main/graph/badge.svg)](https://codecov.io/gh/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project)
![Docker Build](https://img.shields.io/badge/Docker-Build-blue)
![Image Size](https://img.shields.io/docker/image-size/_/python?label=Base%20Image)
![Dockerized](https://img.shields.io/badge/Containerized-PneumoDetect-brightgreen)


# PneumoDetect: Clinical Decision Support System for Pneumonia Detection

Flask + PyTorch is a clinical AI prototype for pneumonia risk triage from chest X-rays (PNG/JPG/DICOM), with Grad-CAM explainability overlays and CI-backed quality checks.

## Status Snapshot

- Deployment target: Render web service
- Runtime: Python 3.11
- Model family: ResNet-50 binary classifier (`PNEUMONIA` vs `NORMAL`)
- Explainability: Grad-CAM overlays on uploaded images
- CI quality gates: Flake8 + pytest + coverage (>= 70% in CI)
- Test suite size: 35 test cases under `tests/`

## Live and Source

- Live app: `https://pneumonia-detection-ai-51h5.onrender.com/`
- Repository: `https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project`
- Maintainer: Adrian Adewunmi

## What This Project Does

- Accepts chest imaging files (`.png`, `.jpg`, `.jpeg`, `.dcm`) through a Flask UI.
- Preprocesses images and runs inference with a ResNet-50 model.
- Returns pneumonia probability, normal probability, and threshold-based triage label.
- Optionally generates and displays Grad-CAM heatmaps for model transparency.
- Exposes a health endpoint for deployment checks.

## Tech Stack

- Language/runtime: Python 3.11
- Deep learning: PyTorch, Torchvision (ResNet-50)
- Data/science: NumPy, Pandas, scikit-learn, tqdm
- Imaging: OpenCV, Pillow, pydicom
- Web app: Flask, Jinja2, Werkzeug, flask-cors, gunicorn
- Explainability: grad-cam
- Testing/quality: pytest, pytest-cov, Flake8
- Deployment/ops: Docker, GitHub Actions, Render

## Surface Map

### Web routes

- `GET /` -> upload and inference form
- `POST /predict` -> prediction pipeline + optional Grad-CAM
- `GET /health` -> `{"status": "OK"}`

### Key runtime settings

- `MODEL_PATH` (default: `saved_models/resnet50_best.pt`)
- `PORT` (default local: `5001`)
- `FLASK_DEBUG` (`true`/`false`)

## Architecture Overview

1. Input ingestion in [`app/app.py`](app/app.py).
2. Image normalization and transform to `224x224` tensor.
3. ResNet-50 forward pass (2-class logits -> softmax probabilities).
4. Threshold logic for risk labeling.
5. Optional Grad-CAM generation via `src/gradcam.py`.
6. Result rendering in `app/templates/result.html`.

## Local Development

### Prerequisites

- Python 3.11
- `pip`
- Optional: Docker Desktop

### Quick start

```bash
git clone https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project.git
cd AI-Assisted-Pneumonia-Detection-Project

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Run the app

```bash
export MODEL_PATH=saved_models/resnet50_best.pt
export PORT=5001
export FLASK_DEBUG=true

python app/app.py
```

Open `http://localhost:5001`.

## Training

Train baseline model:

```bash
python -m src.train --epochs 3 --batch_size 8 --lr 1e-3
```

Train with balanced loader:

```bash
python -m src.train --balanced
```

Resume from checkpoint:

```bash
python -m src.train --resume saved_models/resnet50_finetuned.pt
```

Default training dataset arguments:

- `--csv_path data/rsna_subset/stage_2_train_labels.csv`
- `--img_dir data/rsna_subset/train_images`

## Quality Baseline

### Run lint

```bash
flake8 src app tests --max-line-length=100
```

### Run tests with coverage

```bash
pytest -q --disable-warnings --maxfail=1 \
  --cov=src --cov=app --cov-report=term-missing --cov-fail-under=70
```

### Docker smoke flow

```bash
docker build -t pneumodetect:ci .
docker run --rm -p 5000:5000 pneumodetect:ci
```

## Repository Layout

```text
AI-Assisted-Pneumonia-Detection-Project/
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
├── src/
│   ├── data_loader.py
│   ├── gradcam.py
│   ├── model.py
│   └── train.py
├── tests/
├── notebooks/
├── reports/
├── saved_models/
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml
```

## Known Constraints

- Inference quality depends on checkpoint quality and data representativeness.
- This is a decision-support prototype, not a diagnostic medical device.
- Clinical deployment requires external validation, governance, and regulatory review.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
