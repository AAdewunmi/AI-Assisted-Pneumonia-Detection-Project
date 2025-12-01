
# 🩺 PneumoDetect: Deep Learning for Pneumonia Detection & Clinician Triage Dashboard

## CI Status Badge

---

[![PneumoDetect CI](https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project/actions/workflows/ci.yml)
[![Lint](https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project/actions/workflows/ci.yml/badge.svg)](...)
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen)](...)

---

**Duration:** 4 Weeks (20 Lab Days, Mon–Fri)
**Format:** Hands-on postgraduate programming lab
**Focus:** Deep Learning • Explainability • Flask Apps • MLOps • Ethics in AI

---

## 🎯 Course Overview

Radiology teams face overwhelming imaging workloads. This lab guides you through building **PneumoDetect** — a deep-learning model that detects pneumonia from chest X-rays and surfaces explainable Grad-CAM overlays inside a clinician-style triage dashboard.

You’ll move from raw NIH/RSNA image data to a deployed Flask app in four structured sprints, gaining practical skills in data handling, transfer learning, explainability, deployment, and bias evaluation.

---

## 📆 Weekly Structure

| Week                                  | Theme                                                       | Core Skills                                                                  | Key Deliverables                                                                        |
| ------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **1 — Foundations & Data Pipeline**   | Data exploration, preprocessing, transfer-learning baseline | EDA • PyTorch DataLoaders • Model init • Metrics (AUC, confusion matrix)     | `01_eda_preprocessing.ipynb`, `src/data_loader.py`, `src/model.py`, baseline AUC report |
| **2 — Model Tuning & Explainability** | Class imbalance, fine-tuning, Grad-CAM explainability       | Weighted loss • Fine-tuning • Grad-CAM visuals • Interpretability testing    | `src/gradcam.py`, `tests/test_gradcam.py`, Grad-CAM montage, `performance_report_v1.md` |
| **3 — Flask Dashboard MVP**           | Serving models & designing clinician-style dashboards       | Flask • Bootstrap • Chart.js • UI/UX • API integration • Testing             | `app/app.py`, `index.html`, `test_threshold_logic.py`, dashboard demo GIF               |
| **4 — Deployment & Bias Analysis**    | CI/CD, Dockerization, cloud hosting, fairness auditing      | GitHub Actions • Docker • Render/Railway deploy • Bias metrics • Model cards | `.github/workflows/ci.yml`, `Dockerfile`, `bias_analysis.md`, live app URL + video demo |

---

## 🧪 Learning Outcomes

By completing PneumoDetect, you will be able to:

1. **Engineer medical imaging pipelines** for deep learning using PyTorch.
2. **Train and evaluate** transfer-learning CNNs with balanced sampling.
3. **Implement explainability methods** (Grad-CAM) for clinician insight.
4. **Develop and deploy** Flask web apps with CI/CD and Docker.
5. **Assess model bias** and communicate ethical limitations clearly.

---

## 📚 Assessment & Artifacts

* ✅ Functional source code (tests passing)
* ✅ Model weights + Grad-CAM outputs
* ✅ Deployed triage dashboard (cloud link)
* ✅ Clinical-style performance & bias reports
* ✅ Reflective weekly write-ups (LinkedIn / Medium)

---

## ✍️ Reflective Practice

Each Friday concludes with a short reflection blog:

* *Week 1 – “Cleaning the Chaos: Making X-rays Trainable”*
* *Week 2 – “Teaching the Model to Explain Itself”*
* *Week 3 – “Where AI Meets the Clinician’s Screen”*
* *Week 4 – “Deployment, Bias, and the Human Context”*

These reflections document your growth and reinforce professional communication skills valued in industry and research.

---

## 🧩 Tools & Stack

**Languages:** Python (3.11)
**Libraries:** PyTorch, torchvision, numpy, pandas, matplotlib, Flask, Chart.js, Bootstrap
**Infrastructure:** Docker, GitHub Actions, Render/Railway
**Data:** NIH Chest X-ray 14 / RSNA Pneumonia Detection Subset

---

## 💬 Final Deliverable

A reproducible, deployed dashboard backed by a transparent, explainable CNN model — ready for portfolio, recruiter showcase, or academic presentation.

---



