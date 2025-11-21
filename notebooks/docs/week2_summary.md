# Week 2 Summary — PneumoDetect Lab Series

**Theme:** Model Refinement, Explainability, and Performance Review  
**Duration:** Week 2 (Days 1–5)  
**Project:** [AI-Assisted-Pneumonia-Detection-Project](https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project)

---

## Highlights

| Day | Focus | Core Deliverables |
|-----|--------|------------------|
| W2-D1 | Model Training (Baseline) | `resnet50_baseline.pt`, initial metrics |
| W2-D2 | Data Balancing & Weighted Sampling | Smoothed sample weights + balanced loader |
| W2-D3 | Grad-CAM Integration | Implemented explainability via `src/gradcam.py` |
| W2-D4 | Grad-CAM Refinement & Testing | Added `generate_cam()` + passing unit/integration tests |
| W2-D5 | Performance Review & Clinical Report | `performance_report_v1.md`, Grad-CAM montages, ROC/PR curves |

---

## Model Outcomes

| Variant | Accuracy | Loss | Notes |
|----------|-----------|------|-------|
| Baseline | 0.7518 | 0.4999 | Solid starting point |
| Balanced | 0.8115 | 0.4128 | Improved recall on minority class |
| Fine-Tuned | 0.9985 | 0.0049 | Excellent generalization and focus |

---

## Explainability Insights
- Grad-CAM heatmaps now align with clinical regions of interest.  
- Overlays exported to `/reports/week2_gradcam_refinement/` and `/reports/week2_performance_review/`.  
- Fine-tuned ResNet-50 produced sharper, diagnostically meaningful activations.

---

## Key Learning
- Balancing datasets significantly improves model fairness.
- Explainability pipelines can be systematically tested and versioned.
- Structured documentation (via performance reports) bridges ML outputs and clinical communication.

---

## Next Steps (Week 3)
- Integrate Grad-CAM visualization and metrics into the **Flask-based clinician dashboard**.  
- Develop an interactive report viewer for ROC, PR, and Grad-CAM images.  
- Begin drafting **Clinical Report v2** and web integration prototype.

---

**Maintained by:** Adrian Adewunmi  
**Repository:** [AI-Assisted-Pneumonia-Detection-Project](https://github.com/AAdewunmi/AI-Assisted-Pneumonia-Detection-Project)
