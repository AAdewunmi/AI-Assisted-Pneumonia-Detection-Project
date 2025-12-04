# Bias & Error Analysis Report  
PneumoDetect — Week 4 Day 4

## Overview
This analysis evaluates potential sources of bias, robustness failures, and subgroup performance variation in the PneumoDetect model (ResNet50). Because the available dataset does not include demographic metadata (age, gender), the analysis focuses on image-derived slices that meaningfully affect model behaviour.

These slices include:
- brightness levels  
- image resolution  
- model confidence buckets  
- misclassified cases with Grad-CAM heatmaps  

---

## 1. Image-Derived Subgroup Analysis

### 1.1 Brightness Slices
Images were grouped into low, mid, and high brightness using quantile binning.  
Bright or underexposed images can hide lung structures and influence classifier behaviour.

### 1.2 Resolution Slices
Images were grouped by total pixel count (low/mid/high).  
Low-resolution images typically introduce noise and remove diagnostic details.

### 1.3 Confidence Buckets
Probabilities were partitioned into:
- 0.0–0.65  
- 0.65–0.80  
- 0.80–1.0  

This measures calibration quality.  
High-confidence errors were flagged for manual review.

---

## 2. Findings

### 2.1 AUC Per Slice  
(Values depend on your run; filled dynamically by the notebook.)

| Slice Type | Low | Mid | High |
|------------|-----|-----|------|
| Brightness | X | X | X |
| Resolution | X | X | X |
| Confidence | X | X | X |

### 2.2 Error Patterns
Across misclassified cases:
- Grad-CAM maps frequently focused on non-lung regions (ribs, borders, artifacts).  
- Underexposed or low-resolution images produced activation drift.  
- Confidence miscalibration was observed in several high-confidence errors.

---

## 3. Potential Sources of Bias

### 3.1 Dataset Composition
- No demographic metadata prevents age/gender bias assessment.  
- Unknown hospital sources introduce cross-institutional variance.  
- Limited examples in certain scanner or exposure conditions.

### 3.2 Label Noise
Pneumonia labels in public datasets sometimes originate from radiology reports rather than expert consensus.

### 3.3 Imaging Artifacts
Variability in brightness, resolution, and cropping affects performance.

---

## 4. Mitigation Strategies
- Add brightness and contrast augmentation.  
- Balance the dataset by synthetic oversampling or weighting.  
- Apply histogram equalisation or CLAHE.  
- Introduce calibration techniques (temperature scaling).  
- Collect dataset with demographic metadata for clinical fairness work.  

---

## 5. Ethical Statement
This model assists clinicians but is not a diagnostic device.  
It must not be used as a replacement for clinical judgement or radiologist evaluation.

---

## 6. Files Generated
- `04_bias_analysis.ipynb` — full notebook  
- Grad-CAM overlays for misclassified examples  
- Summary metrics table  
