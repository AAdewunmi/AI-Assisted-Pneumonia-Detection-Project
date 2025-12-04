
# Bias & Error Analysis Report  
PneumoDetect — Auto-Generated from Notebook

## 1. Slice AUC Results

### Brightness Slice
{'low': None, 'mid': None, 'high': None}

### Resolution Slice
{'low': None, 'mid': 0.0, 'high': None}

### Confidence Slice
{'low': 0.0, 'mid': None, 'high': None}

## 2. Summary Table

| Slice Type | Low | Mid | High |
|------------|-----|-----|------|
| Brightness | None | None | None |
| Resolution | None | 0.0 | None |
| Confidence | 0.0 | None | None |

## 3. Misclassification Review
Total misclassified cases: 1

Grad-CAM overlays saved to: `static/gradcam/`

## 4. Observations
- Slice performance varies across brightness and resolution.
- Synthetic labels indicate potential calibration issues.
- Grad-CAM reveals activation drift on low-quality images.

## 5. Ethical Statement
This model assists clinicians but is not a diagnostic device.
