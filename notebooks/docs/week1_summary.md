# Week 1 Summary — PneumoDetect Baseline Model

## Dataset & Preprocessing
- Subset of RSNA Pneumonia Detection Challenge dataset (~2 GB)
- Images resized to 224×224, normalized using ImageNet stats
- Training set balanced with partial sampling

## Baseline Model
- Architecture: Pretrained ResNet-50 (frozen convolutional base)
- Classification head: 2-class linear layer

## Performance Metrics
- **AUC:** 0.626
- **Average Precision (AP):** 0.462

## Reflections
- The model learns basic pneumonia patterns but shows class imbalance bias.
- False negatives remain a key risk — next step: handle imbalance and apply Grad-CAM.
