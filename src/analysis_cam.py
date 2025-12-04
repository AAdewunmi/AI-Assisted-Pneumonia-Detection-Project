from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from src.gradcam import GradCAM


def generate_gradcam_overlay(model, img_path: str, out_dir="static/gradcam"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    img_cv = np.array(img)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0)

    cam = GradCAM(model, target_layer_name="layer4")
    heatmap = cam.generate(tensor)

    heatmap_resized = cv2.resize(np.uint8(255 * heatmap), (img_cv.shape[1],
                                                           img_cv.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.5, heatmap_color, 0.5, 0)

    out_path = out_dir / f"cam_{Path(img_path).stem}.png"
    cv2.imwrite(str(out_path), overlay)
    return out_path
