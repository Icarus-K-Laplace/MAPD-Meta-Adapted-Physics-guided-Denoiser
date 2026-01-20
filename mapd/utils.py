
import cv2
import numpy as np
import os
import yaml

def load_config(path="configs/default.yaml"):
    if not os.path.exists(path):
        # Return default dict if file missing
        return {
            "model": {"input_channels": 1, "hidden_dim": 32},
            "training": {"lr": 1e-3, "batch_size": 16, "epochs": 50}
        }
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_image(path, img):
    """Save image ensuring uint8 format."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)

def add_noise(img, density=0.2):
    """Add salt-and-pepper noise for testing."""
    noise = img.copy()
    h, w = img.shape
    n = int(h * w * density)
    # Fast random indices
    idx = np.random.choice(h * w, n, replace=False)
    coords = np.unravel_index(idx, (h, w))
    
    # Salt & Pepper
    vals = np.random.choice([0, 255], n)
    noise[coords] = vals
    
    mask = np.zeros_like(img)
    mask[coords] = 1
    return noise, mask
