import numpy as np
import cv2
from scipy.ndimage import median_filter

class FeatureExtractor:
    def __init__(self):
        self.cache = {}

    def get_fractional_prior(self, img, alpha=1.3):
        """FFT-based Fractional Derivative Response."""
        h, w = img.shape
        key = (h, w, alpha)
        if key not in self.cache:
            fx = np.fft.fftfreq(h).reshape(-1, 1)
            fy = np.fft.fftfreq(w).reshape(1, -1)
            mag = np.sqrt(fx**2 + fy**2)
            self.cache[key] = np.power(np.maximum(mag, 1e-8), alpha)
        
        fft_img = np.fft.fft2(img.astype(np.float32))
        response = np.fft.ifft2(fft_img * self.cache[key]).real
        return (response - response.min()) / (response.max() - response.min() + 1e-8)

    def extract_all(self, img, alpha=1.3):
        img_f = img.astype(np.float32)
        
        # 1. Intensity & Local Variance
        mean = cv2.GaussianBlur(img_f, (3,3), 0.5)
        variance = (img_f - mean)**2
        variance = cv2.GaussianBlur(variance, (5,5), 1.0)
        variance = variance / (variance.max() + 1e-8)
        
        # 2. Structure Tensor (Gradient)
        gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1)
        edge = np.sqrt(gx**2 + gy**2)
        edge = edge / (edge.max() + 1e-8)
        
        # 3. Fractional Prior
        frac = self.get_fractional_prior(img_f, alpha)
        
        return {
            "intensity": img_f / 255.0,
            "edge": edge,
            "variance": variance,
            "fractional": frac
        }
