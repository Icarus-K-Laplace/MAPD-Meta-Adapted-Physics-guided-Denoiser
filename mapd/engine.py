import torch
import numpy as np
import os
from .models import MetaParameterNet
from .features import FeatureExtractor
from .core import fast_restore_kernel

class MAPDEngine:
    def __init__(self, checkpoint_path=None, use_gpu=False):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.meta_net = MetaParameterNet().to(self.device)
        self.extractor = FeatureExtractor()
        self.loaded = False
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.meta_net.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.loaded = True
            print(f"Loaded Meta-Net from {checkpoint_path}")
        else:
            print("Warning: No checkpoint found. Using random init (for demo).")

    def predict_params(self, img):
        """Inference the Meta-Net to get adaptive parameters."""
        self.meta_net.eval()
        img_tensor = torch.from_numpy(img / 255.0).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.meta_net(img_tensor).cpu().numpy()[0]
            
        # Map [0,1] output to physical parameters
        return {
            "alpha": 1.0 + out[0] * 0.8,    # 1.0 ~ 1.8
            "relax": 0.6 + out[1] * 0.3,    # 0.6 ~ 0.9
            "edge_thr": 0.1 + out[2] * 0.4, # 0.1 ~ 0.5
            "mode": out[3]                  # >0.5 Quality, <0.5 Speed
        }

    def process(self, noisy_img, mask):
        # 1. Meta-Control
        params = self.predict_params(noisy_img)
        print(f"  [Meta] Alpha:{params['alpha']:.2f} | Relax:{params['relax']:.2f} | Mode:{'Quality' if params['mode']>0.5 else 'Speed'}")
        
        # 2. Feature Extraction
        feats = self.extractor.extract_all(noisy_img, alpha=params['alpha'])
        
        # 3. Execution Strategy
        iterations = 4 if params['mode'] > 0.5 else 2
        
        # 4. Core Restoration
        output = np.zeros_like(noisy_img)
        fast_restore_kernel(
            noisy_img.astype(np.float32), 
            output.astype(np.float32), 
            mask,
            feats['intensity'], 
            feats['edge'], 
            feats['fractional'],
            params['relax'], 
            params['edge_thr'], 
            iterations
        )
        
        return output.astype(np.uint8)
