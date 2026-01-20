import sys
import os
import cv2
import time

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mapd.engine import MAPDEngine
from mapd.utils import add_noise, save_image

def main():
    print("Initializing MAPD-Ultimate...")
    # Try to load weights if trained
    weights = "weights/meta_net.pth" if os.path.exists("weights/meta_net.pth") else None
    engine = MAPDEngine(checkpoint_path=weights)
    
    # Create dummy data
    print("Generating test data...")
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.putText(img, "MAPD", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, 200, 10)
    
    # Add noise
    noisy, mask = add_noise(img, density=0.4)
    save_image("demo_noisy.png", noisy)
    
    # Run
    print("Restoring...")
    start = time.time()
    clean = engine.process(noisy, mask)
    end = time.time()
    
    print(f"Done in {end-start:.4f}s")
    save_image("demo_clean.png", clean)
    print("Results saved: demo_noisy.png, demo_clean.png")

if __name__ == "__main__":
    main()
