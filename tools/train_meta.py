import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mapd.models import MetaParameterNet
from mapd.utils import add_noise, load_config

def generate_synthetic_batch(batch_size=8, size=64):
    """
    Generate synthetic training data (blobs/lines) on the fly.
    We don't need real datasets to learn the mapping from Noise->Params.
    """
    batch_x = []
    batch_y = []
    
    for _ in range(batch_size):
        # 1. Create clean image
        img = np.zeros((size, size), dtype=np.uint8)
        # Add random circles/rects
        for _ in range(3):
            cv2.circle(img, (np.random.randint(0,size), np.random.randint(0,size)), 
                       np.random.randint(5,15), np.random.randint(50,200), -1)
        
        # 2. Add noise with random density
        density = np.random.uniform(0.1, 0.6)
        noisy, _ = add_noise(img, density)
        
        # 3. Target Parameters (Heuristic Ground Truth)
        # Higher noise -> Higher Alpha, Higher Relax
        target_alpha = 0.2 + density  # Norm [0,1]
        target_relax = 0.5 + density * 0.5
        target_edge = 1.0 - density
        target_mode = 1.0 if density > 0.4 else 0.0
        
        batch_x.append(noisy / 255.0)
        batch_y.append([target_alpha, target_relax, target_edge, target_mode])
        
    return torch.tensor(np.array(batch_x), dtype=torch.float32).unsqueeze(1), \
           torch.tensor(np.array(batch_y), dtype=torch.float32)

def train():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    model = MetaParameterNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training Loop
    for epoch in range(100): # Quick demo training
        inputs, targets = generate_synthetic_batch(32)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
    # Save
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/meta_net.pth")
    print("✅ Model saved to weights/meta_net.pth")

if __name__ == "__main__":
    train()
