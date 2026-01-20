import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def fast_restore_kernel(img, output, mask, intensity, edge, fractional, 
                        params_relax, params_edge_thr, iterations):
    """
    The JIT-compiled heart of MAPD.
    """
    h, w = img.shape
    pad = 2 # 5x5 window max
    
    # Pre-calculate weights map to save time inside loop
    # Weight = Base(Intensity) * Edge_Penalty * Fractional_Boost
    weights = (0.1 + 9.9 * intensity) * (1.0 + 0.3 * fractional)
    
    for r in range(h):
        for c in range(w):
            if edge[r, c] > params_edge_thr:
                weights[r, c] *= 0.5
                
    # Iterative Restoration
    current = img.copy()
    next_img = img.copy()
    
    for it in range(iterations):
        for r in prange(pad, h - pad):
            for c in range(pad, w - pad):
                if mask[r, c] == 0: # Skip clean pixels
                    continue
                
                # Get neighbors (manual unroll for speed)
                # Using 3x3 for speed in inner loop
                vals = np.zeros(9, dtype=np.float32)
                count = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i==0 and j==0: continue
                        if mask[r+i, c+j] == 0: # Only use valid pixels
                            vals[count] = current[r+i, c+j]
                            count += 1
                
                if count < 3: continue
                
                # Sort valid neighbors
                for i in range(count):
                    for j in range(i+1, count):
                        if vals[i] > vals[j]:
                            vals[i], vals[j] = vals[j], vals[i]
                
                med = vals[count//2]
                w_val = weights[r, c]
                
                # Decision: Poly fit vs Median
                if w_val > 8.0 and count > 4:
                    # Simple linear fit estimation (omitted for brevity, using mean of extremes)
                    fit_val = (vals[0] + vals[count-1]) / 2.0 
                    restored = fit_val
                elif w_val < 0.3:
                    restored = med
                else:
                    alpha = (w_val - 0.3) / 7.7
                    restored = alpha * med + (1 - alpha) * med # Simplify for speed demo
                
                next_img[r, c] = (1 - params_relax) * current[r, c] + params_relax * restored
        
        current[:] = next_img[:]
        
    output[:] = current[:]
