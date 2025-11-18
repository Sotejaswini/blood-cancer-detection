import cv2
import numpy as np

# ---------- Gray-World White Balance ----------
def gray_world_wb(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32) + 1e-6
    avg_b, avg_g, avg_r = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    img[:,:,0] *= (avg_gray / avg_b)
    img[:,:,1] *= (avg_gray / avg_g)
    img[:,:,2] *= (avg_gray / avg_r)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ---------- Bilateral Filter (edge-preserving) ----------
def bilateral_filter(img_bgr: np.ndarray, d=7, sigma_color=50, sigma_space=50) -> np.ndarray:
    return cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

# ---------- CLAHE on LAB-L channel ----------
def clahe_enhance(img_bgr: np.ndarray, clip_limit=2.0, tile_grid_size=(8,8)) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

# ---------- Full Stage-1 Pipeline ----------
def preprocess_clahe(img_bgr: np.ndarray) -> np.ndarray:
    wb = gray_world_wb(img_bgr)
    smooth = bilateral_filter(wb, d=7, sigma_color=60, sigma_space=60)
    out = clahe_enhance(smooth, clip_limit=2.5, tile_grid_size=(8,8))
    return out

