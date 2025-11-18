import numpy as np
import cv2
from skimage.measure import regionprops, label
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import color
from scipy.stats import entropy as shannon_entropy

# ------- helpers -------
def _safe_mask_bbox(mask):
    lab = label(mask > 0)
    if lab.max() == 0:
        return mask, None
    # keep largest component
    sizes = [(lab == i).sum() for i in range(1, lab.max()+1)]
    cc = 1 + int(np.argmax(sizes))
    return (lab == cc).astype(np.uint8), cc

# ------- Shape Features -------
def shape_features(mask: np.ndarray):
    feats = {}
    lab = label(mask > 0)
    if lab.max() == 0:
        for k in ["area","perimeter","eccentricity","solidity","circularity","convexity"]:
            feats[f"shape_{k}"] = 0.0
        for i in range(7):
            feats[f"hu_{i}"] = 0.0
        return feats

    props = regionprops(lab)
    rp = max(props, key=lambda p: p.area)

    area = float(rp.area)
    perimeter = float(rp.perimeter) if rp.perimeter > 0 else 1.0
    ecc = float(rp.eccentricity) if hasattr(rp, "eccentricity") else 0.0
    solidity = float(rp.solidity) if hasattr(rp, "solidity") else 0.0
    circularity = float(4.0 * np.pi * area / (perimeter ** 2))
    convex_perim = float(rp.perimeter) if rp.perimeter > 0 else 1.0  # approximate
    convexity = float(convex_perim / perimeter)

    feats.update({
        "shape_area": area,
        "shape_perimeter": perimeter,
        "shape_eccentricity": ecc,
        "shape_solidity": solidity,
        "shape_circularity": circularity,
        "shape_convexity": convexity
    })

    # Hu Moments (7)
    m = cv2.moments((lab==rp.label).astype(np.uint8))
    hu = cv2.HuMoments(m).flatten()
    for i, h in enumerate(hu):
        feats[f"hu_{i}"] = float(h)
    return feats

# ------- Color Features (HSV + LAB stats) -------
def color_features(img_bgr: np.ndarray, mask: np.ndarray):
    feats = {}
    if mask.max() == 0:
        for space in ["HSV","LAB"]:
            for stat in ["mean","std","min","max"]:
                for c in range(3):
                    feats[f"{space}_{stat}_{c}"] = 0.0
        return feats

    m = (mask > 0)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    for space_name, space in [("HSV",hsv), ("LAB",lab)]:
        vals = space[m]
        for c in range(3):
            ch = vals[:,c].astype(np.float32)
            feats[f"{space_name}_mean_{c}"] = float(np.mean(ch))
            feats[f"{space_name}_std_{c}"]  = float(np.std(ch))
            feats[f"{space_name}_min_{c}"]  = float(np.min(ch))
            feats[f"{space_name}_max_{c}"]  = float(np.max(ch))
    return feats

# ------- Texture Features: LBP + GLCM + Entropy -------
def texture_features(img_bgr: np.ndarray, mask: np.ndarray, lbp_P=8, lbp_R=1):
    feats = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    m = (mask > 0)
    if m.sum() == 0:
        # fill zeros
        for i in range(59): feats[f"lbp_{i}"] = 0.0
        for name in ["contrast","homogeneity","energy","correlation","ASM"]:
            feats[f"glcm_{name}"] = 0.0
        feats["entropy"] = 0.0
        return feats

    roi = gray[m]
    # LBP uniform
    lbp = local_binary_pattern(gray, P=lbp_P, R=lbp_R, method="uniform")
    lbp_roi = lbp[m]
    n_bins = int(lbp_P*(lbp_P-1)+3)  # uniform bins
    hist, _ = np.histogram(lbp_roi, bins=n_bins, range=(0, n_bins), density=True)
    for i, v in enumerate(hist):
        feats[f"lbp_{i}"] = float(v)

    # GLCM on masked gray (quantize)
    q = np.uint8((gray / 8).clip(0,31))  # 32 gray levels
    qm = q.copy()
    qm[~m] = 0
    glcm = graycomatrix(qm, distances=[1], angles=[0], levels=32, symmetric=True, normed=True)
    for prop in ["contrast","homogeneity","energy","correlation","ASM"]:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop)[0,0])

    # Shannon entropy (masked)
    hist_full, _ = np.histogram(roi, bins=64, range=(0,255), density=True)
    feats["entropy"] = float(shannon_entropy(hist_full + 1e-12, base=2))
    return feats

# ------- Main extraction per image -------
def feature_extraction_main(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    feats = {}
    feats.update(shape_features(mask))
    feats.update(color_features(img_bgr, mask))
    feats.update(texture_features(img_bgr, mask))
    return feats

