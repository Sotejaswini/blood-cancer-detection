import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk

# ---------- Effective Fuzzy C-Means with spatial context ----------
def efcm_segmentation(img_bgr: np.ndarray, n_clusters=3, m=2.0, max_iter=30,
                      alpha_base=0.3, window=3, seed=0) -> np.ndarray:
    """
    Spatially-regularized FCM.
    - Init centers using Otsu on gray to separate foreground/background,
      plus middle cluster(s) spread by percentiles.
    - Adaptive alpha = alpha_base * norm(local variance).
    Returns: binary mask (foreground ~ nuclei+cytoplasm).
    """
    rng = np.random.default_rng(seed)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = img.shape
    X = img.reshape(-1, 1)

    # --- init centers via Otsu + percentiles
    t = threshold_otsu(img)
    lows = np.percentile(X, 10)
    highs = np.percentile(X, 90)
    if n_clusters == 2:
        centers = np.array([lows, highs], dtype=np.float32).reshape(-1,1)
    else:
        mids = np.linspace(lows, highs, n_clusters)
        centers = mids.reshape(-1,1).astype(np.float32)

    # initialize memberships uniformly
    U = rng.random((X.shape[0], n_clusters)).astype(np.float32)
    U /= (U.sum(axis=1, keepdims=True) + 1e-12)

    # precompute local variance (for alpha)
    k = window
    pad = k // 2
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    var_map = np.zeros_like(img)
    for i in range(H):
        for j in range(W):
            patch = img_pad[i:i+k, j:j+k]
            var_map[i,j] = np.var(patch)
    var_norm = (var_map - var_map.min()) / (np.ptp(var_map) + 1e-12)
    #var_norm = (var_map - var_map.min()) / (var_mapnp.ptp( + 1e-12)
    alpha_map = (alpha_base * var_norm).reshape(-1, 1).astype(np.float32)

    # neighborhood averaging of U
    def neighborhood_avg(Umat):
        Um = Umat.reshape(H, W, n_clusters)
        Um_pad = np.pad(Um, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
        out = np.zeros_like(Um)
        kernel = np.ones((k,k), dtype=np.float32)
        for c in range(n_clusters):
            plane = Um_pad[:,:,c]
            sm = cv2.filter2D(plane, -1, kernel) / (k*k)
            out[:,:,c] = sm[pad:pad+H, pad:pad+W]
        return out.reshape(-1, n_clusters)

    for _ in range(max_iter):
        # update centers
        Um = (U ** m)
        denom = Um.sum(axis=0, keepdims=True) + 1e-12
        centers = (Um.T @ X) / denom.T

        # distances
        dist = np.linalg.norm(X[:,None,:] - centers[None,:,:], axis=2) + 1e-6

        # new memberships
        power = 2.0 / (m - 1.0)
        inv = (dist[:, :, None] / dist[:, None, :]) ** power  # (N, C, C)
        U_new = 1.0 / (inv.sum(axis=2) + 1e-12)               # (N, C)

        # spatial regularization
        U_bar = neighborhood_avg(U)
        U = (1 - alpha_map) * U_new + alpha_map * U_bar
        U /= (U.sum(axis=1, keepdims=True) + 1e-12)

    # choose cluster with min center as foreground (typical for nuclei)
    fg_cluster = np.argmin(centers.flatten())
    mask = (U[:, fg_cluster].reshape(H, W) > 0.5).astype(np.uint8)
    return mask

# ---------- Iterative Morphological Process (IMP) ----------
def morphological_refine(mask: np.ndarray, min_size=200, iters=2) -> np.ndarray:
    m = mask.astype(bool)
    selem_small = disk(1)
    for _ in range(iters):
        m = binary_opening(m, selem_small)
        m = binary_closing(m, selem_small)
        m = remove_small_objects(m, min_size=min_size)
    return (m.astype(np.uint8) * 255)

# (Optional) Chan-Vese (fallback/refinement) — simple wrapper
def chan_vese_refine(img_gray: np.ndarray, init_mask: np.ndarray, max_it=50) -> np.ndarray:
    # Lightweight fallback: we’ll keep morphological_refine as primary (CPU-friendly).
    # Placeholder for future upgrades.
    return init_mask

