import os, argparse, json, joblib
import numpy as np
import cv2
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# ---------- Import your project modules ----------
from stage1_preprocessing import preprocess_clahe
from stage2_segmentation import efcm_segmentation, morphological_refine
from stage3_feature_extraction import feature_extraction_main

# ==========================================================
# Load Artifacts
# ==========================================================
def load_artifacts(art_dir):
    scaler = joblib.load(os.path.join(art_dir, "scaler.pkl"))
    selector = joblib.load(os.path.join(art_dir, "selector.pkl"))
    with open(os.path.join(art_dir, "label_encoder.json")) as f:
        classes = json.load(f)["classes"]
    base = joblib.load(os.path.join(art_dir, "base_models.pkl"))
    wsv = joblib.load(os.path.join(art_dir, "weighted.pkl"))["weighted"]
    stack = joblib.load(os.path.join(art_dir, "stack_pack.pkl"))["stack"]
    with open(os.path.join(art_dir, "feature_names.json")) as f:
        feat_names = json.load(f)["feature_names"]
    return scaler, selector, classes, base, wsv, stack, feat_names

# ==========================================================
# Add Text to Image
# ==========================================================
def add_text(img, text, position=(10, 10), color=(255, 0, 0)):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, fill=color, font=font)
    return np.array(pil_img)

# ==========================================================
# Full Pipeline with All Stages
# ==========================================================
def predict_full_pipeline(
    img, scaler, selector, classes, base, wsv, stack, feat_names
):
    if img is None:
        return [None]*6 + ["No image"]

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # === STAGE 1: Preprocessing ===
    pre_img = preprocess_clahe(img_bgr)
    pre_rgb = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    pre_rgb = add_text(pre_rgb, "1. Preprocessed (CLAHE + WB + Bilateral)", (10, 10), (0, 255, 0))

    # === STAGE 2: Segmentation ===
    mask_raw = efcm_segmentation(pre_img, n_clusters=3, alpha_base=0.35, max_iter=25)
    mask = morphological_refine(mask_raw, min_size=200, iters=2)
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_colored[:, :, 2] = 0  # Red mask
    mask_colored[:, :, 1] = 0
    seg_overlay = cv2.addWeighted(pre_rgb, 0.7, mask_colored, 0.3, 0)
    seg_overlay = add_text(seg_overlay, "2. Segmented Cell (EFCM + Morphology)", (10, 10), (255, 0, 0))

    # === STAGE 3: Feature Extraction ===
    feats = feature_extraction_main(pre_img, mask)
    top_feats = sorted(
        [(k, v) for k, v in feats.items() if k in feat_names],
        key=lambda x: abs(x[1]), reverse=True
    )[:5]
    feat_text = "Top 5 Features:\n" + "\n".join([f"{k}: {v:.3f}" for k, v in top_feats])
    feat_img = np.ones((200, 500, 3), dtype=np.uint8) * 255
    feat_img = add_text(feat_img, feat_text, (10, 10), (0, 0, 0))
    feat_img = add_text(feat_img, "3. Extracted 102 → Selected 48 Features", (10, 170), (0, 0, 255))

    # === STAGE 4: Prediction ===
    X = np.array([feats.get(k, 0) for k in feat_names], dtype=np.float32).reshape(1, -1)
    Xs = selector.transform(scaler.transform(X))

    Pw = wsv.predict_proba(Xs)[0]
    Ps = stack.predict_proba(Xs)[0]
    P = (Pw + Ps) / 2.0
    idx = int(np.argmax(P))
    conf = float(np.max(P) * 100)
    label = classes[idx]

    # === STAGE 5: Result ===
    status_map = {
        "Benign": f"✅ **No Cancer Detected — Benign Cells Observed (Healthy Sample)**\n\nConfidence: {conf:.1f}%",
        "Early": f"🩸 **Early-Stage Abnormality — Mild Morphological Deviations (Monitor Closely)**\n\nConfidence: {conf:.1f}%",
        "Pre": f"⚠️ **Pre-Cancerous Stage — Atypical Cells Detected (Requires Further Evaluation)**\n\nConfidence: {conf:.1f}%",
        "Pro": f"🚨 **Progressive / Malignant Stage — Blood Cancer Detected (Immediate Attention Needed)**\n\nConfidence: {conf:.1f}%",
    }
    result_text = status_map.get(label, f"Stage: {label}\nConfidence: {conf:.1f}%")
    result_img = np.ones((150, 500, 3), dtype=np.uint8) * 255
    color = (0, 255, 0) if label == "Benign" else (255, 0, 0)
    result_img = add_text(result_img, result_text, (10, 20), color)
    result_img = add_text(result_img, "5. Final Diagnosis (Ensemble)", (10, 100), (0, 0, 255))

    # === STAGE 6: Final Overlay ===
    final_overlay = cv2.addWeighted(pre_img, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    final_overlay = cv2.cvtColor(final_overlay, cv2.COLOR_BGR2RGB)
    final_overlay = add_text(final_overlay, f"6. Final Result: {label} ({conf:.1f}%)", (10, 10), (255, 255, 0))

    # Save for download
    os.makedirs("tmp_outputs", exist_ok=True)
    out_path = "tmp_outputs/segmented_overlay.png"
    cv2.imwrite(out_path, cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR))

    status_text = status_map.get(label, f"{label} ({conf:.2f}%)")
    return (pre_rgb, seg_overlay, feat_img, result_img, final_overlay, out_path, status_text)
# ==========================================================
# Gradio Interface
# ==========================================================
def main(args):
    artifacts = load_artifacts(args.artifacts)

    def predict(image):
        return predict_full_pipeline(image, *artifacts)

    with gr.Blocks(title="Blood Cancer Stage Classifier") as demo:
        gr.Markdown("# 🩸 Blood Cancer Stage Classification — Full Pipeline Visualization")
        gr.Markdown("Upload a microscopic smear image to classify the blood sample stage and visualize the analyzed regions using the segmentation overlay. "
            "You can also download the processed overlay for documentation or reports.")
        with gr.Row():
            inp = gr.Image(type="numpy", label="Upload Smear Image")
            out1 = gr.Image(label="1. Preprocessed")
            out2 = gr.Image(label="2. Segmented Cell")
            out3 = gr.Image(label="3. Top Features")
        with gr.Row():
            out4 = gr.Image(label="4. Diagnosis")
            out5 = gr.Image(label="5. Final Overlay")
            download = gr.File(label="Download Overlay")
            status = gr.Markdown()

        inp.change(predict, inp, [out1, out2, out3, out4, out5, download, status])

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default="./artifacts_v2", help="Path to trained model artifacts directory")
    args = ap.parse_args()
    main(args)
