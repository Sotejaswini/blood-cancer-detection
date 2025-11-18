import os, json, argparse, glob
import numpy as np
import pandas as pd
import joblib
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from stage1_preprocessing import preprocess_clahe
from stage2_segmentation import efcm_segmentation, morphological_refine
from stage3_feature_extraction import feature_extraction_main
from stage4_feature_selection import rfe_selector

from stage5_classification.svm_train import train_svm
from stage5_classification.rf_train import train_rf
from stage5_classification.et_train import train_et
from stage5_classification.weighted_voting import WeightedSoftVoting
from stage5_classification.stacked_ensemble import StackedEnsemble

def load_images(root):
    # Expect: root/Benign, root/Early, root/Pre, root/Pro
    classes = ["Benign","Early","Pre","Pro"]
    items = []
    for c in classes:
        for p in glob.glob(os.path.join(root, c, "*")):
            if p.lower().endswith((".png",".jpg",".jpeg",".tif",".bmp")):
                items.append((p, c))
    return items, classes

def build_features(items, save_dir="./artifacts_v2", checkpoint_every=200):
    """
    Extracts features from all images with live progress + auto checkpoints.
    """
    os.makedirs(save_dir, exist_ok=True)
    records = []
    tmp_csv = os.path.join(save_dir, "tmp_features.csv")

    print(f"[INFO] Starting feature extraction for {len(items)} images...")
    for idx, (path, lab) in enumerate(tqdm(items, desc="Extracting features", ncols=80)):
        img = cv2.imread(path)
        if img is None:
            continue

        try:
            # Stage 1 – Preprocessing
            pp = preprocess_clahe(img)

            # Stage 2 – EFCM + IMP
            mask0 = efcm_segmentation(pp, n_clusters=3, m=2.0, max_iter=25, alpha_base=0.35, window=3)
            mask = morphological_refine(mask0, min_size=200, iters=2)

            # Stage 3 – Feature extraction
            feats = feature_extraction_main(pp, mask)
            feats["label"] = lab
            feats["path"] = path
            records.append(feats)

            # ---- Checkpoint every N images ----
            if (idx + 1) % checkpoint_every == 0:
                pd.DataFrame(records).to_csv(tmp_csv, index=False)
                print(f"[checkpoint] Saved progress at {idx+1}/{len(items)} → {tmp_csv}")

        except Exception as e:
            print(f"[WARN] Skipped {path} due to error: {e}")

    df = pd.DataFrame(records)
    print(f"[DONE] Extracted features for {len(df)} valid images.")
    return df

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    print("[INFO] Loading image paths...")
    items, classes = load_images(args.data_root)
    print(f"[INFO] Found {len(items)} images.")

    # Extract features (cached if exists)
    feat_csv = os.path.join(args.save_dir, "summary.csv")
    if os.path.exists(feat_csv) and not args.reextract:
        print("[INFO] Using cached features.")
        df = pd.read_csv(feat_csv)
    else:
        df = build_features(items)
        df.to_csv(feat_csv, index=False)
        print(f"[INFO] Saved features to {feat_csv}")

    # Prepare X, y
    y_raw = df["label"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes_ = np.array(le.classes_)
    with open(os.path.join(args.save_dir, "label_encoder.json"), "w") as f:
        json.dump({"classes": le.classes_.tolist()}, f)

    X = df.drop(columns=["label","path"]).values.astype(np.float32)

    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(args.save_dir, "scaler.pkl"))

    # Split
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.2, random_state=42, stratify=ytr)

    # Feature selection (RFE on training only)
    print("[INFO] Feature selection (RFE, keep=48)...")
    selector = rfe_selector(Xtr, ytr, keep=48, random_state=0)
    joblib.dump(selector, os.path.join(args.save_dir, "selector.pkl"))

    Xtr_s = selector.transform(Xtr)
    Xval_s = selector.transform(Xval)
    Xte_s  = selector.transform(Xte)

    # Base models
    print("[INFO] Training SVM...")
    svm_clf, svm_params = train_svm(Xtr_s, ytr, cv=5)
    print("[INFO] Training RF...")
    rf_clf, rf_params = train_rf(Xtr_s, ytr, cv=5)
    print("[INFO] Training ET...")
    et_clf, et_params = train_et(Xtr_s, ytr, cv=5)

    # Weighted soft voting (weights from validation F1 per class)
    print("[INFO] Weighted soft voting...")
    wsv = WeightedSoftVoting([svm_clf, rf_clf, et_clf], classes_=np.arange(len(classes_)))
    wsv.fit_weights(Xval_s, yval)

    # Stacking (OOF)
    print("[INFO] Stacked ensemble...")
    stack = StackedEnsemble([svm_clf, rf_clf, et_clf], classes_=np.arange(len(classes_)))
    stack.fit(np.concatenate([Xtr_s, Xval_s]), np.concatenate([ytr, yval]))

    # Save base models
    joblib.dump({"svm":svm_clf, "rf":rf_clf, "et":et_clf},
                os.path.join(args.save_dir, "base_models.pkl"))
    # Save ensembles
    joblib.dump({"weighted":wsv}, os.path.join(args.save_dir, "weighted.pkl"))
    joblib.dump({"stack":stack}, os.path.join(args.save_dir, "stack_pack.pkl"))

    # Save feature names
    feat_names = [c for c in df.columns if c not in ("label","path")]
    with open(os.path.join(args.save_dir, "feature_names.json"), "w") as f:
        json.dump({"feature_names": feat_names}, f)

    # Metadata
    meta = {
        "svm_params": svm_params,
        "rf_params": rf_params,
        "et_params": et_params,
        "classes": classes_.tolist()
    }
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[DONE] Training completed. Artifacts saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="path to data/Original")
    ap.add_argument("--save_dir", type=str, default="./artifacts_v2")
    ap.add_argument("--reextract", action="store_true")
    args = ap.parse_args()
    main(args)

