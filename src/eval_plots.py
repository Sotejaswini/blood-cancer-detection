import os, json, argparse, joblib, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from stage6_evaluation.eval_metrics import compute_metrics
from stage6_evaluation.eval_plots import plot_confusion, plot_roc_ovr, plot_acc_bar, latex_table


def main(args):
    art = args.artifacts
    os.makedirs(os.path.join(art, "plots"), exist_ok=True)

    # === Load data ===
    df = pd.read_csv(os.path.join(art, "summary.csv"))
    with open(os.path.join(art, "label_encoder.json")) as f:
        classes = np.array(json.load(f)["classes"])

    feat_names = [c for c in df.columns if c not in ("label", "path")]
    X = df[feat_names].values.astype(np.float32)
    y_raw = df["label"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # === Load preprocessing objects ===
    scaler = joblib.load(os.path.join(art, "scaler.pkl"))
    selector = joblib.load(os.path.join(art, "selector.pkl"))
    Xs = selector.transform(scaler.transform(X))

    # === Split data ===
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.2, random_state=42, stratify=ytr)

    # === Load trained models ===
    base = joblib.load(os.path.join(art, "base_models.pkl"))
    wsv = joblib.load(os.path.join(art, "weighted.pkl"))["weighted"]
    stack = joblib.load(os.path.join(art, "stack_pack.pkl"))["stack"]

    names = ["SVM", "RF", "ET", "Weighted", "Stacked"]
    model_keys = ["svm", "rf", "et"]
    probs, preds = [], []

    # --- Base models ---
    for k in model_keys:
        P = base[k].predict_proba(Xte)
        yhat = np.argmax(P, axis=1)
        probs.append(P)
        preds.append(yhat)

    # --- Ensembles ---
    Pw = wsv.predict_proba(Xte)
    Ps = stack.predict_proba(Xte)
    yw, ys = np.argmax(Pw, axis=1), np.argmax(Ps, axis=1)
    probs += [Pw, Ps]
    preds += [yw, ys]

    # === Metrics, plots, and results ===
    metrics_map = {}
    acc_bar = {}

    for name, P, yhat in zip(names, probs, preds):
        m = compute_metrics(yte, yhat, P, classes_=np.arange(len(classes)))
        metrics_map[name] = {
            k: (
                float(v)
                if isinstance(v, (int, float, np.floating))
                else (v.tolist() if k == "cm" else None)
            )
            for k, v in m.items()
        }
        acc_bar[name] = m["accuracy"]

        # Confusion matrices and ROC curves
        plot_confusion(m["cm"], classes, os.path.join(art, "plots", f"cm_{name}.png"))
        if name in ["Weighted", "Stacked"]:
            plot_roc_ovr(yte, P, classes, os.path.join(art, "plots", f"roc_{name}.png"))

    # --- Accuracy Bar Plot ---
    plot_acc_bar(acc_bar, os.path.join(art, "plots", "acc_bar.png"))

    # --- LaTeX Table ---
    latex_table(metrics_map, classes, os.path.join(art, "results_table.tex"))

    # === Create Summary CSV ===
    df_out = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": metrics_map[name]["accuracy"],
            "Precision": metrics_map[name]["precision"],
            "Recall": metrics_map[name]["recall"],
            "F1": metrics_map[name]["f1_score"],
            "AUC": metrics_map[name]["auc"],
        }
        for name in names
    ])
    csv_path = os.path.join(art, "model_metrics_summary.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"[SAVED] Metrics summary -> {csv_path}")

    # === Enhanced Accuracy Plot ===
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_out["Model"], df_out["Accuracy"], color="#6EC6FF", edgecolor="black")

    # Value labels on bars
    for bar, acc in zip(bars, df_out["Accuracy"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.015,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.title("Model Comparison on Enhanced Blood Cell Classification", fontsize=13, weight="bold")
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.ylim(0.9, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    plot_path_png = os.path.join(art, "plots", "accuracy_comparison.png")
    plot_path_svg = os.path.join(art, "plots", "accuracy_comparison.svg")
    plt.savefig(plot_path_png, dpi=400)
    plt.savefig(plot_path_svg)
    plt.close()
    print(f"[SAVED] Accuracy plots -> {plot_path_png}, {plot_path_svg}")

    # === Save metrics to JSON ===
    with open(os.path.join(art, "metrics.json"), "w") as f:
        json.dump(metrics_map, f, indent=2)

    print("[DONE] Plots + CSV + LaTeX saved under artifacts_v2/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, required=True)
    args = ap.parse_args()
    main(args)

