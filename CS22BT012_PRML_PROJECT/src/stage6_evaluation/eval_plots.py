import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import os

def plot_confusion(cm, classes, save_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_ovr(y_true, y_prob, classes, save_path):
    plt.figure(figsize=(7,6))
    n_classes = len(classes)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        plt.plot(fpr, tpr, color=color,
                 label=f'ROC curve (class {classes[i]}) (area = {auc(fpr,tpr):.2f})')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_acc_bar(acc_map, save_path):
    plt.figure(figsize=(7,4))
    plt.bar(acc_map.keys(), acc_map.values(), color='skyblue')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def latex_table(metrics_map, classes, save_path):
    """Save LaTeX-formatted table summarizing metrics."""
    import pandas as pd
    rows = []
    for model, vals in metrics_map.items():
        rows.append({
            "Model": model,
            "Accuracy": f"{vals['accuracy']:.3f}",
            "Precision": f"{vals['precision']:.3f}",
            "Recall": f"{vals['recall']:.3f}",
            "F1": f"{vals['f1_score']:.3f}",
            "AUC": f"{vals['auc']:.3f}" if vals['auc'] else "NA"
        })
    df = pd.DataFrame(rows)
    with open(save_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))

