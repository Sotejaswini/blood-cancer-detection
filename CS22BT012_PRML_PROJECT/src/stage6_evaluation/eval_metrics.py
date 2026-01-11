import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)

def compute_metrics(y_true, y_pred, y_prob=None, classes_=None):
    """Compute classification metrics and confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted')
    rec  = recall_score(y_true, y_pred, average='weighted')

    auc = None
    if y_prob is not None and y_prob.shape[1] > 1:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except Exception:
            auc = None

    return {
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc if auc is not None else 0.0,
        "cm": cm
    }

