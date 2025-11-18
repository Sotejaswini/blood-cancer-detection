import numpy as np
from sklearn.metrics import f1_score

class WeightedSoftVoting:
    """
    Per-class weights from validation F1s.
    P_ens(c) = sum_i w_{c,i} * P_i(c)
    """
    def __init__(self, clfs, classes_):
        self.clfs = clfs
        self.classes_ = classes_
        self.weights_ = None  # (n_models, n_classes)

    def fit_weights(self, X_val, y_val):
        C = len(self.classes_)
        M = len(self.clfs)
        # compute per-class F1 for each model
        W = np.zeros((M, C), dtype=float)
        for m, clf in enumerate(self.clfs):
            yhat = clf.predict(X_val)
            for ci, c in enumerate(self.classes_):
                f1 = f1_score((y_val==c).astype(int), (yhat==c).astype(int), zero_division=0)
                W[m, ci] = f1
        # normalize across models per class to sum to 1
        W = W + 1e-9
        W = W / W.sum(axis=0, keepdims=True)
        self.weights_ = W

    def predict_proba(self, X):
        # average weighted by per-class weights
        probs = [clf.predict_proba(X) for clf in self.clfs]  # list of (N, C)
        P = np.stack(probs, axis=0)  # (M, N, C)
        if self.weights_ is None:
            # fall back to equal
            W = np.ones((len(self.clfs), P.shape[-1])) / len(self.clfs)
        else:
            W = self.weights_
        # broadcast (M,1,C) * (M,N,C) -> (M,N,C)
        ens = (W[:,None,:] * P).sum(axis=0)
        return ens

    def predict(self, X):
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        return self.classes_[idx]

