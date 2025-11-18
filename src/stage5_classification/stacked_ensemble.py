import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

class StackedEnsemble:
    """
    10-fold OOF stacking with Logistic Regression meta-learner.
    Meta features = concatenated base-model probabilities.
    """
    def __init__(self, base_clfs, classes_):
        self.base_clfs = base_clfs
        self.meta = LogisticRegression(max_iter=200)
        self.classes_ = classes_

    def fit(self, X, y, n_splits=10, random_state=0):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_meta = np.zeros((len(y), len(self.base_clfs)*len(self.classes_)))
        for train_idx, val_idx in skf.split(X, y):
            Xtr, Xval = X[train_idx], X[val_idx]
            ytr = y[train_idx]
            # train copies of base clfs
            trained = []
            for clf in self.base_clfs:
                clone = clf
                clone.fit(Xtr, ytr)
                trained.append(clone)
            # build meta features on val
            Ps = [cl.predict_proba(Xval) for cl in trained]
            oof_meta[val_idx] = np.concatenate(Ps, axis=1)
        self.meta.fit(oof_meta, y)

        # final fit of base clfs on full data for inference
        for clf in self.base_clfs:
            clf.fit(X, y)

    def predict_proba(self, X):
        Ps = [cl.predict_proba(X) for cl in self.base_clfs]
        metaX = np.concatenate(Ps, axis=1)
        return self.meta.predict_proba(metaX)

    def predict(self, X):
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        return self.classes_[idx]

