import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.svm import LinearSVC

def rfe_selector(X, y, keep=48, random_state=0):
    base = LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=random_state)
    sel = RFE(estimator=base, n_features_to_select=keep, step=0.1)
    sel.fit(X, y)
    return sel

def kbest_selector(X, y, k=64):
    # chi2 requires non-negative
    Xn = X - X.min(axis=0, keepdims=True)
    sel = SelectKBest(score_func=chi2, k=k)
    sel.fit(Xn, y)
    return sel

