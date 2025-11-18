from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

def train_svm(X, y, cv=5, n_jobs=-1):
    base = SVC(kernel="rbf", probability=True)
    grid = {
        "C": [0.5, 1, 2, 4],
        "gamma": ["scale", 0.01, 0.001]
    }
    gs = GridSearchCV(base, grid, cv=cv, scoring="f1_macro", n_jobs=n_jobs, refit=True)
    gs.fit(X, y)
    # Calibrate for better probabilities
    clf = CalibratedClassifierCV(gs.best_estimator_, cv=3)
    clf.fit(X, y)
    return clf, gs.best_params_

