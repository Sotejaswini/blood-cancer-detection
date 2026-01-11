from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

def train_et(X, y, cv=5, n_jobs=-1):
    base = ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=n_jobs)
    grid = {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    gs = GridSearchCV(base, grid, cv=cv, scoring="f1_macro", n_jobs=n_jobs, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_

