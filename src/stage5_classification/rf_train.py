from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_rf(X, y, cv=5, n_jobs=-1):
    base = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=n_jobs)
    grid = {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    gs = GridSearchCV(base, grid, cv=cv, scoring="f1_macro", n_jobs=n_jobs, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_

