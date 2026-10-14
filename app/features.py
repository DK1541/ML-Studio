"""Picklable feature extractors and the catalog of classic (scikit-learn) models."""
import importlib.util

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def has(mod):
    return importlib.util.find_spec(mod) is not None


def _hog(g, cell=8, bins=9):
    """Histogram of oriented gradients in plain NumPy (no OpenCV/skimage version dependence)."""
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins        # unsigned gradients, 0..bins
    b = np.minimum(ang.astype(int), bins - 1)
    h, w = g.shape
    ch, cw = h // cell, w // cell
    hist = np.zeros((ch, cw, bins), dtype="float32")
    for i in range(ch):
        for j in range(cw):
            m = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            bb = b[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            hist[i, j] = np.bincount(bb, weights=m, minlength=bins)
    out = []
    for i in range(max(ch - 1, 1)):
        for j in range(max(cw - 1, 1)):
            blk = hist[i:i + 2, j:j + 2].ravel()
            out.append(blk / (np.linalg.norm(blk) + 1e-6))
    return np.concatenate(out)


class ImageFeatures(BaseEstimator, TransformerMixin):
    """Turns image file paths into a feature vector: resized pixels + HOG + colour histogram."""

    def __init__(self, size=48, use_hog=True, gray=False):
        self.size = size
        self.use_hog = use_hog
        self.gray = gray

    def fit(self, X, y=None):
        return self

    def _one(self, path):
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot read image {path}")
        img = cv2.resize(img, (self.size, self.size))
        feats = []
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        base = g if self.gray else img
        feats.append(base.astype("float32").ravel() / 255.0)
        if self.use_hog:
            feats.append(_hog(g.astype("float32") / 255.0))
        if not self.gray:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
            feats.append(hist / (hist.sum() + 1e-9))
        return np.concatenate(feats)

    def transform(self, X):
        return np.vstack([self._one(p) for p in X])


# --------------------------------------------------------------------------- catalog
# name -> (factory(params) -> estimator, default params, description)

def _sk():
    from sklearn import ensemble, linear_model, naive_bayes, neighbors, neural_network, svm, tree
    from sklearn import discriminant_analysis as da
    return locals()


def classifiers(task="classification"):
    """Returns {name: {"factory": callable, "params": {...}, "group": str}} for sklearn-family models."""
    from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                                  RandomForestClassifier, HistGradientBoostingClassifier, BaggingClassifier)
    from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

    M = {
        "Fine KNN (k=1)": (lambda p: KNeighborsClassifier(**p), {"n_neighbors": 1}, "Nearest neighbors"),
        "KNN": (lambda p: KNeighborsClassifier(**p), {"n_neighbors": 5, "weights": "distance"}, "Nearest neighbors"),
        "Decision Tree": (lambda p: DecisionTreeClassifier(**p), {"max_depth": None}, "Decision trees"),
        "Random Forest": (lambda p: RandomForestClassifier(**p), {"n_estimators": 200, "n_jobs": -1}, "Ensembles"),
        "Extra Trees": (lambda p: ExtraTreesClassifier(**p), {"n_estimators": 200, "n_jobs": -1}, "Ensembles"),
        "Gradient Boosting": (lambda p: GradientBoostingClassifier(**p), {"n_estimators": 150}, "Ensembles"),
        "Hist Gradient Boosting": (lambda p: HistGradientBoostingClassifier(**p), {"max_iter": 200}, "Ensembles"),
        "AdaBoost": (lambda p: AdaBoostClassifier(**p), {"n_estimators": 100}, "Ensembles"),
        "Bagging": (lambda p: BaggingClassifier(**p), {"n_estimators": 30}, "Ensembles"),
        "Logistic Regression": (lambda p: LogisticRegression(**p), {"max_iter": 2000}, "Linear"),
        "Ridge Classifier": (lambda p: RidgeClassifier(**p), {}, "Linear"),
        "SGD (linear SVM)": (lambda p: SGDClassifier(**p), {"max_iter": 2000}, "Linear"),
        "SVM (RBF)": (lambda p: SVC(**p), {"kernel": "rbf", "probability": True}, "SVM"),
        "SVM (Linear)": (lambda p: SVC(**p), {"kernel": "linear", "probability": True}, "SVM"),
        "SVM (Polynomial)": (lambda p: SVC(**p), {"kernel": "poly", "degree": 3, "probability": True}, "SVM"),
        "Linear SVC": (lambda p: LinearSVC(**p), {"max_iter": 5000}, "SVM"),
        "Gaussian Naive Bayes": (lambda p: GaussianNB(**p), {}, "Naive Bayes"),
        "Multinomial Naive Bayes": (lambda p: MultinomialNB(**p), {}, "Naive Bayes"),
        "Complement Naive Bayes": (lambda p: ComplementNB(**p), {}, "Naive Bayes"),
        "Linear Discriminant": (lambda p: LinearDiscriminantAnalysis(**p), {}, "Discriminant"),
        "Quadratic Discriminant": (lambda p: QuadraticDiscriminantAnalysis(**p), {"reg_param": 0.1}, "Discriminant"),
        "Neural Net (MLP)": (lambda p: MLPClassifier(**p), {"hidden_layer_sizes": (128, 64), "max_iter": 400}, "Neural nets"),
    }
    if has("xgboost"):
        from xgboost import XGBClassifier
        M["XGBoost"] = (lambda p: XGBClassifier(**p), {"n_estimators": 200, "eval_metric": "mlogloss"}, "Ensembles")
    if has("lightgbm"):
        from lightgbm import LGBMClassifier
        M["LightGBM"] = (lambda p: LGBMClassifier(**p), {"n_estimators": 200, "verbose": -1}, "Ensembles")
    if has("catboost"):
        from catboost import CatBoostClassifier
        M["CatBoost"] = (lambda p: CatBoostClassifier(**p), {"iterations": 300, "verbose": 0}, "Ensembles")
    return {k: {"factory": f, "params": p, "group": g} for k, (f, p, g) in M.items()}


def regressors():
    from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                                  RandomForestRegressor, HistGradientBoostingRegressor)
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    M = {
        "Linear Regression": (lambda p: LinearRegression(**p), {}, "Linear"),
        "Ridge": (lambda p: Ridge(**p), {"alpha": 1.0}, "Linear"),
        "Lasso": (lambda p: Lasso(**p), {"alpha": 0.1}, "Linear"),
        "Elastic Net": (lambda p: ElasticNet(**p), {"alpha": 0.1}, "Linear"),
        "KNN": (lambda p: KNeighborsRegressor(**p), {"n_neighbors": 5}, "Nearest neighbors"),
        "Decision Tree": (lambda p: DecisionTreeRegressor(**p), {}, "Decision trees"),
        "Random Forest": (lambda p: RandomForestRegressor(**p), {"n_estimators": 200, "n_jobs": -1}, "Ensembles"),
        "Extra Trees": (lambda p: ExtraTreesRegressor(**p), {"n_estimators": 200, "n_jobs": -1}, "Ensembles"),
        "Gradient Boosting": (lambda p: GradientBoostingRegressor(**p), {"n_estimators": 150}, "Ensembles"),
        "Hist Gradient Boosting": (lambda p: HistGradientBoostingRegressor(**p), {"max_iter": 200}, "Ensembles"),
        "AdaBoost": (lambda p: AdaBoostRegressor(**p), {"n_estimators": 100}, "Ensembles"),
        "SVR (RBF)": (lambda p: SVR(**p), {"kernel": "rbf"}, "SVM"),
        "Neural Net (MLP)": (lambda p: MLPRegressor(**p), {"hidden_layer_sizes": (128, 64), "max_iter": 500}, "Neural nets"),
    }
    if has("xgboost"):
        from xgboost import XGBRegressor
        M["XGBoost"] = (lambda p: XGBRegressor(**p), {"n_estimators": 200}, "Ensembles")
    if has("lightgbm"):
        from lightgbm import LGBMRegressor
        M["LightGBM"] = (lambda p: LGBMRegressor(**p), {"n_estimators": 200, "verbose": -1}, "Ensembles")
    return {k: {"factory": f, "params": p, "group": g} for k, (f, p, g) in M.items()}
