# Master_model/src/models.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression


def make_ridge_pipeline(ridge_alpha: float = 1.0) -> Pipeline:
    """
    Ridge regression pipeline with:
      - median imputation
      - variance threshold
      - scaling
      - ridge regression
    Matches the pattern you used in relative LOOCV/LOGO.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("varthresh", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler(with_mean=False)),
        ("ridge", Ridge(alpha=ridge_alpha, random_state=42)),
    ])


def fold_center_by_gene(y, groups, tr_mask):
    """
    Center y by gene on the *training* fold mean (no z-scoring).
      - y: array-like of KD values (0..1)
      - groups: array-like of gene IDs
      - tr_mask: boolean mask for training samples
    Returns: centered y (np.ndarray)
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups, dtype=object)
    tr_mask = np.asarray(tr_mask, dtype=bool)

    y_rel = np.empty_like(y, dtype=float)
    global_train_mean = float(np.nanmean(y[tr_mask]))

    gene_mean = {}
    for g in np.unique(groups):
        idx_g_tr = (groups == g) & tr_mask
        gene_mean[g] = float(np.nanmean(y[idx_g_tr])) if idx_g_tr.any() else global_train_mean

    for g in np.unique(groups):
        y_rel[groups == g] = y[groups == g] - gene_mean[g]

    return y_rel


def add_partial_pooling_X(df: pd.DataFrame, group_col: str) -> np.ndarray:
    """
    Random-intercept approximation via ridge-penalized group dummies.
    Returns an (n, k) matrix of one-hot group indicators (no drop-first).
    If the column is missing, returns an (n, 0) empty matrix.
    """
    if group_col not in df.columns:
        return np.zeros((len(df), 0), dtype=float)
    dummies = pd.get_dummies(df[group_col], drop_first=False)
    return dummies.values


# ---- Optional helpers (kept tiny for compatibility) ----

def fit_monotone(score: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """
    Monotone calibrator (increasing). Useful if you want a constrained
    relationship between a scalar score and KD.
    """
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    ir.fit(np.asarray(score, float), np.asarray(y, float))
    return ir


def fit_ridge(X, y, alpha: float = 1.0, sample_weight=None) -> Ridge:
    """
    Plain Ridge fit on y as provided (compatibility shim).
    Prefer train_eval.fit_ridge_logit for the KD pipeline (logit(KD)).
    """
    m = Ridge(alpha=alpha, random_state=42)
    m.fit(X, y, sample_weight=sample_weight)
    return m
