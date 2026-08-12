from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

from labels_residuals import residualize_cv_split


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    mae_abs: float
    r2_abs: float
    spearman_within: float

@dataclass
class RunSummary:
    mae_abs_mean: float
    r2_abs_mean: float
    spearman_within_macro: float
    per_fold: List[FoldResult]

def _spearman_within_gene(y_true: pd.Series, y_pred: pd.Series, genes: pd.Series) -> float:
    """
    Compute Spearman within each gene then macro-average across genes with >=2 points.
    """
    vals = []
    df = pd.DataFrame({"y": y_true, "p": y_pred, "g": genes})
    for g, sub in df.groupby("g"):
        if len(sub) >= 2:
            vals.append(spearmanr(sub["y"].values, sub["p"].values).correlation)
    if len(vals) == 0:
        return np.nan
    return float(np.nanmean(vals))

# add once near top if not present
def _safe_nanmean(x):
    if x is None:
        return float("nan")
    x = [v for v in x if v == v]  # drop NaNs
    return float(np.mean(x)) if len(x) else float("nan")

@dataclass
class ResidSummary:
    mae_abs_mean: float
    r2_abs_mean: float
    spearman_within_macro: float
    per_fold: list
    
def fit_ridge_residualized_cv(
    X: pd.DataFrame,
    y: pd.Series,
    genes: pd.Series,
    families: Optional[pd.Series],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    alpha: float = 1.0,
) -> RunSummary:
    """
    Phase 8a core: ridge on *residualized* labels per training fold.
    """
    per_fold: List[FoldResult] = []

    y = pd.Series(y).astype(float)
    genes = pd.Series(genes).astype(str)
    families = None if families is None else pd.Series(families).astype(str)

    for i, (tr_idx, te_idx) in enumerate(folds):
        # Residualization using training-only stats
        y_tr_resid, test_add_back, grp_means, fam_means, gmean = residualize_cv_split(
            y, genes, families, tr_idx, te_idx
        )

        # Train model on residuals
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
        model.fit(X.iloc[tr_idx], y_tr_resid)

        # Predict residuals for test
        y_te_resid_hat = model.predict(X.iloc[te_idx])
        # Recenter for absolute KD
        y_te_abs_hat = y_te_resid_hat + test_add_back

        # Metrics
        y_te_true = y.iloc[te_idx].values
        mae_abs = mean_absolute_error(y_te_true, y_te_abs_hat)
        r2_abs = r2_score(y_te_true, y_te_abs_hat)
        spearman_w = _spearman_within_gene(
            pd.Series(y_te_true), pd.Series(y_te_abs_hat), genes.iloc[te_idx]
        )

        per_fold.append(FoldResult(
            fold=i, n_train=len(tr_idx), n_test=len(te_idx),
            mae_abs=mae_abs, r2_abs=r2_abs, spearman_within=spearman_w
        ))

    # --- Aggregate safely ---
    def _safe_nanmean(vals):
        # Drop NaNs, return nan if nothing valid
        vals = [v for v in vals if v == v]   # keep only finite
        return float(np.mean(vals)) if vals else float("nan")

    maes   = [f.mae_abs for f in per_fold]
    r2s    = [f.r2_abs for f in per_fold]
    spear  = [f.spearman_within for f in per_fold]

    mae_mean   = _safe_nanmean(maes)
    r2_mean    = _safe_nanmean(r2s)
    sp_macro   = _safe_nanmean(spear)

    return RunSummary(
        mae_abs_mean=mae_mean,
        r2_abs_mean=r2_mean,
        spearman_within_macro=sp_macro,
        per_fold=per_fold
    )

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression

@dataclass
class PairwiseFoldResult:
    acc: float
    n_pairs: int

@dataclass
class PairwiseSummary:
    pair_acc_mean: float
    per_fold: list

def fit_pairwise_logit_cv(X, y, genes, folds, C=1.0, max_iter=200):
    """
    X, y are the original (per-sample) design/labels.
    For each fold, build pairwise examples from the TRAIN set only,
    fit logistic regression, evaluate pairwise accuracy on the TEST set pairs.
    """
    from utils_rank import build_pairwise_diffs  # local import to avoid cycles

    per_fold, accs = [], []

    for train_idx, test_idx in folds:
        # Build pairwise training set from train indices only
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        g_tr = genes.iloc[train_idx] if hasattr(genes, "iloc") else genes[train_idx]

        out = build_pairwise_diffs(X_tr, y_tr, g_tr)
        if out[0] is None:
            per_fold.append(PairwiseFoldResult(acc=np.nan, n_pairs=0))
            accs.append(np.nan)
            continue
        X_pair_tr, y_pair_tr, _ = out

        # Fit a small logistic model
        clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=max_iter)
        clf.fit(X_pair_tr, (y_pair_tr > 0).astype(int))  # {0,1} targets

        # Evaluate on TEST set: build pairs within test genes only
        X_te = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        y_te = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]
        g_te = genes.iloc[test_idx] if hasattr(genes, "iloc") else genes[test_idx]

        out_te = build_pairwise_diffs(X_te, y_te, g_te)
        if out_te[0] is None:
            per_fold.append(PairwiseFoldResult(acc=np.nan, n_pairs=0))
            accs.append(np.nan)
            continue

        X_pair_te, y_pair_te, _ = out_te
        yhat_prob = clf.predict_proba(X_pair_te)[:, 1]
        yhat = (yhat_prob >= 0.5).astype(int)
        acc = (yhat == (y_pair_te > 0).astype(int)).mean()

        per_fold.append(PairwiseFoldResult(acc=float(acc), n_pairs=int(len(y_pair_te))))
        accs.append(float(acc))

    # Mean over non-nan folds
    accs_valid = [a for a in accs if a == a]
    pair_acc_mean = float(np.mean(accs_valid)) if accs_valid else float("nan")
    return PairwiseSummary(pair_acc_mean=pair_acc_mean, per_fold=per_fold)
