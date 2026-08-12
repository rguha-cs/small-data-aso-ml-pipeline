from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Iterable, Optional


def _safe_group_means(y: pd.Series, groups: pd.Series) -> Dict[str, float]:
    g = {}
    for k, sub in y.groupby(groups):
        if len(sub) == 0:
            continue
        g[str(k)] = float(np.mean(sub.values))
    return g


def _vectorize_means(groups: Iterable, means: Dict[str, float], default: float) -> np.ndarray:
    out = np.empty(len(groups), dtype=float)
    for i, g in enumerate(groups):
        out[i] = means.get(str(g), default)
    return out


def residualize_by_group(
    y: pd.Series,
    group: pd.Series,
    family: Optional[pd.Series] = None,
) -> Tuple[pd.Series, Dict[str, float], Optional[Dict[str, float]], float]:
    """
    Residualize y by *group* means. Optionally also compute family means.
    Returns (residuals, group_means, family_means_or_None, global_mean).
    """
    y = pd.Series(y).astype(float)
    group = pd.Series(group).astype(str)
    grp_means = _safe_group_means(y, group)
    resid = y.values - _vectorize_means(group, grp_means, default=float(y.mean()))
    fam_means = None
    if family is not None:
        fam_means = _safe_group_means(y, pd.Series(family).astype(str))
    return pd.Series(resid, index=y.index), grp_means, fam_means, float(y.mean())


def residualize_cv_split(
    y: pd.Series,
    group: pd.Series,
    family: Optional[pd.Series],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Optional[Dict[str, float]], float]:
    """
    Residualize using statistics computed *on the training fold only*.
    Returns:
      y_train_resid, test_add_back, train_group_means, train_family_means, train_global_mean
    """
    y = pd.Series(y).astype(float)
    group = pd.Series(group).astype(str)
    fam = None if family is None else pd.Series(family).astype(str)

    y_tr = y.iloc[train_idx]; y_te = y.iloc[test_idx]
    g_tr = group.iloc[train_idx]; g_te = group.iloc[test_idx]
    f_tr = None if fam is None else fam.iloc[train_idx]
    f_te = None if fam is None else fam.iloc[test_idx]

    # Training stats
    y_tr_resid, grp_means, fam_means, gmean = residualize_by_group(y_tr, g_tr, f_tr)

    # Test recentering
    grp_vec = _vectorize_means(g_te, grp_means, default=np.nan)
    if fam is not None:
        fam_vec = _vectorize_means(f_te, fam_means or {}, default=np.nan)
    else:
        fam_vec = np.full_like(grp_vec, np.nan, dtype=float)

    # Choose add-back mean
    add_back = np.where(np.isfinite(grp_vec), grp_vec,
                 np.where(np.isfinite(fam_vec), fam_vec, gmean))

    return y_tr_resid.values, add_back, grp_means, fam_means, gmean
