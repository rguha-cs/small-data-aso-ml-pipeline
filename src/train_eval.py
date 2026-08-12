'''
train_eval.py: Steps “6 Fit&LOOCV”, “7 Relative eval”, “6 Optional ablation”, “7 Optional LOGO”, “8 Viz”

Functions:

    fit_ridge_logit(X_train, y_train, w_train, ridge_alpha) -> sklearn_model (your ridge fit)
    loocv_metrics_real_only(model_dir, sim_mode, X_train, y_train, w_train, y_real_len, verbose) -> (preds, metrics_path, coef_path)
    Reproduce your LOOCV loop over first len(y_real) rows, save metrics & coefs
    relative_loocv_logo(F, model_dir, ridge_alpha, verbose) -> (rel_logo_csv_path or None)
    Move your _fold_center_by_gene_safe, make_ridge_pipeline, and both REL-LOOCV and REL-LOGO sections here
    run_ablation_and_logo(F, X_design, core_feats, ridge_alpha, run_ablation, run_logo, model_dir, fig_dir, mode_tag, relative, verbose) -> (ablation_csv_path or None, logo_csv_path or None)
    Move “Optional: Ablation” and “Optional: LOGO Spearman (by target)” here
    save_calibration_plot(y, preds, fig_dir, sim_mode, relative) -> calib_path (your matplotlib block)

'''

# Master_model/src/train_eval.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Dict

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr


# -----------------------------
# small numeric helpers
# -----------------------------
def _logit(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = np.asarray(y, float)
    y = np.clip(y, eps, 1 - eps)
    return np.log(y / (1 - y))

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return 1.0 / (1.0 + np.exp(-z))

def _spearman_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(spearmanr(y_true, y_pred).correlation)

def _ensure_schema(X, feature_order):
    """
    Reindex X to exactly the feature_order used at fit-time.
    Fills missing cols with 0.0, drops extras, and preserves order.
    Works for both DataFrames and arrays; returns DataFrame.
    """
    if not isinstance(X, pd.DataFrame):
        # In case upstream already handed arrays; turn back into DF to reindex safely
        X = pd.DataFrame(X, columns=feature_order)
    return X.reindex(columns=feature_order, fill_value=0.0)


# ============================================================
# 1) Fit the global ridge on logit(KD) with (optional) weights
# ============================================================
def fit_ridge_logit(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: Optional[np.ndarray],
    ridge_alpha: float,
) -> Ridge:
    model = Ridge(alpha=ridge_alpha, random_state=42)
    logit_y = _logit(y_train)
    model.fit(X_train, logit_y, sample_weight=None if w_train is None else w_train)
    return model


# ========================================================================
# 2) LOOCV on REAL rows ONLY (first y_real_len rows), save metrics & coefs
# ========================================================================
def loocv_metrics_real_only(
    model_dir: Path,
    sim_mode: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: Optional[np.ndarray],
    y_real_len: int,
    ridge_alpha: float,
    verbose: bool = True,
) -> Tuple[np.ndarray, Path, Path]:
    """
    Mirrors your notebook:
    - full fit (for coefficients dump)
    - LOOCV folds: i in [0, y_real_len) → leave out ONLY that real row; keep all synth rows
    - predict on left-out real row via logistic link
    - save coef_ from the full fit; save LOOCV metrics JSON
    """
    # ---- full fit (for coefficient dump)
    full = Ridge(alpha=ridge_alpha, random_state=42)
    logit_y_full = _logit(y_train)
    
    # lock the in-scope feature order once from X_train
    feature_order_local = list(X_train.columns)
    
    X_full = _ensure_schema(X_train, X_train.columns)
    full.fit(X_full, logit_y_full, sample_weight=w_train)

    coef_df = pd.DataFrame({"feature": X_train.columns, "coef": full.coef_})
    coef_df = pd.concat(
        [coef_df, pd.DataFrame({"feature": ["intercept"], "coef": [full.intercept_]})],
        ignore_index=True,
    )
    coef_path = Path(model_dir) / f"coef_{sim_mode}.csv"
    coef_df.to_csv(coef_path, index=False)

    # ---- LOOCV over real rows
    preds: List[float] = []
    for i in range(y_real_len):
        tr_mask = np.ones(len(X_train), dtype=bool)
        tr_mask[i] = False
        
        m = Ridge(alpha=ridge_alpha, random_state=42)
        logit_y_tr = _logit(y_train[tr_mask])
        
        # reindex BOTH train and test to the same locked order
        X_tr = _ensure_schema(X_train.loc[tr_mask], feature_order_local)
        X_te = _ensure_schema(X_train.iloc[[i]],   feature_order_local)
        
        # DEBUG ON FIRST FOLD ONLY
        if i == 0:
            print("[debug][train_eval] fit cols:", list(X_tr.columns))
            print("[debug][train_eval]  te cols:", list(X_te.columns))

        # fit & predict (use the ensured matrices)
        m.fit(X_tr, logit_y_tr, sample_weight=None if w_train is None else w_train[tr_mask])
        preds.append(float(_sigmoid(m.predict(X_te)[0])))
    
    preds = np.array(preds, dtype=float)
    mae = float(mean_absolute_error(y_train[:y_real_len], preds))
    r2  = float(r2_score(y_train[:y_real_len], preds))
    metrics = {"alpha": ridge_alpha, "LOOCV_MAE": mae, "LOOCV_R2": r2}

    metrics_path = Path(model_dir) / f"metrics_{sim_mode}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"[loocv] MAE={mae:.3f}, R2={r2:.3f} → {metrics_path}")

    return preds, metrics_path, coef_path


# =======================================================================
# 3) Relative evaluation: fold-safe centering + LOOCV and per-target LOGO
# =======================================================================
def _fold_center_by_gene_safe(y: np.ndarray, groups: np.ndarray, tr_mask: np.ndarray) -> np.ndarray:
    """
    Center y by gene on the *training* fold mean only.
    If a gene has no train members, fall back to the global train mean.
    """
    y = np.asarray(y, float)
    groups = np.asarray(groups, object)
    y_rel = np.empty_like(y, dtype=float)

    global_train_mean = float(np.nanmean(y[tr_mask]))
    gene_means: Dict[object, float] = {}
    for g in np.unique(groups):
        idx = (groups == g) & tr_mask
        gene_means[g] = float(np.nanmean(y[idx])) if idx.any() else global_train_mean

    for g in np.unique(groups):
        y_rel[groups == g] = y[groups == g] - gene_means[g]
    return y_rel


def _make_ridge_pipeline(ridge_alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("varthresh", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", Ridge(alpha=ridge_alpha, random_state=42)),
        ]
    )

def relative_loocv_logo(
    F: pd.DataFrame,
    model_dir: Path,
    ridge_alpha: float,
    sim_mode: str,                 # include sim_mode for per-mode filenames
    verbose: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Path]]:
    """
    Implements your Step-7 relative blocks:
      - build X0 from raw features (neg_dG_bind, p_unpaired_site, OT_weighted, GC_pen, PC)
      - add logit(p), drop raw p
      - z-score with saved mu/sd
      - REL-LOOCV using fold-safe centering
      - REL-LOGO per 'target', reporting Spearman and n_test
    Returns: (rel_loocv_df, rel_logo_df, rel_logo_csv_path)
    """
    # Design for relative uses raw feature set (no dummies)
    cols = ["neg_dG_bind", "p_unpaired_site", "OT_weighted", "GC_pen", "PC"]
    present = [c for c in cols if c in F.columns]
    X0 = F[present].copy()
    if "p_unpaired_site" in X0.columns:
        X0["logit_punp"] = _logit(np.asarray(X0["p_unpaired_site"], float))
        X0 = X0.drop(columns=["p_unpaired_site"])

    # z-score with saved mu/sd
    mu = pd.read_json(Path(model_dir) / "mu.json", typ="series")
    sd = pd.read_json(Path(model_dir) / "sd.json", typ="series").replace(0, 1)
    # Align
    for col in X0.columns:
        if col not in mu.index:
            mu[col] = 0.0
            sd[col] = 1.0
    mu = mu.reindex(X0.columns).fillna(0.0)
    sd = sd.reindex(X0.columns).replace(0, 1).fillna(1.0)
    X0 = (X0 - mu) / sd

    y = pd.to_numeric(F["KD"], errors="coerce").to_numpy(dtype=float)
    groups = F["target"].to_numpy(object) if "target" in F.columns else np.array(["ALL"] * len(F), object)

    # Drop ultra-sparse columns (>80% NaN) before CV
    nan_frac = X0.isna().mean()
    to_drop = nan_frac[nan_frac > 0.80].index.tolist()
    if to_drop:
        if verbose:
            print("[relative] Dropping very sparse columns:", to_drop)
        X0 = X0.drop(columns=to_drop)

    # ---- REL-LOOCV
    loo = LeaveOneOut()
    preds_rel: List[float] = []
    trues_rel: List[float] = []
    model = _make_ridge_pipeline(ridge_alpha=ridge_alpha)

    for tr, te in loo.split(X0):
        tr_mask = np.zeros(len(X0), dtype=bool)
        tr_mask[tr] = True

        y_rel = _fold_center_by_gene_safe(y, groups, tr_mask=tr_mask)

        X_train = X0.iloc[tr]
        y_train = y_rel[tr]

        # (optional) drop rows that are almost entirely NaN pre-impute
        row_na_frac = X_train.isna().mean(axis=1)
        keep_rows = row_na_frac <= 0.95
        if not keep_rows.all():
            X_train = X_train.loc[keep_rows]
            y_train = y_train[keep_rows.to_numpy()]

        # >>> BEGIN: enforce per-run feature order & debug prints
        order_path = Path(model_dir) / "feature_order.json"
        if order_path.exists():
            try:
                feature_order = json.loads(order_path.read_text())
                # Align train/test to the saved order (fill any new cols with 0)
                X_train = X_train.reindex(columns=feature_order, fill_value=0.0)
                X_test  = X_test.reindex(columns=feature_order,  fill_value=0.0)
            except Exception as e:
                print("[warn][train_eval] could not read feature_order:", e)

        print("[debug][train_eval] reading order/scaler from:", Path(model_dir))
        print("[debug][train_eval] fit cols:", X_train.columns.tolist()[:12], "… total:", X_train.shape[1])
        print("[debug][train_eval]  te cols:",  X_test.columns.tolist()[:12],  "… total:", X_test.shape[1])
        # >>> END

        model.fit(X_train, y_train)
        x_test = X0.iloc[te]
        if x_test.isna().all(axis=1).iloc[0]:
            preds_rel.append(0.0)  # centered mean fallback
        else:
            preds_rel.append(float(model.predict(x_test)[0]))
        trues_rel.append(float(y_rel[te][0]))

    rel_loocv_df = pd.DataFrame(
        {"y_rel_true": trues_rel, "y_rel_pred": preds_rel}
    )
    if verbose:
        mae = mean_absolute_error(rel_loocv_df["y_rel_true"], rel_loocv_df["y_rel_pred"])
        r2  = r2_score(rel_loocv_df["y_rel_true"], rel_loocv_df["y_rel_pred"])
        sp  = _spearman_safe(rel_loocv_df["y_rel_true"].to_numpy(), rel_loocv_df["y_rel_pred"].to_numpy())
        print(f"[rel-loocv] MAE={mae:.3f}, R2={r2:.3f}, Spearman={sp:.3f}")

    # ---- REL-LOGO
    rows = []
    if "target" in F.columns:
        gkf = GroupKFold(n_splits=len(np.unique(groups)))
        for tr, te in gkf.split(X0, y, groups):
            tgt = str(np.unique(groups[te])[0])
            tr_mask = np.zeros(len(X0), dtype=bool)
            tr_mask[tr] = True
            y_rel = _fold_center_by_gene_safe(y, groups, tr_mask=tr_mask)

            X_train = X0.iloc[tr]
            y_train = y_rel[tr]
            # same defensive drop
            row_na_frac = X_train.isna().mean(axis=1)
            keep_rows = row_na_frac <= 0.95
            if not keep_rows.all():
                X_train = X_train.loc[keep_rows]
                y_train = y_train[keep_rows.to_numpy()]

            model = _make_ridge_pipeline(ridge_alpha=ridge_alpha)
            model.fit(X_train, y_train)
            yhat_rel = model.predict(X0.iloc[te])

            rows.append({
                "heldout_target": tgt,
                "REL_Spearman": _spearman_safe(y_rel[te], yhat_rel),
                "n_test": int(len(te)),
            })
    rel_logo_df = pd.DataFrame(rows) if rows else None

    rel_logo_csv_path = None
    if rel_logo_df is not None and len(rel_logo_df):
        rel_logo_csv_path = Path(model_dir) / f"logo_metrics_relative_{sim_mode}.csv"
        rel_logo_df.to_csv(rel_logo_csv_path, index=False)

    return rel_loocv_df, rel_logo_df, rel_logo_csv_path

# ============================================================
# 4) Ablations and absolute LOGO (by group) on chosen features
# ============================================================
def loocv_mae_r2(
    F: pd.DataFrame,
    feats: List[str],
    ridge_alpha: float,
) -> Tuple[float, float]:
    """
    Simple LOOCV on absolute KD using the selected features.
    (Assumes F already contains the standardized design columns named in feats.)
    """
    X = F[feats].copy()
    y = pd.to_numeric(F["KD"], errors="coerce").to_numpy(dtype=float)

    loo = LeaveOneOut()
    preds = np.zeros(len(X), dtype=float)
    for tr, te in loo.split(X):
        m = Ridge(alpha=ridge_alpha, random_state=42)
        X_tr = _ensure_schema(X.iloc[tr], feats)
        X_te = _ensure_schema(X.iloc[te], feats)
        m.fit(X_tr, _logit(y[tr]))
        preds[te] = _sigmoid(m.predict(X_te))
    return float(mean_absolute_error(y, preds)), float(r2_score(y, preds))


def ablation_runs(core_feats: List[str]) -> Iterable[Tuple[str, List[str]]]:
    """
    Generates (name, drops) pairs.
    Baseline = all core_feats; then drop each feature one-by-one.
    """
    yield ("baseline", [])
    for f in core_feats:
        yield (f"drop_{f}", [f])


def logo_spearman(
    F_with_X: pd.DataFrame,
    feats: List[str],
    ridge_alpha: float,
    group_col: str = "target",
) -> pd.DataFrame:
    """
    Absolute LOGO Spearman by group_col.
    """
    if group_col not in F_with_X.columns:
        return pd.DataFrame(columns=["heldout_target", "Spearman", "n_test"])

    X = F_with_X[feats].copy()
    y = pd.to_numeric(F_with_X["KD"], errors="coerce").to_numpy(dtype=float)
    groups = F_with_X[group_col].astype(str).to_numpy()

    rows = []
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    for tr, te in gkf.split(X, y, groups):
        m = Ridge(alpha=ridge_alpha, random_state=42)
        X_tr = _ensure_schema(X.iloc[tr], feats)
        X_te = _ensure_schema(X.iloc[te], feats)
        m.fit(X_tr, _logit(y[tr]))
        yhat = _sigmoid(m.predict(X_te))
        rows.append({
            "heldout_target": str(np.unique(groups[te])[0]),
            "Spearman": _spearman_safe(y[te], yhat),
            "n_test": int(len(te)),
        })
    return pd.DataFrame(rows)


def run_ablation_and_logo(
    F: pd.DataFrame,
    X_design: pd.DataFrame,
    core_feats: List[str],
    ridge_alpha: float,
    run_ablation: bool,
    run_logo_flag: bool,
    model_dir: Path,
    fig_dir: Path,
    sim_mode: str,           # include sim_mode in filenames
    relative: bool,
    verbose: bool = True,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Runs:
      - ablation LOOCV (absolute) over core_feats (baseline + drop-1)
      - absolute LOGO Spearman (by 'target') on core_feats
    Saves CSVs and returns their paths (or None).
    """
    abl_path = None
    logo_path = None

    # Prepare a frame that contains KD and the chosen design columns
    F_with_X = F.assign(**{c: X_design[c] for c in core_feats if c in X_design.columns})

    if run_ablation:
        rows = []
        for name, drops in ablation_runs(core_feats):
            use_feats = [f for f in core_feats if f not in drops]
            if not use_feats:
                rows.append({"model": name, "LOOCV_MAE": np.nan, "LOOCV_R2": np.nan})
                continue
            mae, r2 = loocv_mae_r2(F_with_X, use_feats, ridge_alpha=ridge_alpha)
            rows.append({"model": name, "LOOCV_MAE": mae, "LOOCV_R2": r2})
        ablation_df = pd.DataFrame(rows)
        suffix = "__rel" if relative else "__abs"
        abl_path = Path(model_dir) / f"ablation_{sim_mode}{suffix}.csv"
        ablation_df.to_csv(abl_path, index=False)
        if verbose:
            print("[ablation] wrote:", abl_path)

    if run_logo_flag and ("target" in F_with_X.columns):
        logo_df = logo_spearman(F_with_X, core_feats, ridge_alpha=ridge_alpha, group_col="target")
        suffix = "__rel" if relative else "__abs"
        logo_path = Path(model_dir) / f"logo_metrics_{sim_mode}{suffix}.csv"
        logo_df.to_csv(logo_path, index=False)
        if verbose:
            print("[logo] wrote:", logo_path)

    return abl_path, logo_path


# =========================================
# 5) Simple calibration plot (observed vs)
# =========================================
def save_calibration_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fig_dir: Path,
    sim_mode: str,
    relative: bool,
) -> Path:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Observed KD")
    plt.ylabel("Predicted KD (LOOCV)")
    plt.title(f"Calibration — mode={sim_mode}, relative={relative}")
    path = fig_dir / f"calibration_{sim_mode}{'__rel' if relative else '__abs'}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

## --------- For capturing data and stats per run --------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def calibration_stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    yp = np.clip(np.asarray(y_pred, float), eps, 1 - eps)
    x = np.log(yp/(1-yp))
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, np.asarray(y_true, float), rcond=None)
    return float(beta[0]), float(beta[1])

def permutation_baseline_spearman(y_true: np.ndarray, n: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); y = np.asarray(y_true, float)
    vals = [float(spearmanr(y, rng.permutation(y)).correlation) for _ in range(n)]
    vals = np.asarray(vals, float)
    return {"perm_spearman_mean": float(np.nanmean(vals)),
            "perm_spearman_p05": float(np.nanpercentile(vals, 5)),
            "perm_spearman_p95": float(np.nanpercentile(vals, 95)),
            "perm_spearman_n": int(n)}

def summarize_logo_csv(logo_df: pd.DataFrame) -> dict:
    out = {}
    if "Spearman" in logo_df.columns: out["logo_macro_spearman"] = float(logo_df["Spearman"].mean())
    if "MAE" in logo_df.columns:      out["logo_macro_mae"]      = float(logo_df["MAE"].mean())
    if "REL_Spearman" in logo_df.columns:
        out["logo_macro_rel_spearman"] = float(logo_df["REL_Spearman"].mean())
    return out

def residualize_within_group(y: np.ndarray, group_ids):
    """
    y: shape (n,)
    group_ids: array-like length n (e.g., gene IDs)
    Returns y_resid and a dict of group -> mean used, so you can add back later if needed.
    """
    s = pd.Series(y)
    g = pd.Series(group_ids)
    group_means = s.groupby(g).transform("mean")
    y_resid = s - group_means
    return y_resid.values.astype(float), dict(s.groupby(g).mean())

# ============== Phase 9 Helper Functions =======================
# --- Add to src/train_eval.py ---
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

def calibrate_by_group(scores: np.ndarray, y: np.ndarray, group: pd.Series, kind: str = "isotonic"):
    kind = (kind or "linear").lower()
    group = pd.Series(group).astype(str).reset_index(drop=True)
    models = {}
    for g, idx in group.groupby(group).groups.items():
        idx = np.array(sorted(list(idx)))
        if len(idx) < 3:
            continue # too small; will back off to global
        if kind == "isotonic":
            m = IsotonicRegression(out_of_bounds="clip")
        else:
            m = LinearRegression()
        m.fit(scores[idx].reshape(-1,1), y[idx])
        models[g] = m

    # global fallback
    if kind == "isotonic":
        global_model = IsotonicRegression(out_of_bounds="clip").fit(scores.reshape(-1,1), y)
    else:
        global_model = LinearRegression().fit(scores.reshape(-1,1), y)

    def predict(scores_new: np.ndarray, group_new: pd.Series):
        group_new = pd.Series(group_new).astype(str).reset_index(drop=True)
        yhat = np.empty_like(scores_new, dtype=float)
        for g, idx in group_new.groupby(group_new).groups.items():
            model = models.get(g, global_model)
            yhat[np.array(list(idx))] = model.predict(scores_new[np.array(list(idx))].reshape(-1,1))
        return yhat

    return models, global_model, predict