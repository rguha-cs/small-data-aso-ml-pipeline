'''
Function:

    make_synthetic_training(F, X_design, y, model_dir, feature_order, relative, ridge_alpha, synth_n_per_real, synth_weight, noise_sigma0, noise_c1, noise_c2) -> (X_train, y_train, w_train)
    Recreates the exact feature-jitter DataFrame S and noise model you coded
    Builds X_synth with the saved mu/sd and same feature_order
    Calibrates KD* from a ridge on X_design→logit(y) (same as your ridge_cal)
    Draws KD_synth with noise; concatenates to real; builds w_train with synth_weight

'''

# Master_model/src/synthetic.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import json

# ---- helpers kept local (identical math to your notebook) ----
def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, float)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return 1.0 / (1.0 + np.exp(-z))

# ----------------------------------------------------------------
# Public API: Step-5 Synthetic generator (feature-space replicas)
# ----------------------------------------------------------------
def make_synthetic_training(
    F: pd.DataFrame,                  # real features table AFTER Step-4 masking
    X_design: pd.DataFrame,           # real design matrix (z-scored + dummies if not relative)
    y: np.ndarray,                    # real KD (0..1) aligned with X_design
    model_dir: Path,                  # where mu.json/sd.json live
    feature_order: list[str],         # order of columns in X_design to enforce
    relative: bool,                   # if True, do NOT add target dummies for synth
    ridge_alpha: float,               # same as your main model alpha
    synth_n_per_real: int = 100,      # replicas per real sample
    synth_weight: float = 0.3,        # per-sample weight for synthetic points
    noise_sigma0: float = 0.03,       # base noise
    noise_c1: float = 0.07,           # + (1 - p_unpaired)
    noise_c2: float = 0.05,           # + OT_weighted
    seed: int = 42,                   # reproducible RNG
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns:
      X_train : pd.DataFrame (real then synthetic), column order == feature_order
      y_train : np.ndarray (KD for real then KD_synth)
      w_train : np.ndarray (weights: 1 for real, synth_weight for synthetic)

    Behavior matches your Step-5 notebook block:
      - jitter neg_dG_bind, p_unpaired_site, OT_weighted, GC_pen, PC
      - build X_synth with logit_punp (drop raw p)
      - z-score with saved mu/sd; add target dummies only if NOT relative
      - calibrate KD* from ridge on real (logit KD)
      - draw KD_synth with heteroscedastic noise sigma0 + c1*(1-p) + c2*OT
      - concat real + synth and return weights
    """
    rng = np.random.default_rng(seed)

    # ---------- 1) Build synthetic feature DataFrame S by jittering F ----------
    # Columns expected by your generator:
    need_cols = {
        "target": None,          # used only for dummies when relative=False
        "neg_dG_bind": 0.0,
        "p_unpaired_site": 0.5,
        "OT_weighted": 0.0,
        "GC_pen": 0.0,
        "PC": 0.5,
    }
    S_base = pd.DataFrame(index=F.index)
    for c, default in need_cols.items():
        if c in F.columns:
            S_base[c] = pd.to_numeric(F[c], errors="coerce")
        else:
            S_base[c] = default
    S_base["target"] = F.get("target", S_base["target"])

    rows = []
    # Jitter scales mirror your notebook (mean-centered noise)
    for _, r in S_base.iterrows():
        if pd.isna(r.get("PC")) and pd.isna(r.get("p_unpaired_site")) and pd.isna(r.get("neg_dG_bind")):
            continue  # skip pathological row
        for _ in range(int(synth_n_per_real)):
            rows.append({
                "target":        r["target"],
                "neg_dG_bind":   float(r["neg_dG_bind"] + rng.normal(0, 0.6)),
                "p_unpaired_site": float(np.clip((r["p_unpaired_site"] if pd.notna(r["p_unpaired_site"]) else 0.5)
                                                 + rng.normal(0, 0.08), 0.01, 0.99)),
                "OT_weighted":   float(max(0.0, (r["OT_weighted"] if pd.notna(r["OT_weighted"]) else 0.0)
                                                 + rng.normal(0, 0.15))),
                "GC_pen":        float(max(0.0, (r["GC_pen"] if pd.notna(r["GC_pen"]) else 0.0)
                                                 + rng.normal(0, 0.01))),
                "PC":            float(np.clip((r["PC"] if pd.notna(r["PC"]) else 0.5)
                                                 + rng.normal(0, 0.05), 0.0, 1.0)),
            })
    S = pd.DataFrame(rows)

    # ---------- 2) Build X_synth in the SAME way as real X_design ----------
    # (a) add logit_punp and drop raw p
    S["p_unpaired_site"] = pd.to_numeric(S.get("p_unpaired_site", 0.5), errors="coerce").fillna(0.5)
    S["logit_punp"] = _logit(S["p_unpaired_site"].to_numpy())
    Xs2 = S[["neg_dG_bind", "logit_punp", "OT_weighted", "GC_pen", "PC"]].copy()

    # (b) z-score with saved mu/sd
    mu = pd.read_json(Path(model_dir) / "mu.json", typ="series")
    sd = pd.read_json(Path(model_dir) / "sd.json", typ="series").replace(0, 1)
    # Align to mu/sd index (fill missing with 0 then scale safely)
    for col in Xs2.columns:
        if col not in mu.index:
            # column not used in real design; keep it but center/scale with zeros/ones
            mu[col] = 0.0
            sd[col] = 1.0
    mu = mu.reindex(Xs2.columns).fillna(0.0)
    sd = sd.reindex(Xs2.columns).replace(0, 1).fillna(1.0)
    Xs2 = (Xs2 - mu) / sd

    # (c) target dummies only if NOT relative
    if relative:
        dumm2 = pd.DataFrame(index=Xs2.index)
    else:
        dumm2 = pd.get_dummies(S["target"], prefix="target", drop_first=True)

    X_synth = pd.concat([Xs2, dumm2], axis=1)

    # (d) Enforce the exact feature order
    #     Missing columns -> fill 0; extra columns -> drop them
    X_synth = X_synth.reindex(columns=feature_order, fill_value=0.0)

    # ---------- 3) Calibrate KD* from real (ridge on logit(KD)) ----------
    ridge_cal = Ridge(alpha=ridge_alpha, random_state=42)
    logit_y_real = _logit(y)  # y is 0..1
    ridge_cal.fit(X_design, logit_y_real)
    KD_star = _sigmoid(ridge_cal.predict(X_synth))

    # ---------- 4) Heteroscedastic noise model (exact params) ----------
    # sigma = sigma0 + c1*(1 - p_unpaired) + c2*OT_weighted
    # NaN-safe defaults already applied above
    sigma = (float(noise_sigma0)
             + float(noise_c1) * (1.0 - S["p_unpaired_site"].to_numpy())
             + float(noise_c2) * (S["OT_weighted"].to_numpy()))
    # guard against any stray NaNs
    sigma = np.where(np.isfinite(sigma), sigma, float(noise_sigma0))

    KD_synth = np.clip(KD_star + rng.normal(0.0, sigma), 0.0, 1.0)

    # ---------- 5) Concatenate real + synthetic and build weights ----------
    X_train = pd.concat([X_design, X_synth], axis=0, ignore_index=True)
    y_train = np.concatenate([y, KD_synth])
    w_train = np.concatenate([np.ones(len(y), dtype=float),
                              np.full(len(KD_synth), float(synth_weight), dtype=float)])

    return X_train, y_train, w_train
