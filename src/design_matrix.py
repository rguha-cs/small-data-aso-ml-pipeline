'''
Functions:

    choose_and_build_design(feat_path, model_dir, relative, verbose) -> (F, X_design, y, feature_order, core_feats)
    Reads feat_path → does your numeric coercions for neg_dG_bind, p_unpaired_site, logit_punp, duplex_dG, neg_duplex_dG, PC
    Builds base_feats, coverage checks, feature tiers, lock feature_cols
    Standardize (mu/sd), save mu.json, sd.json, feature_order.json
    If not relative, add target_* dummies
    returns everything you use later
    
'''

# Master_model/src/design_matrix.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np
import json

def _logit(p: pd.Series, eps: float = 1e-6) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce")
    p = p.clip(eps, 1 - eps)
    return np.log(p / (1 - p))

def choose_and_build_design(
    feat_path: Path | str,
    model_dir: Path | str,
    relative: bool,
    feature_set: str | None = None,   # <— NEW
    use_gene_z: bool = True,          # <— NEW (for P9 z-scores)
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str], List[str]]:
    """
    Step 4: Design matrix & scalers
    Mirrors your notebook block (4a–4k):
      - numeric coercions
      - create/repair neg_duplex_dG
      - base_feats and coverage checks
      - tiered feature selection (with n_keep >= 2)
      - add logit(access) and drop raw p
      - z-score with saved mu/sd; add target dummies only if not relative
      - save mu.json, sd.json, feature_order.json

    Returns:
      F            : cleaned features DataFrame (post mask)
      X_design     : final design matrix (z-scored + dummies if not relative)
      y            : KD vector (numpy float array)
      feature_order: list[str] order of X_design columns
      core_feats   : list[str] X_design columns that are NOT target dummies
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- load features
    F = pd.read_csv(feat_path).copy()
    TARGET_COL = "KD"

    # 4a) numeric coercions (exact set you used)
    numeric_cols = ["neg_dG_bind", "p_unpaired_site", "logit_punp", "duplex_dG", "neg_duplex_dG", "PC"]
    for c in numeric_cols:
        if c in F.columns:
            F[c] = pd.to_numeric(F[c], errors="coerce")

    # 4b) ensure positive-oriented duplex feature
    if "duplex_dG" in F.columns and "neg_duplex_dG" not in F.columns:
        F["neg_duplex_dG"] = -pd.to_numeric(F["duplex_dG"], errors="coerce")

    # --- target present?
    if TARGET_COL not in F.columns:
        raise KeyError("Training: 'KD' missing in features CSV (merge it in Step-2).")
    F[TARGET_COL] = pd.to_numeric(F[TARGET_COL], errors="coerce")
    
    # >>> P9 path: bail out early if requested <<<
    if (feature_set or "").lower() == "p9_context":
        if verbose:
            print("[train][P9] using p9_context design (context + z-scores)")
        # Build P9 matrix
        X_raw, p9_cols, F = build_design_p9(F, use_gene_z=use_gene_z)

        # Standardize (mu/sd) same as your normal path
        mu = X_raw.mean(numeric_only=True)
        sd = X_raw.std(numeric_only=True).replace(0, 1)
        Xs = (X_raw - mu) / sd

        # Add target dummies only if NOT relative
        if relative:
            dummies = pd.DataFrame(index=Xs.index)
        else:
            dummies = (pd.get_dummies(F["target"], prefix="target", drop_first=True)
                       if "target" in F.columns else pd.DataFrame(index=Xs.index))

        X_design = pd.concat([Xs, dummies], axis=1)

        # y and persistence
        y = pd.to_numeric(F[TARGET_COL], errors="coerce").to_numpy(dtype=float)
        (Path(model_dir) / "mu.json").write_text(mu.to_json())
        (Path(model_dir) / "sd.json").write_text(sd.to_json())
        feature_order = X_design.columns.tolist()
        with open(Path(model_dir) / "feature_order.json", "w") as f:
            json.dump(feature_order, f, indent=2)

        core_feats = [c for c in X_design.columns if not c.startswith("target_")]
        return F, X_design, y, feature_order, core_feats

    # 4c) base feature set (exact columns, keep order)
    base_feats = [c for c in ["neg_dG_bind", "logit_punp", "p_unpaired_site", "neg_duplex_dG", "PC"] if c in F.columns]
    # If logit exists, drop raw p to avoid duplicate information
    if "logit_punp" in base_feats and "p_unpaired_site" in base_feats:
        base_feats = [c for c in base_feats if c != "p_unpaired_site"]

    if not base_feats:
        raise KeyError("No usable feature columns found in features CSV.")

    # visibility
    if verbose:
        nonnull = F[ [c for c in base_feats if c in F.columns] + [TARGET_COL] ].notna().sum().to_dict()
        print("[train] non-null counts:", {k:int(v) for k,v in nonnull.items()})

    # 4d) coverage checks
    cov_logit  = float(F["logit_punp"].notna().mean())    if "logit_punp"    in F.columns else 0.0
    cov_duplex = float(F["neg_duplex_dG"].notna().mean()) if "neg_duplex_dG" in F.columns else 0.0
    if verbose:
        print("[train] coverage:", {"logit_punp": round(cov_logit,3), "neg_duplex_dG": round(cov_duplex,3)})

    # 4e) FORCE rich set when coverage is good; otherwise fall back to tier search
    chosen_feats = None
    if cov_logit >= 0.90 and cov_duplex >= 0.90:
        desired = [c for c in ["neg_dG_bind", "logit_punp", "neg_duplex_dG", "PC"] if c in F.columns]
        mask = F[desired + [TARGET_COL]].notna().all(axis=1)
        n_keep = int(mask.sum())
        if n_keep >= 2:
            chosen_feats = desired
            F = F.loc[mask].reset_index(drop=True)
            if verbose:
                print(f"[train] using features (forced): {chosen_feats} (n={n_keep})")

    # Fallback tiers (exact order you used)
    if chosen_feats is None:
        feature_tiers = [
            [c for c in ["neg_dG_bind", "logit_punp", "neg_duplex_dG", "PC"] if c in F.columns],  # desired first
            [f for f in base_feats if f != "p_unpaired_site"],                                     # without raw p
            base_feats,
            [f for f in base_feats if f in {"neg_dG_bind", "PC"}],
            ["neg_dG_bind"] if "neg_dG_bind" in F.columns else [],
        ]
        for feats_try in feature_tiers:
            if not feats_try:
                continue
            mask = F[feats_try + [TARGET_COL]].notna().all(axis=1)
            n_keep = int(mask.sum())
            if n_keep >= 2:
                chosen_feats = feats_try
                F = F.loc[mask].reset_index(drop=True)
                if verbose:
                    print(f"[train] using features: {chosen_feats} (n={n_keep})")
                break

    if chosen_feats is None:
        raise ValueError(
            f"[train][fatal] Not enough samples after dropping NaNs with any feature tier. "
            f"Check KD and feature computation. (KD non-null={int(F[TARGET_COL].notna().sum())})"
        )

    # --- ADD THIS: include k-mer columns present in the snapshot ---
    kmer_cols = [c for c in F.columns if c.startswith(
        ("kmer5_PC", "kmer6_PC", "kmer_pca", "kmer_raw")
    )]
    if kmer_cols:
        # keep k-mers in the same frame we pass forward
        chosen_feats = chosen_feats + kmer_cols
        if verbose:
            print(f"[train] detected k-mer columns: {len(kmer_cols)} (e.g., {kmer_cols[:5]})")

    # 4f) Build X from chosen features
    X = F[chosen_feats].copy()

    # 4g) Add logit(access) if present, then drop raw p to keep the design clean
    if "p_unpaired_site" in X.columns:
        X["logit_punp"] = _logit(X["p_unpaired_site"])
        X = X.drop(columns=["p_unpaired_site"])
        chosen_feats = [f for f in chosen_feats if f != "p_unpaired_site"] + ["logit_punp"]

    # 4h) Lock feature_cols and standardize
    feature_cols = chosen_feats[:]  # locked
    if verbose:
        print("[train][locked] using features:", feature_cols)

    mu = X.mean(numeric_only=True)
    sd = X.std(numeric_only=True).replace(0, 1)
    Xs = (X - mu) / sd

    # 4i) Add target dummies only if NOT relative
    if relative:
        dummies = pd.DataFrame(index=Xs.index)
    else:
        dummies = (pd.get_dummies(F["target"], prefix="target", drop_first=True)
                   if "target" in F.columns else pd.DataFrame(index=Xs.index))

    X_design = pd.concat([Xs, dummies], axis=1)

    # 4j) y (and logit(y) if caller wants to compute later)
    y = pd.to_numeric(F[TARGET_COL], errors="coerce").to_numpy(dtype=float)

    # 4k) Persist scalers & feature order
    (Path(model_dir) / "mu.json").write_text(mu.to_json())
    (Path(model_dir) / "sd.json").write_text(sd.to_json())
    feature_order = X_design.columns.tolist()
    with open(Path(model_dir) / "feature_order.json", "w") as f:
        json.dump(feature_order, f, indent=2)

    # core features = columns that are NOT target dummies
    core_feats = [c for c in X_design.columns if not c.startswith("target_")]

    return F, X_design, y, feature_order, core_feats

# --- Add to src/design_matrix.py ---
import numpy as np, pandas as pd

P9_BASE_COLS = [
    "neg_dG_bind", "neg_duplex_dG", "logit_punp",
    "dG_open_mean", "dG_open_min", "neg_dG_eff",
    "aso_gc_frac", "target_gc_pm10", "target_gc_pm30",
]

def _add_gene_zscores(F: pd.DataFrame, cols, gene_col="target"):
    if gene_col not in F.columns:
        return F
    G = F.copy()
    gb = G.groupby(gene_col)
    for c in cols:
        if c in G.columns:
            mu = gb[c].transform("mean"); sd = gb[c].transform("std").replace(0, np.nan)
            G[f"zs_{c}"] = (G[c] - mu) / sd
    return G

# Extend existing builder
def build_design_p9(F_raw: pd.DataFrame, use_gene_z=True, z_cols=None):
    F = F_raw.copy()
    # Ensure all requested columns exist (fill missing with 0)
    for c in P9_BASE_COLS:
        if c not in F.columns:
            F[c] = 0.0
    X = F[P9_BASE_COLS].copy()

    # Optional z‑scores within gene
    if use_gene_z:
        zcols = z_cols or ["neg_dG_bind","neg_duplex_dG","logit_punp","neg_dG_eff"]
        Fz = _add_gene_zscores(F, zcols, gene_col=("target" if "target" in F.columns else "gene"))
        for c in zcols:
            zc = f"zs_{c}"
            if zc in Fz.columns:
                X[zc] = Fz[zc]
                
    feature_cols = list(X.columns)
    return X, feature_cols, F # F is the full frame for downstream bookkeeping