'''
thermo.py (Step 3: Biophysics upgrades” for enhanced/synthetic)
Function:

    thermo_upgrade(feat_path, merged, aso_col, targ_col, feat_dir, model_dir, features_dir, full_bio, thermo_cache, force_recompute_thermo, verbose) -> (merged_updated, full_thermo_csv_path)
    explicit overrides (THERMO_ASO_COL_OVERRIDE, THERMO_TARGET_COL_OVERRIDE)
    signature + cache handling with _load_thermo_cache / _save_thermo_cache
    Vienna-only enforcement (full_bio)
    id-aware merge path (your special case)
    writing both *_with_thermo.csv and the gold snapshot named by sig
    guardrails/warnings and feature list update (returned so the runner can keep order)

'''

# Master_model/src/thermo.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd, numpy as np, json, hashlib, os, time

from thermo_backends import ThermoComputer, make_thermo

# -------------------------------------------------------------------
# Manual overrides (same semantics as your notebook Section 3)
# -------------------------------------------------------------------
THERMO_ASO_COL_OVERRIDE: Optional[str] = None   # e.g., "Sequence"
THERMO_TARGET_COL_OVERRIDE: Optional[str] = None  # e.g., "target_window"

# -------------------------------------------------------------------
# Helpers: signature + cache IO (pairs JSONL + META JSON)
# -------------------------------------------------------------------
def _dataset_signature(df: pd.DataFrame, aso_col: str, targ_col: str) -> str:
    key = (
        df[[aso_col, targ_col]]
        .astype(str)
        .apply(lambda s: s.str.upper().str.replace("T", "U", regex=False))
        .agg("|".join, axis=1)
    )
    return hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:16]

def _cache_pair(thermo_cache_dir: Path, sig: str) -> tuple[Path, Path]:
    thermo_cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = thermo_cache_dir / f"thermo_{sig}.jsonl"
    meta_path = thermo_cache_dir / f"thermo_{sig}.meta.json"
    return data_path, meta_path

def _load_thermo_cache(sig: str, thermo_cache_dir: Path) -> tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    data_path, meta_path = _cache_pair(thermo_cache_dir, sig)
    if not data_path.exists():
        return None, {}
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    return df, meta

def _save_thermo_cache(df_keep: pd.DataFrame, sig: str, thermo_cache_dir: Path,
                       aso_col: str, targ_col: str,
                       vienna_only: bool, params: Dict[str, Any]) -> None:
    data_path, meta_path = _cache_pair(thermo_cache_dir, sig)
    with data_path.open("w", encoding="utf-8") as f:
        for _, r in df_keep.iterrows():
            f.write(json.dumps({k: (None if pd.isna(v) else v) for k, v in r.items()}) + "\n")
    meta = {
        "vienna_only": bool(vienna_only),
        "params": dict(params or {}),
        "aso_col": aso_col,
        "targ_col": targ_col,
        "sig": sig,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

# -------------------------------------------------------------------
# Low-level thermo (unchanged behavior): compute ΔG and neg_dG_bind
# -------------------------------------------------------------------
def compute_thermo_features(
    df_pairs: pd.DataFrame,
    aso_col: str = "Sequence",
    targ_col: str = "target_window",
    cache_dir: Path = Path("thermo_cache"),
    thermo: Optional[ThermoComputer] = None
) -> pd.DataFrame:
    """
    df_pairs must contain: ['aso_id','target_id', aso_col, targ_col]
    Returns: ['aso_id','target_id','dG37_kcalmol','neg_dG_bind','thermo_backend','thermo_notes']
    """
    global THERMO_ASO_COL_OVERRIDE, THERMO_TARGET_COL_OVERRIDE
    if THERMO_ASO_COL_OVERRIDE:
        aso_col = THERMO_ASO_COL_OVERRIDE
    if THERMO_TARGET_COL_OVERRIDE:
        targ_col = THERMO_TARGET_COL_OVERRIDE
    if thermo is None:
        thermo = make_thermo(backend_auto=True, prefer_vienna=True)

    sig = _dataset_signature(df_pairs, aso_col=aso_col, targ_col=targ_col)
    data_path, _ = _cache_pair(cache_dir, sig)

    if data_path.exists():
        # read cached
        rows = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        out = pd.DataFrame(rows)
    else:
        comp = thermo.add_neg_dG_bind(
            df_pairs.rename(columns={aso_col: "__aso", targ_col: "__targ"}),
            aso_col="__aso", target_col="__targ"
        ).rename(columns={"__aso": aso_col, "__targ": targ_col})

        keep_cols = ["aso_id", "target_id", "dG37_kcalmol", "neg_dG_bind", "thermo_backend", "thermo_notes"]
        comp = comp[keep_cols]
        with data_path.open("w", encoding="utf-8") as f:
            for _, r in comp.iterrows():
                f.write(json.dumps({k: (None if pd.isna(v) else v) for k, v in r.items()}) + "\n")
        out = comp

    return out.sort_values(["aso_id", "target_id"]).reset_index(drop=True)

# -------------------------------------------------------------------
# Smart merge back to features (handles your 'id' special-case)
# -------------------------------------------------------------------
def _merge_back_thermo(
    merged: pd.DataFrame,
    feats: pd.DataFrame,
    feats_keep: pd.DataFrame
) -> pd.DataFrame:
    """
    Your Section-3 merge logic:
      - If 'id' exists in both merged & feats, then:
            feats_keep := concat([feats[['id']], feats_keep], axis=1)
            merged := merged.merge(feats_keep, on='id', how='left')
      - else:
            merged := concat([merged.reset_index(drop=True), feats_keep.reset_index(drop=True)], axis=1)
    """
    out = merged.copy()
    if "id" in merged.columns and "id" in feats.columns:
        fk = pd.concat([feats[["id"]].reset_index(drop=True),
                        feats_keep.reset_index(drop=True)], axis=1)
        out = out.merge(fk, on="id", how="left")
    else:
        out = pd.concat([out.reset_index(drop=True),
                         feats_keep.reset_index(drop=True)], axis=1)

    # Prefer thermo ΔG over proxy when duplicate suffix appears
    if "neg_dG_bind.1" in out.columns:
        out["neg_dG_bind"] = out["neg_dG_bind.1"]
        out.drop(columns=["neg_dG_bind.1"], inplace=True, errors="ignore")

    return out

# -------------------------------------------------------------------
# High-level Section-3 upgrade (exact behavior & ordering)
# -------------------------------------------------------------------
def thermo_upgrade_section3(
    sim_mode: str,
    feat_path: Path,
    merged: pd.DataFrame,
    features_dir: Path,
    thermo_cache_dir: Path,
    *,
    full_bio: bool,
    thermo_cache: bool,
    force_recompute_thermo: bool,
    verbose: bool = True
) -> tuple[pd.DataFrame, Path, Path]:
    """
    Mirrors your Section-3 block precisely. Returns:
      merged_updated, thermo_feat_path (with_thermo.csv), gold_snapshot_path
    """
    assert sim_mode in {"enhanced", "synthetic"}, "thermo_upgrade_section3 should be called only for enhanced/synthetic"

    t0 = time.time()
    feats = pd.read_csv(feat_path)

    # 3a) explicit overrides (hard-fail if missing)
    aso_col = THERMO_ASO_COL_OVERRIDE
    targ_col = THERMO_TARGET_COL_OVERRIDE
    print(f"[thermo] USING COLUMNS: ASO={aso_col}  TARGET={targ_col}")

    if (not aso_col) or (not targ_col) or (aso_col not in feats.columns) or (targ_col not in feats.columns):
        print("[thermo] Available columns:", list(feats.columns))
        raise KeyError("Set THERMO_ASO_COL_OVERRIDE / THERMO_TARGET_COL_OVERRIDE to valid column names.")

    # Normalize target to RNA alphabet (T->U)
    feats[targ_col] = feats[targ_col].astype(str).str.upper().str.replace("T", "U", regex=False)

    # 3b) Choose backend & compute ΔG / neg_dG_bind with cache
    sig = _dataset_signature(feats, aso_col, targ_col)
    feats_keep: Optional[pd.DataFrame] = None
    meta_used: Dict[str, Any] = {}
    feats_aug: Optional[pd.DataFrame] = None

    # Try cache first
    if thermo_cache and not force_recompute_thermo:
        feats_keep, meta_used = _load_thermo_cache(sig, thermo_cache_dir)
        if feats_keep is not None:
            if not meta_used.get("vienna_only", False):
                print("[thermo-cache] not Vienna-only → ignoring")
                feats_keep = None
            else:
                print(f"[thermo-cache] loaded {sig} (rows={len(feats_keep)})")

    # If no cache, compute
    if feats_keep is None:
        tc = make_thermo(backend_auto=False, prefer_vienna=True)
        feats_aug = tc.add_neg_dG_bind(feats, aso_col=aso_col, target_col=targ_col)

        be_counts = feats_aug["thermo_backend"].value_counts(dropna=False)
        print("[thermo] backend counts:\n", be_counts.to_string())

        if full_bio:
            non_vienna = [b for b in be_counts.index if str(b) != "vienna"]
            if non_vienna:
                raise RuntimeError(f"full_bio=True but got {non_vienna}")

        feats_keep = feats_aug[["neg_dG_bind", "dG37_kcalmol", "thermo_backend"]].copy()

        if thermo_cache:
            _save_thermo_cache(
                feats_keep, sig, thermo_cache_dir,
                aso_col=aso_col, targ_col=targ_col,
                vienna_only=True, params={"full_bio": full_bio}
            )

    # 3c) Merge the new features (with your exact missing/merge logic)
    thermo_keep = ["neg_dG_bind", "dG37_kcalmol", "thermo_backend"]

    if feats_keep is None and feats_aug is None:
        raise RuntimeError("Thermo step produced neither cache nor computed features (feats_keep/feats_aug are None).")

    if feats_keep is not None:
        missing = [c for c in thermo_keep if c not in feats_keep.columns]
        if missing:
            if feats_aug is not None and all(c in feats_aug.columns for c in thermo_keep):
                print(f"[thermo] cache missing {missing}; filling from computed frame.")
                feats_keep = feats_aug[thermo_keep].copy()
            else:
                raise RuntimeError(f"Thermo cache missing columns: {missing}. Consider force_recompute_thermo=True once.")
    else:
        feats_keep = feats_aug[thermo_keep].copy()  # type: ignore[index]

    merged_updated = _merge_back_thermo(merged, feats, feats_keep)

    # 3d) Update feature list — we **return** the info to caller so ordering is preserved upstream
    # (Your original code mutated local `feature_cols`; runner should handle this if needed.)

    # 3e) Persist augmented CSV (_with_thermo)
    thermo_feat_path = feat_path.with_name(feat_path.stem + "_with_thermo.csv")
    try:
        pd.concat([feats.reset_index(drop=True), feats_keep.reset_index(drop=True)], axis=1).to_csv(thermo_feat_path, index=False)
        if verbose: print("[thermo] wrote:", thermo_feat_path)
    except Exception as _e:
        if verbose: print("[thermo] skip write (_with_thermo.csv):", _e)

    # 3f) Guardrails
    if "neg_dG_bind" not in merged_updated.columns or merged_updated["neg_dG_bind"].isna().all():
        raise RuntimeError("Thermo step produced all-NaN neg_dG_bind; check overrides and inputs.")
    if merged_updated["neg_dG_bind"].nunique(dropna=True) <= 1:
        raise RuntimeError("Thermo step produced a constant neg_dG_bind; likely wrong columns.")

    d = pd.to_numeric(merged_updated.get("dG37_kcalmol", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(d):
        d_min, d_max = float(d.min()), float(d.max())
        if d_min > -3 or d_max < -40:
            print(f"[thermo][warn] dG range odd (min={d_min:.2f}, max={d_max:.2f})")

    # GOLD SNAPSHOT (same name pattern)
    gold_path = features_dir / f"aso_features_fullthermo_{sig}.csv"
    pd.concat([feats.reset_index(drop=True), feats_keep.reset_index(drop=True)], axis=1).to_csv(gold_path, index=False)
    print("[thermo] gold features saved:", gold_path)

    print(f"[thermo] elapsed: {time.time() - t0:.3f}s (N={len(feats)})")
    return merged_updated, thermo_feat_path, gold_path
