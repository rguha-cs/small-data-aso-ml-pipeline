'''
Functions:
    ensure_sequence_column(df) -> pd.DataFrame (exact code from your _ensure_sequence_column)
    _nn_fallback_dG(dna_20) and _neg_from_seq_or_default(df) as in your cell
    finalize_and_save_features(df, feat_dir, merged, verbose=True) -> Path
    Adds neg_dG_bind if missing (via _neg_from_seq_or_default)
    Fills defaults for p_unpaired_site, OT_weighted, GC_pen, PC exactly as you do
    Merges KD if missing (aso_id or Sequence)
    Writes features/aso_features.csv and returns its Path
'''

# Master_model/src/features.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence as SeqT
import re
import pandas as pd
import numpy as np

# -----------------------------
# Ensure we have a Sequence column (DNA 20-mer)
# -----------------------------
def ensure_sequence_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantees an uppercase DNA 'Sequence' column exists.
    Tries common aliases, else tries to merge from DAZ/FAM TSVs, else raises.
    """
    d = df.copy()

    # If already present
    if "Sequence" in d.columns:
        d["Sequence"] = d["Sequence"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
        return d

    # common alternates
    cands = ["Sequence","sequence","seq","aso_seq","ASO","ASO_seq","dna_seq","tile_sequence","oligo"]
    for c in cands:
        if c in d.columns:
            d["Sequence"] = d[c].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
            return d

    # last resort: try uploaded TSVs (merge by aso_id if present)
    tsv_paths = [Path("/mnt/data/DAZ_top5.tsv"), Path("/mnt/data/FAM_top5.tsv")]
    tsvs = []
    for p in tsv_paths:
        if p.exists():
            try:
                tsvs.append(pd.read_csv(p, sep="\t"))
            except Exception:
                pass
    if tsvs:
        tsv_all = pd.concat(tsvs, ignore_index=True, sort=False)
        src_col = "Sequence" if "Sequence" in tsv_all.columns else ("seq" if "seq" in tsv_all.columns else None)
        if src_col is not None:
            if "aso_id" in d.columns and "aso_id" in tsv_all.columns:
                seq_map = tsv_all[["aso_id", src_col]].drop_duplicates()
                d = d.merge(seq_map.rename(columns={src_col: "Sequence"}), on="aso_id", how="left")
                d["Sequence"] = d["Sequence"].astype(str).str.upper().str.replace(r"\s+","",regex=True)
                return d
            else:
                d["Sequence"] = tsv_all[src_col].astype(str).str.upper().str.replace(r"\s+","",regex=True).iloc[:len(d)].values
                return d

    raise KeyError("Could not find ASO 20-mers. Add a DNA 20-mer column named 'Sequence' to the dataframe.")

# -----------------------------
# Your proxy ΔG helpers (2.4)
# -----------------------------
def _nn_fallback_dG(dna_20: str) -> float:
    """Very light ΔG°37 fallback for proxies only (NOT for reporting).
    GC ≈ -1.6 kcal/mol, AU ≈ -1.0, others -0.5; +0.5 terminal penalty."""
    s = str(dna_20).upper()
    if not s:
        return 0.0
    dG = 0.5
    for ch in s:
        if ch in "GC":
            dG += -1.6
        elif ch in "AT":
            dG += -1.0
        else:
            dG += -0.5
    return float(dG)

def _neg_from_seq_or_default(df: pd.DataFrame) -> pd.Series:
    # Prefer an existing neg_dG-like column if any
    for cand in ("neg_dG_bind_proxy", "neg_dG_gc", "neg_dG", "neg_dG_bind"):
        if cand in df.columns:
            x = pd.to_numeric(df[cand], errors="coerce")
            return x.clip(lower=0.0, upper=40.0)
    # Else compute from Sequence (DNA 20-mer) if present
    if "Sequence" in df.columns:
        vals = df["Sequence"].fillna("").map(_nn_fallback_dG).astype(float)
        return (-vals).clip(lower=0.0, upper=40.0)  # neg_dG = max(0, -dG)
    # Last resort → constant proxy
    return pd.Series(5.0, index=df.index, dtype="float64")

# -----------------------------
# Finalize required columns, merge KD, and save features (2.4 + 2.5)
# -----------------------------
def finalize_and_save_features(
    df: pd.DataFrame,
    feat_dir: Path,
    merged_for_kd: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Path:
    """
    Ensures all required columns exist, merges KD if missing, writes features/aso_features.csv.
    Expects mapping.py to have already set: target_window, target_start_idx,
      p_unpaired_site/logit_punp (if computed), duplex_dG (if computed).
    Adds: neg_dG_bind (proxy if missing), OT_weighted, GC_pen, PC.
    Prints the same debug telemetry you used in the notebook.
    """
    d = df.copy()

    # 2.4a) neg_dG_bind
    if "neg_dG_bind" not in d.columns:
        d["neg_dG_bind"] = _neg_from_seq_or_default(d)

    # 2.4b) p_unpaired_site default (keep transcript-based value if already present)
    if "p_unpaired_site" not in d.columns:
        d["p_unpaired_site"] = 0.5

    # 2.4c) OT_weighted default (if only raw OT)
    if "OT_weighted" not in d.columns:
        if "OT" in d.columns:
            d["OT_weighted"] = np.log1p(pd.to_numeric(d["OT"], errors="coerce")).fillna(0.0)
        else:
            d["OT_weighted"] = 0.0

    # 2.4d) GC_pen default
    if "GC_pen" not in d.columns:
        if "Sequence" in d.columns:
            seq_len = d["Sequence"].str.len().replace(0, np.nan)
            gc_frac = d["Sequence"].str.upper().str.count(r"[GC]").div(seq_len).fillna(0.5)
            d["GC_pen"] = ((gc_frac - 0.5) ** 2) * 0.01
        else:
            d["GC_pen"] = 0.0025

    # 2.4e) PC default
    if "PC" not in d.columns:
        d["PC"] = 0.5

    # Merge KD if missing (aso_id first, else Sequence)
    if "KD" not in d.columns or d["KD"].isna().all():
        if merged_for_kd is not None:
            if {"aso_id","KD"} <= set(merged_for_kd.columns) and "aso_id" in d.columns:
                d = d.merge(merged_for_kd[["aso_id","KD"]], on="aso_id", how="left")
            elif {"Sequence","KD"} <= set(merged_for_kd.columns) and "Sequence" in d.columns:
                d = d.merge(merged_for_kd[["Sequence","KD"]], on="Sequence", how="left")

    # Coerce KD numeric & quick diagnostics (== your Section-2 behavior)
    if "KD" in d.columns:
        d["KD"] = pd.to_numeric(d["KD"], errors="coerce")
        n_nan_kd = int(d["KD"].isna().sum())
        if verbose and n_nan_kd:
            print(f"[features][warn] KD NaN in {n_nan_kd}/{len(d)} rows; those rows will be dropped in training.")
    else:
        print("[features][FATAL] KD column missing; training will fail. Please ensure KD is merged into df here.")
        # If you want a hard fail like the notebook, uncomment the next line:
        # raise KeyError("Step 2: 'KD' column missing in features DataFrame before write.")

    # --- Your debug telemetry before save (added verbatim) ---
    if verbose:
        mapped_count = int((d.get("target_start_idx", -1) >= 0).sum())
        punp_nonnull = int(d["p_unpaired_site"].notna().sum()) if "p_unpaired_site" in d else 0
        has_logit = "logit_punp" in d.columns
        print("[debug] mapped_count:", mapped_count)
        print("[debug] punp_nonnull:", punp_nonnull)
        print("[debug] has_logit_punp:", has_logit)
        if "mapping_mismatches" in d.columns:
            try:
                print("[debug] mismatches stats:", d["mapping_mismatches"].dropna().describe())
            except Exception:
                pass
        # show a small preview of columns to be saved
        print("[debug] columns about to save:", d.columns.tolist()[:20])

    # --- Save features with Sequence + target_window included ---
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_path = feat_dir / "aso_features.csv"
    d.to_csv(out_path, index=False)
    if verbose:
        print(f"[features] wrote {out_path.resolve()}")

    # Optional: enforce KD must exist (hard stop right after write), mirroring your raise
    if "KD" not in d.columns:
        raise KeyError("Step 2: 'KD' column missing in features DataFrame before write.")

    # Optional echo similar to "[train] using features: [...]" (preview only; real selection in Step-4)
    if verbose:
        blacklist = {
            "target","aso_id","group","KD","rel_expr",
            "sequence","Sequence","target_window","target_start_idx",
            "mapped_transcript","mapping_mismatches","thermo_backend",
            "GC_pen","gc_proxy","gc_content","GC","OT_weighted","neg_dG_bind_proxy",
        }
        numeric = [c for c in d.select_dtypes(include=["number"]).columns if c not in blacklist]
        prefer = []
        if "neg_dG_bind" in d.columns: prefer.append("neg_dG_bind")
        if {"p_unpaired_site","logit_punp"} <= set(d.columns):
            ok = d["p_unpaired_site"].notna() & (d.get("mapping_mismatches", 99).fillna(99) <= 3)
            if ok.mean() >= 0.75:
                prefer += ["p_unpaired_site","logit_punp"]
        vis_feats = []
        for c in prefer + numeric:
            if c not in vis_feats:
                vis_feats.append(c)
        print("[train] using features (preview):", vis_feats[:8], "...")

    return out_path

# ========  Phase 9 features and helpers  ===========
import json, numpy as np
from typing import Iterable, Tuple, List

R_GAS = 0.0019872041 # kcal/mol/K
TEMP_K = 310.15

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))

def _dg_open_from_punp(punp: Iterable[float], T: float = TEMP_K) -> Tuple[float, float]:
    p = np.clip(np.asarray(list(punp), dtype=float), 1e-9, 1.0 - 1e-9)
    ED = -R_GAS * T * np.log(p) # per‑base opening energy
    return float(ED.mean()), float(ED.min())

def _extract_punp_profile(row) -> List[float]:
    # Preferred: a JSON list column with per‑base probabilities
    if "punp_profile" in row and isinstance(row["punp_profile"], str):
        try:
            return list(map(float, json.loads(row["punp_profile"])))
        except Exception:
            pass
    # Fallback: derive a flat profile from aggregated logit_punp
    if "logit_punp" in row and row["logit_punp"] is not None:
        p = float(_sigmoid(row["logit_punp"]))
        return [p] * 20
    return [0.5] * 20 # last‑ditch neutral fallback

NUC = "ACGT"
K2 = [a+b for a in NUC for b in NUC]
K3 = [a+b+c for a in NUC for b in NUC for c in NUC]

def _kmer_counts(seq: str, k: int) -> List[int]:
    s = (seq or "").upper().replace("U", "T")
    out = {}
    for i in range(len(s) - k + 1):
        km = s[i:i+k]
        if set(km) <= set(NUC):
            out[km] = out.get(km, 0) + 1
    vocab = K2 if k == 2 else (K3 if k == 3 else [])
    return [out.get(v, 0) for v in vocab]

from sklearn.decomposition import PCA

def add_phase9_features(df, context_windows=(10, 30), k_orders=(2, 3), kmer_pca_dims=10):
    df = df.copy()

    # ΔG_open
    dg_open_mean, dg_open_min = [], []
    for _, row in df.iterrows():
        profile = _extract_punp_profile(row)
        m, mn = _dg_open_from_punp(profile)
        dg_open_mean.append(m)
        dg_open_min.append(mn)
    df["dG_open_mean"] = dg_open_mean
    df["dG_open_min"] = dg_open_min


    # ΔG_eff (use duplex if present else infer from neg_duplex_dG)
    if "duplex_dG" in df.columns:
        duplex = df["duplex_dG"].astype(float)
    elif "neg_duplex_dG" in df.columns:
        duplex = -df["neg_duplex_dG"].astype(float)
    else:
        duplex = np.nan
    df["dG_eff"] = duplex - df["dG_open_mean"].astype(float)
    df["neg_dG_eff"] = -df["dG_eff"]


    # ASO GC%
    if "Sequence" in df.columns:
        s = df["Sequence"].astype(str).str.upper().str.replace("U", "T")
        df["aso_gc_frac"] = s.apply(lambda x: (x.count("G") + x.count("C")) / max(1, len(x)))

    # Target local GC in ±W windows (requires mapped context if available)
    for W in (context_windows or []):
        col = f"target_gc_pm{W}"
        if f"target_context_pm{W}" in df.columns:   # preferred: explicit window
            seqs = df[f"target_context_pm{W}"].astype(str)
        else:
            # Safe fallback order:
            # 1) explicit target window
            # 2) target sequence column if you have one
            # 3) "Sequence" (older schema) or "sequence" (ASO 20-mer proxy)
            # 4) empty strings as a well-formed Series
            if "target_window" in df.columns:
                seqs = df["target_window"].astype(str)
            elif "target_seq" in df.columns:        # if you add this later for thermo
                seqs = df["target_seq"].astype(str)
            elif "target" in df.columns:
                seqs = df["target"].astype(str)
            elif "Sequence" in df.columns:
                seqs = df["Sequence"].astype(str)
            elif "sequence" in df.columns:
                seqs = df["sequence"].astype(str)
            else:
                seqs = pd.Series([""] * len(df), index=df.index, dtype=str)

        df[col] = seqs.str.upper().str.replace("U", "T", regex=False).apply(
            lambda x: (x.count("G") + x.count("C")) / max(1, len(x))
        )

    # Small k-mer embedding → PCA
    k_blocks = []
    if "Sequence" in df.columns:
        seqs = df["Sequence"].astype(str)
        block_cols = []
        for k in (k_orders or []):
            Xk = np.array([_kmer_counts(s, k) for s in seqs])
            k_blocks.append(Xk)
            block_cols += (K2 if k == 2 else (K3 if k == 3 else []))
        if k_blocks:
            X = np.concatenate(k_blocks, axis=1)
            # Safe n_components: cannot exceed rank ≤ min(n_samples-1, n_features)
            n_samples, n_features = X.shape
            n_max = int(min(kmer_pca_dims, n_features, max(1, n_samples - 1)))
            if n_max >= 1:
                try:
                    Z = PCA(n_components=n_max, random_state=0).fit_transform(X)
                    for i in range(Z.shape[1]):
                        df[f"kmer_pca{i+1}"] = Z[:, i]
                except ValueError as e:
                    # Fallback: if PCA still complains (e.g., zero variance), skip PCA
                    # and keep raw k-mer counts (prefixed) so downstream has signal.
                    for j in range(n_features):
                        df[f"kmer_raw{j+1}"] = X[:, j]
            # else: too few samples to run PCA; skip silently
    
    return df

# --- features.py (add near your other feature builders) ---
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
from sklearn.decomposition import PCA

def _all_kmers(k: int):
    alpha = ['A','C','G','T']
    return [''.join(p) for p in product(alpha, repeat=k)]

def _kmer_counts_vocab(seq: str, kmers):
    n, k = len(seq), len(kmers[0])
    if n < k: return np.zeros(len(kmers), dtype=float)
    counts = Counter(seq[i:i+k] for i in range(n-k+1))
    return np.array([counts.get(km, 0) for km in kmers], dtype=float)

def add_kmer_pca_features(df: pd.DataFrame,
                          seq_col: str = "sequence",
                          k_list=(5,6),
                          n_components_per_k=3,
                          prefix="kmer"):
    """
    Adds PCA-compressed k-mer features for the ASO sequence.
    - Fits PCA on the present frame; in your pipeline, do this on TRAIN only and
      reuse the fitted PCA to transform VAL (prevent leakage).
    - Returns df and a dict of fitted PCA models keyed by k.
    """
    out = df.copy()
    pca_models = {}
    for k in k_list:
        kmers = _all_kmers(k)
        X = np.vstack([_kmer_counts_vocab(s, kmers) for s in out[seq_col].astype(str)])
        # normalize to frequencies to reduce length effects
        X = X / (X.sum(axis=1, keepdims=True) + 1e-9)
        # small dimensionality (safe for small-n)
        comps = min(n_components_per_k, min(X.shape)-1)
        if comps <= 0: continue
        pca = PCA(n_components=comps, svd_solver="auto", random_state=0)
        Z = pca.fit_transform(X)
        for j in range(Z.shape[1]):
            out[f"{prefix}{k}_PC{j+1}"] = Z[:, j]
        pca_models[k] = pca
    return out, pca_models

def transform_kmer_pca_features(df: pd.DataFrame, pca_models: dict,
                                seq_col: str = "sequence",
                                prefix="kmer"):
    """Use fitted PCA models (from TRAIN) to transform a new frame (VAL)."""
    out = df.copy()
    for k, pca in pca_models.items():
        kmers = _all_kmers(k)
        X = np.vstack([_kmer_counts_vocab(s, kmers) for s in out[seq_col].astype(str)])
        X = X / (X.sum(axis=1, keepdims=True) + 1e-9)
        Z = pca.transform(X)
        for j in range(Z.shape[1]):
            out[f"{prefix}{k}_PC{j+1}"] = Z[:, j]
    return out
