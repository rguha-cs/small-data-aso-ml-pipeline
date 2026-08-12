from __future__ import annotations

import re, numpy as np, pandas as pd
from pathlib import Path
import hashlib
import pandas as pd
from datetime import datetime

def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1-eps)
    return np.log(p/(1-p))

def sigmoid(z):
    z = np.asarray(z, float)
    return 1.0/(1.0 + np.exp(-z))

def _read_design_any_tsv(path):
    import pandas as pd
    try:
        df = pd.read_csv(path, sep="\t")
        if any(("id" in str(c).lower() and "chrom" in str(c).lower()) for c in df.columns):
            df = pd.read_csv(path, sep=r"\s+", engine="python")
        return df
    except Exception:
        return pd.read_csv(path, sep=r"\s+", engine="python")

def _norm_col(c:str)->str:
    return str(c).strip().lower().replace(" ","_")

def _pick_col(df, cands):
    m = {_norm_col(c): c for c in df.columns}
    for k in cands:
        if k in m: return m[k]
    return None

def _norm_id(x):
    s = str(x).strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else s

def _infer_target_from_name(p):
    n = str(p).lower()
    if "daz" in n: return "DAZ"
    if "fam" in n: return "FAM"
    return None

def _has_tool(name: str) -> bool:
    from shutil import which
    return which(name) is not None

def _rc_dna(s: str) -> str:
    tbl = str.maketrans("ACGTacgt", "TGCAtgca")
    return str(s).translate(tbl)[::-1].upper()

def _dna20_to_rna_window(aso20: str) -> str:
    return _rc_dna(aso20).replace("T","U")

def _aso_rc_rna(dna_20):
    tbl = str.maketrans("ACGTacgt", "TGCAtgca")
    rc = str(dna_20).translate(tbl)[::-1].upper()
    return rc.replace("T","U")

def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))

def _spearman_safe(y_true, y_pred):
    from scipy.stats import spearmanr
    if len(y_true) < 2: return np.nan
    return spearmanr(y_true, y_pred).correlation

## RUNS Relate UTILS
def make_run_id(
    *,
    phase: str | None,
    mode: str,                  # basic | enhanced | synthetic
    relative: bool,             # abs vs rel
    alpha: float,
    ablation: bool,
    use_duplex: int,            # 0/1
    use_access: int,            # 0/1
    synthetic: str = "none",    # none | A | AB
    synth_n_per_real: int | None = None,
    synth_weight: float | None = None,
) -> str:
    """
    Build a collision-free, human-readable ID encoding the run configuration.
    Example: P3__basic__abs__a1__abl0__dup1__acc1__synnone__n0__w0
    """
    parts = []
    if phase: parts.append(phase)
    parts.append(mode)
    parts.append("rel" if relative else "abs")
    parts.append(f"a{alpha:g}")
    parts.append(f"abl{1 if ablation else 0}")
    parts.append(f"dup{use_duplex}")
    parts.append(f"acc{use_access}")
    parts.append(f"syn{synthetic}")
    parts.append(f"n{(synth_n_per_real or 0)}")
    parts.append(f"w{(synth_weight or 0)}")
    return "__".join(parts)

def run_paths(root: Path, run_id: str) -> tuple[Path, Path]:
    """
    Create per-run artifact folders and return (model_run_dir, figures_run_dir).
    """
    model_run_dir   = root / "model"   / "runs" / run_id
    figures_run_dir = root / "figures" / "runs" / run_id
    model_run_dir.mkdir(parents=True, exist_ok=True)
    figures_run_dir.mkdir(parents=True, exist_ok=True)
    return model_run_dir, figures_run_dir

def compute_map_sig(df: pd.DataFrame) -> str:
    """
    Step-2 snapshot signature: hash of mapping-defining columns.
    """
    # Be defensive: only use cols if they exist
    seq = df.get("Sequence", pd.Series(dtype=str)).astype(str)
    tw  = df.get("target_window", pd.Series(dtype=str)).astype(str)
    key = (seq + "|" + tw).to_list()
    dig = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:12]
    return dig
# ===========================
# Safe directory management for P8+
# ===========================

def build_results_dir(results_root: str, phase: str, exp_tag: str, timestamped: bool=True) -> Path:
    """
    Build results directory path as results/<phase>/<exp_tag>/<timestamp>/.
    """
    base = Path(results_root) / phase / exp_tag
    if timestamped:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = base / ts
    return base

def safe_mkdir(path: Path, refuse_overwrite: bool=True) -> Path:
    """
    Create directory safely. Refuse to overwrite if path exists (unless disabled).
    """
    if path.exists():
        if refuse_overwrite:
            raise FileExistsError(f"[SAFETY] Refusing to overwrite: {path}")
        # else allow reuse
    path.mkdir(parents=True, exist_ok=not refuse_overwrite)
    return path

def assert_safe_to_write(path: Path, refuse_overwrite: bool=True):
    """
    Guard before writing a file. Raises if file exists and overwrite is not allowed.
    """
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"[SAFETY] Refusing to overwrite existing file: {path}")
    
def config_get(cfg, key: str, default=None):
    # Works for dicts OR objects with attributes
    try:
        return cfg.get(key, default)   # if dict-like
    except AttributeError:
        return getattr(cfg, key, default)