'''
    Function: build_merged(lab_csv, designs, data_dir, verbose=True) -> pd.DataFrame
    LAB + DESIGNS merge block from orig pipeline (KD = 1-rel_expr, id/seq normalization, optional target)
    Uses helpers from utils (_read_design_any_tsv, _pick_col, _norm_id, _infer_target_from_name)
    Writes DATA_DIR/merged_aso_results.csv

    Returns merged
'''

# Master_model/src/io_frames.py

from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Iterable, List, Union

# Local helpers expected in Master_model/src/utils.py
try:
    from utils import _read_design_any_tsv, _pick_col, _norm_id, _infer_target_from_name
except Exception as e:
    raise ImportError(
        "io_frames.py expects utils.py in the same package with "
        "_read_design_any_tsv, _pick_col, _norm_id, _infer_target_from_name"
    ) from e


def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _normalize_lab_table(lab: pd.DataFrame) -> pd.DataFrame:
    """Standardize lab table:
    - Filter to group == 'treated' if 'group' column exists
    - Create KD = 1 - rel_expr (requires 'rel_expr')
    - Normalize 'aso_id' (accept 'ID' or 'id')
    - Ensure optional 'target' exists
    """
    if "group" in lab.columns:
        lab = lab[lab["group"].astype(str).str.lower().eq("treated")].copy()

    if "rel_expr" not in lab.columns:
        raise ValueError("Lab CSV must have 'rel_expr'.")

    lab = lab.copy()
    lab["KD"] = 1.0 - lab["rel_expr"]

    # Normalize aso_id
    if "aso_id" not in lab.columns:
        if "ID" in lab.columns:
            lab = lab.rename(columns={"ID": "aso_id"})
        elif "id" in lab.columns:
            lab = lab.rename(columns={"id": "aso_id"})
        else:
            raise ValueError("Lab CSV must have 'aso_id' or 'ID' (or 'id').")

    lab["aso_id"] = lab["aso_id"].apply(_norm_id)

    # Ensure 'target' is present (may be NaN)
    if "target" not in lab.columns:
        lab["target"] = pd.NA
    else:
        lab["target"] = lab["target"].astype(str).str.strip()

    return lab


def _read_and_normalize_design(path: Path) -> pd.DataFrame:
    """Read a design TSV and return columns: ['aso_id','sequence','target'] (target optional)."""
    df = _read_design_any_tsv(path)

    id_col = _pick_col(df, ["aso_id", "id", "name", "oligo", "oligo_id", "aso", "tile_id"])
    seq_col = _pick_col(df, ["sequence", "seq", "tile_sequence", "oligo", "aso_seq", "dna_seq", "probe_seq"])

    if id_col is None or seq_col is None:
        raise ValueError(f"Design {path.name}: missing id/sequence column. Columns={list(df.columns)}")

    out = df[[id_col, seq_col]].copy()
    out.columns = ["aso_id", "sequence"]

    tgt_col = _pick_col(df, ["target", "gene", "gene_symbol", "target_gene"])
    if tgt_col:
        out["target"] = df[tgt_col].astype(str).str.strip()
    else:
        out["target"] = _infer_target_from_name(path)

    out["aso_id"] = out["aso_id"].apply(_norm_id)
    out["sequence"] = (
        out["sequence"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )
    return out


def build_merged(
    lab_csv: Union[str, Path],
    designs: Iterable[Union[str, Path]],
    data_dir: Union[str, Path],
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge lab KD table with one or more design TSVs.
    - lab_csv: path to CSV with rel_expr and aso_id/ID (optional 'target')
    - designs: iterable of TSV paths (DAZ_top5.tsv, FAM_top5.tsv, ...)
    - data_dir: where to write merged_aso_results.csv
    - returns: merged DataFrame (also written to disk)
    """
    data_dir = _ensure_dir(data_dir)

    lab = pd.read_csv(lab_csv)
    lab = _normalize_lab_table(lab)

    # Read all designs
    design_frames: List[pd.DataFrame] = []
    for p in designs:
        p = Path(p)
        if not p.exists():
            if verbose:
                print(f"[merge][warn] design file not found: {p}")
            continue
        try:
            design_frames.append(_read_and_normalize_design(p))
        except Exception as e:
            raise RuntimeError(f"Failed parsing design file {p}: {e}") from e

    design = pd.concat(design_frames, ignore_index=True) if design_frames else pd.DataFrame()

    # Join keys: prefer aso_id+target if both tables have usable 'target'
    use_two = ("target" in design.columns) and design["target"].notna().any() and lab["target"].notna().any()
    keys = ["aso_id", "target"] if use_two else ["aso_id"]

    merged = pd.merge(lab, design, on=keys, how="left", suffixes=("", "_design"))
    # if both sequence and sequence_design exist, fill from design into sequence
    if "sequence_design" in merged.columns:
        if "sequence" in merged.columns:
            merged["sequence"] = merged["sequence"].fillna(merged["sequence_design"])
        else:
            merged["sequence"] = merged["sequence_design"]
        merged.drop(columns=["sequence_design"], inplace=True, errors="ignore")

    out_path = Path(data_dir) / "merged_aso_results.csv"
    merged.to_csv(out_path, index=False)

    if verbose:
        print(f"[merge] wrote {out_path} rows={len(merged)}")

    return merged


__all__ = ["build_merged"]
