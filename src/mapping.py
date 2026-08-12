'''
Function: map_and_access(df, transcripts_dict, verbose=True) -> pd.DataFrame

    Input is the merged from IO step after ensuring Sequence exists
    Responsibilities (in order, same as your cell):
    Create target_window from Sequence (fallback RC→RNA) and set target_start_idx=-1, logit_punp=NaN
    Normalize target → target_norm via alias dict
    For each gene present:
    load candidate FASTAs into RNA strings
    try ≤3 mismatches on forward/reverse; if not found, run RNAduplex best site across all transcripts (your _duplex_best_site), then set target_start_idx, target_window, mapped_transcript, and duplex_dG when duplex fallback used
    Run RNAplfold (W=120, U=20) per used transcript; parse .lunp; fill p_unpaired_site and logit_punp at target_start_idx
    Compute duplex_dG for all mapped rows (your _duplex_energy loop) and make neg_duplex_dG
    Drop target_norm and return updated df
'''


# Master_model/src/mapping.py
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# =========================
# Basic sequence utilities
# =========================

_DNA_COMP = str.maketrans("ATCGatcg", "TAGCtagc")

def _dna_to_rna(s: str) -> str:
    return str(s).upper().replace("T", "U")

def _revcomp_dna(s: str) -> str:
    return str(s).translate(_DNA_COMP)[::-1].upper()

def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))

def safe_logit(p: float, eps: float = 1e-6) -> float:
    """Numerically stable logit on [0,1], clipping away from 0/1."""
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))

def _aso_rc_rna(dna_20: str) -> str:
    """RC(ASO DNA)->RNA 20-mer used for duplex and mapping."""
    rc = _revcomp_dna(dna_20)
    return _dna_to_rna(rc)

# =========================
# FASTA / Vienna helpers
# =========================

def _read_fasta_to_rna(path: str | Path) -> str:
    """Concatenate FASTA sequence lines into one RNA string (T->U)."""
    p = Path(path)
    lines: List[str] = []
    with p.open("r") as f:
        for line in f:
            if not line.startswith(">"):
                lines.append(line.strip())
    rna = "".join(lines).upper().replace("T", "U")
    if not rna or any(c not in "ACGU" for c in rna):
        raise ValueError(f"Transcript parse failed for {path}")
    return rna

def _duplex_energy(aso_rna: str, target_rna: str) -> float:
    """
    Compute RNAduplex energy between two RNAs.
    Returns np.nan on failure.
    """
    try:
        p = subprocess.run(
            ["RNAduplex"],
            input=f"{aso_rna}\n{target_rna}\n".encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        out = p.stdout.decode().strip().splitlines()
        if not out:
            return np.nan
        line = out[-1]  # pattern: UUG..&AUG.. (-12.30)
        return float(line.rsplit("(", 1)[-1].rstrip(")"))
    except Exception:
        return np.nan

def _duplex_best_site(trna_rna: str, aso_dna: str) -> Tuple[int, float, Optional[str]]:
    """
    Scan all 20-nt windows in transcript RNA; pick the index with the most negative RNAduplex ΔG.
    Returns (start_idx, dG, raw_line_or_None). start_idx = -1 if no window could be evaluated.
    """
    aso_rna = _aso_rc_rna(aso_dna)
    L = len(aso_rna)
    if len(trna_rna) < L:
        return -1, float("inf"), None

    best_idx, best_dG, best_txt = -1, float("inf"), None
    for i in range(0, len(trna_rna) - L + 1):
        win = trna_rna[i : i + L]
        try:
            p = subprocess.run(
                ["RNAduplex"],
                input=f"{aso_rna}\n{win}\n".encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            line = p.stdout.decode().strip().splitlines()
            if not line:
                continue
            txt = line[-1]
            dG = float(txt.rsplit("(", 1)[-1].rstrip(")"))
            if dG < best_dG:  # more negative is better
                best_idx, best_dG, best_txt = i, dG, txt
        except Exception:
            continue
    return best_idx, best_dG, best_txt

def _run_rnaplfold(trna_rna: str, W: int = 120, U: int = 20) -> Optional[np.ndarray]:
    """
    Run RNAplfold on a transcript RNA string and return the 'lunp' matrix of shape [N, U],
    where entry [i, u-1] is Pu(i, l=u).
    Returns None if plfold fails or output cannot be parsed.
    """
    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            fa = td_path / "seq.fa"
            fa.write_text(f">transcript\n{trna_rna}\n")

            # Run RNAplfold (stdin mode also works; FASTA file helps keep things explicit)
            p = subprocess.run(
                ["RNAplfold", "-W", str(W), "-u", str(U), "--noLP"],
                cwd=td_path,
                input=(trna_rna + "\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            # Locate *.lunp
            lunp_path = None
            for pat in ("*_lunp", "*.lunp", "plfold_lunp"):
                hits = list(td_path.glob(pat))
                if hits:
                    lunp_path = hits[0]
                    break
            if lunp_path is None:
                return None

            rows: List[List[float]] = []
            with lunp_path.open("r") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    parts = s.split()
                    # header lines begin with 'i' or include 'l=' tokens; skip them
                    if parts[0].lower() == "i" or any(p.startswith("l=") for p in parts):
                        continue
                    if len(parts) < U + 1:
                        continue
                    try:
                        row = [float(x) for x in parts[1 : U + 1]]
                    except ValueError:
                        continue
                    rows.append(row)

            if not rows:
                return None
            return np.array(rows, dtype=float)
    except subprocess.CalledProcessError:
        return None
    except Exception:
        return None

# ===========================================
# Public API: string match + duplex + plfold
# ===========================================

def find_binding_site(transcript_rna: str, aso_dna: str, max_mismatches: int = 0):
    """
    Return (start_idx, mismatches) of the best match of revcomp(ASO_DNA) in transcript RNA,
    allowing up to `max_mismatches`. Returns (-1, None) if not found.
    """
    q = _dna_to_rna(_revcomp_dna(aso_dna))
    L = len(q)
    # exact first
    idx = transcript_rna.find(q)
    if idx >= 0:
        return idx, 0
    # fallback with mismatches
    if max_mismatches > 0 and len(transcript_rna) >= L:
        best = None
        for i in range(0, len(transcript_rna) - L + 1):
            w = transcript_rna[i : i + L]
            mm = _hamming(w, q)
            if mm <= max_mismatches:
                if best is None or mm < best[1]:
                    best = (i, mm)
                    if mm == 0:
                        break
        if best:
            return best
    return -1, None

def map_and_access(
    df: pd.DataFrame,
    transcripts: Dict[str, List[str]],
    aliases: Optional[Dict[str, str]] = None,
    W: int = 120,
    U: int = 20,
    max_mismatches: int = 3,
    use_duplex_fallback: bool = True,
    compute_accessibility: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Main mapping + accessibility routine.

    Inputs
    ------
    df : DataFrame
        Must include columns:
          - 'Sequence'  (DNA ASO 20-mer)
          - 'target'    (gene label; can be missing — then only fallback is used)
    transcripts : dict[str, list[str]]
        Mapping of canonical gene key -> list of FASTA file paths to try.
        Example keys: 'DAZ','FAM','TARDBP'
    aliases : dict[str,str] or None
        Optional alias mapping from label in df['target'] to canonical transcript key.
        If None, target values are used as-is (uppercased).
    W, U : RNAplfold window parameters
    max_mismatches : int
        Max mismatches for string-based mapping before falling back to physics.
    use_duplex_fallback : bool
        If True, when string matching fails, choose site by minimal RNAduplex ΔG across all transcripts.
    compute_accessibility : bool
        If True and RNAplfold is available, compute p_unpaired and logit_punp at the mapped site.
    """
    df = df.copy()

    # --- ensure fallbacks and holder columns
    if "Sequence" not in df.columns:
        raise KeyError("df must have 'Sequence' (DNA ASO 20-mer).")
    if "target" not in df.columns:
        df["target"] = np.nan

    df["target_window"] = df["Sequence"].map(lambda s: _aso_rc_rna(str(s)))
    df["target_start_idx"] = -1
    df["mapped_transcript"] = None
    df["mapping_mismatches"] = np.nan
    df["p_unpaired_site"] = np.nan
    df["logit_punp"] = np.nan
    df["duplex_dG"] = np.nan
    df["neg_duplex_dG"] = np.nan

    # --- normalize/alias target labels
    if aliases is not None:
        df["target_norm"] = df["target"].astype(str).map(lambda x: aliases.get(x.upper(), x.upper()))
    else:
        df["target_norm"] = df["target"].astype(str).str.upper()

    # --- iterate per gene present in df
    for gene_key in sorted(df["target_norm"].dropna().unique()):
        if gene_key not in transcripts:
            if verbose:
                print(f"[map] No transcript list for '{gene_key}' — using fallback only.")
            continue

        # Load candidate transcript RNAs
        cand: List[Tuple[str, str]] = []
        for fasta_name in transcripts[gene_key]:
            fp = Path(fasta_name)
            if not fp.exists():
                if verbose:
                    print(f"[map][warn] FASTA missing: {fp}")
                continue
            try:
                trna = _read_fasta_to_rna(fp)
                cand.append((fasta_name, trna))
            except Exception as e:
                if verbose:
                    print(f"[map][warn] failed to read {fp}: {e}")

        if not cand:
            if verbose:
                print(f"[map][warn] No valid transcripts loaded for {gene_key}.")
            continue

        # Subset rows for this gene
        sel = df["target_norm"].eq(gene_key).values
        idxs = np.where(sel)[0]
        if len(idxs) == 0:
            continue

        # Try mapping each ASO
        mapped_count = 0
        for i in idxs:
            dna20 = str(df.at[i, "Sequence"])
            win_rna = _aso_rc_rna(dna20)  # default fallback window
            L = len(win_rna)

            # 1) try string match with ≤ max_mismatches (forward, then reverse)
            chosen = None  # (pos, mm, fasta_name, trna_seq, orientation)
            for fasta_name, trna in cand:
                pos, mm = find_binding_site(trna, dna20, max_mismatches=max_mismatches)
                if pos != -1:
                    chosen = (pos, mm, fasta_name, trna, "+")
                    break
                # reverse orientation (rare but harmless to try)
                trna_rev = trna[::-1]
                pos2, mm2 = find_binding_site(trna_rev, dna20, max_mismatches=max_mismatches)
                if pos2 != -1:
                    pos_fwd = len(trna) - (pos2 + L)
                    chosen = (pos_fwd, mm2, fasta_name, trna, "-")
                    break

            # 2) duplex best-site fallback if needed
            if chosen is None and use_duplex_fallback:
                best = (-1, float("inf"), None, None)  # (idx, dG, fasta_name, trna_seq)
                for fasta_name2, trna2 in cand:
                    i_best, dG_best, _txt = _duplex_best_site(trna2, aso_dna=dna20)
                    if i_best >= 0 and dG_best < best[1]:
                        best = (i_best, dG_best, fasta_name2, trna2)
                if best[0] >= 0:
                    i_best, dG_best, best_fn, best_trna = best
                    df.at[i, "target_start_idx"] = int(i_best)
                    df.at[i, "target_window"] = best_trna[i_best : i_best + L]
                    df.at[i, "mapped_transcript"] = best_fn
                    df.at[i, "mapping_mismatches"] = np.nan  # physics-driven
                    df.at[i, "duplex_dG"] = float(dG_best)
                    mapped_count += 1
                    continue  # proceed to next ASO (accessibility below)

            # 3) finalize string mapping if found
            if chosen is not None:
                pos, mm, fasta_name, trna, orient = chosen
                df.at[i, "target_start_idx"] = int(pos)
                df.at[i, "target_window"] = trna[pos : pos + L]
                df.at[i, "mapped_transcript"] = fasta_name
                df.at[i, "mapping_mismatches"] = float(mm)
                mapped_count += 1
            # else: keep fallback target_window and -1 start_idx

        if verbose:
            print(f"[map] {gene_key}: matched {mapped_count}/{len(idxs)} ASOs to transcript(s).")

        # --- accessibility per used transcript
        if compute_accessibility and mapped_count > 0:
            used = sorted(set(df.loc[sel & df["mapped_transcript"].notna(), "mapped_transcript"]))
            for fasta_name in used:
                trna = dict(cand)[fasta_name]
                lunp = _run_rnaplfold(trna, W=W, U=U)
                if lunp is None:
                    if verbose:
                        print(f"[plfold][warn] could not compute lunp for {fasta_name}")
                    continue

                mask = sel & df["mapped_transcript"].eq(fasta_name) & (df["target_start_idx"] >= 0)
                for i in np.where(mask)[0]:
                    s = int(df.at[i, "target_start_idx"])
                    if s < 0 or s + U > lunp.shape[0]:
                        df.at[i, "p_unpaired_site"] = np.nan
                        df.at[i, "logit_punp"] = np.nan
                    else:
                        p = float(lunp[s, U - 1])  # Pu(i=s, l=U)
                        df.at[i, "p_unpaired_site"] = p
                        df.at[i, "logit_punp"] = safe_logit(p)

        if verbose and compute_accessibility:
            filled = int(df.loc[sel, "p_unpaired_site"].notna().sum())
            print(f"[plfold] {gene_key}: filled accessibility for {filled} row(s).")

    # --- duplex for ALL mapped rows (even string-matched)
    has_mapped = df["target_start_idx"].astype(int) >= 0
    for i in np.where(has_mapped)[0]:
        dna20 = str(df.at[i, "Sequence"])
        tgt_rna = str(df.at[i, "target_window"])
        if not tgt_rna or len(tgt_rna) < len(dna20):
            continue
        aso_rna = _aso_rc_rna(dna20)
        dG = _duplex_energy(aso_rna, tgt_rna)
        if not np.isnan(dG):
            df.at[i, "duplex_dG"] = float(dG)

    if "duplex_dG" in df.columns:
        df["neg_duplex_dG"] = -pd.to_numeric(df["duplex_dG"], errors="coerce")

    # cleanup helper column
    if "target_norm" in df.columns:
        df.drop(columns=["target_norm"], inplace=True)

    return df
