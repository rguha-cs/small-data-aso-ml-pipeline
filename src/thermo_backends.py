from __future__ import annotations
import os, re, shutil, subprocess
from dataclasses import dataclass

def _which(cmd: str):
    return shutil.which(cmd)

def _dna_to_rna(seq: str) -> str:
    return str(seq).upper().replace("T","U").replace(" ","")

def _clean_seq(seq: str) -> str:
    import re
    return re.sub(r"[^ACGTUacgtu]", "", str(seq)).upper()

def _vienna_rnaduplex(aso_dna: str, target_rna: str):
    exe = _which("RNAduplex")
    if not exe:
        return None, "RNAduplex not found"
    q = _dna_to_rna(aso_dna)
    t = _dna_to_rna(target_rna)
    try:
        p = subprocess.run(
            [exe, "--noconv"],
            input=f"{q}\n{t}\n".encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        out = p.stdout.decode("utf-8", errors="ignore").strip()
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", out)
        if not nums:
            return None, f"RNAduplex parse fail: {out[:200]}"
        dG = float(nums[-1])
        return dG, "ViennaRNA RNAduplex OK"
    except Exception as e:
        return None, f"ViennaRNA RNAduplex error: {e}"

def _nupack_dG(aso_dna: str, target_rna: str, T_C: float = 37.0, Na_mM: float = 100.0):
    try:
        import nupack
    except Exception as e:
        return None, f"NUPACK not available: {e}"
    mat = nupack.Model(material='rna', celsius=T_C, sodium=Na_mM*1e-3, magnesium=0.0)
    s1 = nupack.Strand(_dna_to_rna(aso_dna), name='aso')
    s2 = nupack.Strand(_dna_to_rna(target_rna), name='target')
    cplx = nupack.Complex([s1, s2])
    try:
        result = nupack.mfe(cplx, model=mat)
        dG = float(result[0].energy)
        return dG, "NUPACK mfe OK"
    except Exception as e:
        return None, f"NUPACK energy error: {e}"

def _fallback_nn_dG(aso_dna: str, target_rna: str):
    q = _clean_seq(aso_dna).replace("U","T")
    t = _clean_seq(target_rna).replace("T","U")
    n = min(len(q), len(t))
    if n == 0:
        return 0.0, "fallback-NN: empty"
    dG = 0.0
    for i in range(n):
        a = q[i]; b = t[i]
        if (a,b) in {("G","C"),("C","G")}:
            dG += -1.6
        elif (a,b) in {("A","U"),("T","A")}:
            dG += -1.0
        else:
            dG += -0.5
    dG += 0.5
    return float(dG), "fallback-NN approx (replace with real params)"

@dataclass
class ThermoComputer:
    backend: str = "auto"   # "auto" | "vienna" | "nupack" | "fallback"
    Na_mM: float = 100.0
    T_C: float = 37.0

    def _pick_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if _which("RNAduplex"):
            return "vienna"
        try:
            import nupack  # noqa
            return "nupack"
        except Exception:
            return "fallback"

    def compute_dG37(self, aso_dna_20mer: str, target_rna_20mer: str):
        b = self._pick_backend()
        aso = _clean_seq(aso_dna_20mer)
        targ = _clean_seq(target_rna_20mer)
        if len(aso) == 0 or len(targ) == 0:
            return None, b, "empty sequence"
        if b == "vienna":
            val, note = _vienna_rnaduplex(aso, targ)
            if val is not None:
                return val, b, note
            b = "nupack"
        if b == "nupack":
            val, note = _nupack_dG(aso, targ, T_C=self.T_C, Na_mM=self.Na_mM)
            if val is not None:
                return val, "nupack", note
            b = "fallback"
        val, note = _fallback_nn_dG(aso, targ)
        return val, "fallback", note

    def add_neg_dG_bind(self, df, aso_col: str, target_col: str):
        dG_vals, neg_vals, backends, notes = [], [], [], []
        for aso, targ in zip(df[aso_col], df[target_col]):
            dG, b, note = self.compute_dG37(str(aso), str(targ))
            dG_vals.append(dG)
            neg = None if dG is None else float(max(0.0, min(40.0, -float(dG))))
            neg_vals.append(neg)
            backends.append(b)
            notes.append(note)
        out = df.copy()
        out["dG37_kcalmol"] = dG_vals
        out["neg_dG_bind"] = neg_vals
        out["thermo_backend"] = backends
        out["thermo_notes"] = notes
        return out

def make_thermo(backend_auto: bool = True, prefer_vienna: bool = True) -> ThermoComputer:
    if backend_auto:
        return ThermoComputer(backend="auto", Na_mM=100.0, T_C=37.0)
    return ThermoComputer(backend="vienna" if prefer_vienna else "nupack",
                          Na_mM=100.0, T_C=37.0)
