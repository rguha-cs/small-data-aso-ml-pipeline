# Master_model/src/aso_kd/preflight.py
from pathlib import Path

def check_inputs(required_files, base=Path(".")):
    results = {}
    for name in required_files:
        p = base / name
        results[name] = p.exists()
    return results

def print_lab_columns(lab_csv):
    p = Path(lab_csv)
    if p.exists():
        import pandas as pd
        lab = pd.read_csv(p)
        print("[lab columns]:", list(lab.columns))
        return list(lab.columns)
    return []
