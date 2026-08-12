# Master_model/src/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# ---------- Dataclasses ----------
@dataclass
class Paths:
    root: Path
    data: Path
    features: Path
    thermo_cache: Path
    synthetic: Path
    model: Path
    predictions: Path
    figures: Path
    results: Path

@dataclass
class Config:
    paths: Paths
    extras: Dict[str, Any]  # everything else from YAML (GENOME, TILE_LEN, etc.)

# ---------- Helpers ----------
def _resolve(root: Path, maybe_rel: str | Path) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (root / p)

def _mkdir_all(paths: Paths) -> None:
    for p in [
        paths.data, paths.features, paths.thermo_cache, paths.synthetic,
        paths.model, paths.predictions, paths.figures, paths.results
    ]:
        p.mkdir(parents=True, exist_ok=True)

# ---------- Loader ----------
def load_config(yaml_path: str | Path = "Master_model/config.yaml",
                project_root_override: Optional[str | Path] = None) -> Config:
    """
    Load config.yaml and return a Config with absolute paths.
    - yaml_path: location of your YAML file.
    - project_root_override: if provided, overrides PROJECT_ROOT from YAML.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {yaml_path}")

    with yaml_path.open("r") as f:
        y = yaml.safe_load(f) or {}

    # Pull root and let caller override if desired
    yaml_root = Path(y.get("PROJECT_ROOT", "Master_model"))
    root = Path(project_root_override) if project_root_override else yaml_root
    root = root if root.is_absolute() else (yaml_path.parent / root)
    root = root.resolve()

    # Map YAML keys -> Paths (resolve relative to root)
    paths = Paths(
        root=root,
        data=_resolve(root, y.get("DATA_DIR", "data/")),
        features=_resolve(root, y.get("FEATURES_DIR", "features/")),
        thermo_cache=_resolve(root, y.get("THERMO_CACHE_DIR", "thermo_cache/")),
        synthetic=_resolve(root, y.get("SYNTHETIC_DIR", "synthetic/")),
        model=_resolve(root, y.get("MODEL_DIR", "model/")),
        predictions=_resolve(root, y.get("PREDICTIONS_DIR", "predictions/")),
        figures=_resolve(root, y.get("FIGURES_DIR", "figures/")),
        results=_resolve(root, y.get("RESULTS_DIR", "results/")),
    )
    _mkdir_all(paths)

    # Keep the rest of YAML and resolve path-like extras relative to root
    extras = dict(y)
    for key in ["GENOME", "PHASTCONS", "BOWTIE_INDEX", "CHROM_SIZES", "OUTPUT_DIR"]:
        if key in extras and isinstance(extras[key], str):
            extras[key] = str(_resolve(root, extras[key]))
    if "REGIONS" in extras and isinstance(extras["REGIONS"], list):
        extras["REGIONS"] = [str(_resolve(root, p)) for p in extras["REGIONS"]]

    # Also store the resolved root for convenience
    extras["PROJECT_ROOT"] = str(root)

    return Config(paths=paths, extras=extras)
