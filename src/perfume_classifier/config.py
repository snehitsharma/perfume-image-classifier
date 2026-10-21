"""Load config.yaml and describe the code and data versions for a run."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    seed: int
    repo_id: str
    revision: str
    csv_file: str
    images_file: str
    query_items: int
    held_out_items: int
    embed_model: str
    embed_pretrained: str
    embed_batch_size: int
    clf_test_size: float
    clf_C: float
    raw_dir: Path
    images_dir: Path
    queries_dir: Path
    features_dir: Path
    reports_dir: Path


def load_config(path: Path = ROOT / "config.yaml") -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ds, paths, emb, clf = raw["dataset"], raw["paths"], raw["embedding"], raw["classifier"]
    return Config(
        seed=int(raw["seed"]),
        repo_id=ds["repo_id"],
        revision=ds["revision"],
        csv_file=ds["csv_file"],
        images_file=ds["images_file"],
        query_items=int(raw["queries"]["items"]),
        held_out_items=int(raw["queries"]["held_out_items"]),
        embed_model=emb["model"],
        embed_pretrained=emb["pretrained"],
        embed_batch_size=int(emb["batch_size"]),
        clf_test_size=float(clf["test_size"]),
        clf_C=float(clf["C"]),
        raw_dir=ROOT / paths["raw_dir"],
        images_dir=ROOT / paths["images_dir"],
        queries_dir=ROOT / paths["queries_dir"],
        features_dir=ROOT / paths["features_dir"],
        reports_dir=ROOT / paths["reports_dir"],
    )


def code_version() -> str:
    """Current git commit, with a '-dirty' suffix if there are uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def run_tags(cfg: Config) -> dict[str, str]:
    """Tags recorded on every MLflow run."""
    return {
        "dataset": cfg.repo_id,
        "dataset_revision": cfg.revision,
        "code_version": code_version(),
        "seed": str(cfg.seed),
    }
