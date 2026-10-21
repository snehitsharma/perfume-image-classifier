"""Download and load the perfume catalog."""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from perfume_classifier.config import Config

TEXT_COLUMNS = [
    "brand", "name_perfume", "family", "subfamily", "fragrances",
    "origin", "gender", "years", "description", "image_name",
]


def download(cfg: Config) -> None:
    """Fetch the CSV and image archive at the pinned revision, then extract the images."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in (cfg.csv_file, cfg.images_file):
        hf_hub_download(
            repo_id=cfg.repo_id, repo_type="dataset", filename=filename,
            revision=cfg.revision, local_dir=cfg.raw_dir,
        )
    if not cfg.images_dir.exists():
        safe_extract(cfg.raw_dir / cfg.images_file, cfg.images_dir.parent)


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip, refusing any member that would land outside dest."""
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"unsafe path in archive: {member}")
        zf.extractall(dest)


def parse_ingredients(value: object) -> list[str]:
    """The CSV stores ingredients as a Python list literal, e.g. "['Rose', 'Musk']"."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip() for x in parsed if str(x).strip()]


def load_catalog(csv_path: Path) -> pd.DataFrame:
    """Read the pipe-separated catalog. Empty strings become NA; ingredients become lists."""
    df = pd.read_csv(csv_path, sep="|", dtype=str, keep_default_na=False)
    for col in TEXT_COLUMNS:
        df[col] = df[col].str.strip().replace("", pd.NA)
    df["ingredients"] = df["ingredients"].map(parse_ingredients)
    return df
