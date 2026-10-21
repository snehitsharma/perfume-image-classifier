"""Data checks run before building anything: gaps, duplicates, broken images, balance."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image


def check_image(path: Path) -> tuple[bool, tuple[int, int] | None, str | None]:
    """Return (ok, (width, height), mode). A file that PIL cannot fully decode is not ok."""
    try:
        with Image.open(path) as im:
            im.load()
            return True, im.size, im.mode
    except (OSError, ValueError, Image.DecompressionBombError):
        return False, None, None


def missing_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = {col: int(df[col].isna().sum()) for col in df.columns if col != "ingredients"}
    counts["ingredients"] = int((df["ingredients"].str.len() == 0).sum())
    return counts


def duplicate_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        "image_name": int(df["image_name"].dropna().duplicated().sum()),
        "brand_and_name": int(df.duplicated(["brand", "name_perfume"]).sum()),
        "brand_and_name_case_insensitive": int(
            df.assign(b=df["brand"].str.lower(), n=df["name_perfume"].str.lower())
            .duplicated(["b", "n"]).sum()
        ),
    }


def image_report(df: pd.DataFrame, images_dir: Path) -> dict[str, object]:
    on_disk = {p.name for p in images_dir.iterdir() if p.is_file()}
    referenced = set(df["image_name"].dropna())
    broken: list[str] = []
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    for name in sorted(referenced & on_disk):
        ok, size, mode = check_image(images_dir / name)
        if not ok:
            broken.append(name)
            continue
        sizes[size] += 1  # type: ignore[index]
        modes[mode] += 1  # type: ignore[index]
    return {
        "files_on_disk": len(on_disk),
        "rows_with_image_name": len(referenced),
        "rows_missing_image_file": len(referenced - on_disk),
        "files_without_row": len(on_disk - referenced),
        "broken_images": broken,
        "most_common_sizes": [[list(s), n] for s, n in sizes.most_common(5)],
        "modes": dict(modes),
    }


def balance(df: pd.DataFrame) -> dict[str, object]:
    brand_counts = df["brand"].value_counts()
    return {
        "family": df["family"].value_counts().to_dict(),
        "subfamily_classes": int(df["subfamily"].nunique()),
        "gender": df["gender"].value_counts().to_dict(),
        "brands": int(brand_counts.size),
        "top_brands": brand_counts.head(10).to_dict(),
        "brands_with_one_perfume": int((brand_counts == 1).sum()),
    }


def usable_rows(df: pd.DataFrame, images_dir: Path, broken: list[str]) -> pd.Series:
    """Rows that have a readable image on disk."""
    bad = set(broken)
    return df["image_name"].map(
        lambda n: isinstance(n, str) and n not in bad and (images_dir / n).is_file()
    )
