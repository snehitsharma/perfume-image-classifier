"""Build the augmented query sets.

manifest.csv: sampled catalog items, one query per difficulty level (the items stay in the index).
held_out_manifest.csv: a disjoint sample whose items are removed from the index for the
"not in the catalog" test.
"""

import numpy as np
import pandas as pd
from PIL import Image

from perfume_classifier.augment import LEVELS, make_query
from perfume_classifier.config import load_config
from perfume_classifier.data import usable_catalog


def build(catalog: pd.DataFrame, items: np.ndarray, subdir: str, seed: int, queries_dir) -> pd.DataFrame:
    rows = []
    for level_no, level in enumerate(LEVELS):
        (queries_dir / subdir / level).mkdir(parents=True, exist_ok=True)
        for item in items:
            # One RNG per (item, level) so a query never depends on the order it was made in.
            item_rng = np.random.default_rng([seed, int(item), level_no])
            name = catalog.loc[item, "image_name"]
            with Image.open(cfg.images_dir / name) as img:
                query = make_query(img, level, item_rng)
            path = f"{subdir}/{level}/{item:05d}.jpg".lstrip("/")
            query.save(queries_dir / path, quality=95)
            rows.append({"query_path": path, "level": level, "target": int(item), "image_name": name})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = load_config()
    catalog = usable_catalog(cfg)
    rng = np.random.default_rng(cfg.seed)
    items = np.sort(rng.choice(len(catalog), size=cfg.query_items, replace=False))
    rest = np.setdiff1d(np.arange(len(catalog)), items)
    held_out = np.sort(rng.choice(rest, size=cfg.held_out_items, replace=False))

    build(catalog, items, "", cfg.seed, cfg.queries_dir).to_csv(cfg.queries_dir / "manifest.csv", index=False)
    build(catalog, held_out, "held_out", cfg.seed, cfg.queries_dir).to_csv(
        cfg.queries_dir / "held_out_manifest.csv", index=False)
    print(f"{len(items) * len(LEVELS)} in-catalog and {len(held_out) * len(LEVELS)} held-out queries "
          f"over a catalog of {len(catalog)}")
