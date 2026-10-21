"""Build the augmented query set: sampled catalog items, one query image per difficulty level."""

import numpy as np
import pandas as pd
from PIL import Image

from perfume_classifier.augment import LEVELS, make_query
from perfume_classifier.config import load_config
from perfume_classifier.data import usable_catalog

if __name__ == "__main__":
    cfg = load_config()
    catalog = usable_catalog(cfg)
    rng = np.random.default_rng(cfg.seed)
    items = np.sort(rng.choice(len(catalog), size=cfg.query_items, replace=False))

    rows = []
    for level_no, level in enumerate(LEVELS):
        (cfg.queries_dir / level).mkdir(parents=True, exist_ok=True)
        for item in items:
            # One RNG per (item, level) so a query never depends on the order it was made in.
            item_rng = np.random.default_rng([cfg.seed, int(item), level_no])
            name = catalog.loc[item, "image_name"]
            with Image.open(cfg.images_dir / name) as img:
                query = make_query(img, level, item_rng)
            path = f"{level}/{item:05d}.jpg"
            query.save(cfg.queries_dir / path, quality=95)
            rows.append({"query_path": path, "level": level, "target": int(item), "image_name": name})

    pd.DataFrame(rows).to_csv(cfg.queries_dir / "manifest.csv", index=False)
    print(f"{len(rows)} queries from {len(items)} items over a catalog of {len(catalog)}")
