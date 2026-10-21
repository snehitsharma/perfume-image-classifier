"""Evaluate colour-histogram retrieval on the augmented query set and log it to MLflow."""

import hashlib
import json
from multiprocessing import Pool

import mlflow
import numpy as np
import pandas as pd

from perfume_classifier.augment import LEVELS
from perfume_classifier.baseline import DIM, H_BINS, S_BINS, V_BINS, histogram_from_path
from perfume_classifier.config import load_config, run_tags
from perfume_classifier.data import usable_catalog
from perfume_classifier.retrieval import hit_rank, same_label_rank, search, topk_accuracy


def features(paths: list, cache_dir, name: str) -> np.ndarray:
    """Histograms for paths, cached under a key built from the file names and sizes."""
    key = hashlib.sha1("\n".join(f"{p.name}:{p.stat().st_size}" for p in paths).encode()).hexdigest()[:12]
    cache = cache_dir / f"{name}_{key}.npy"
    if cache.exists():
        return np.load(cache)
    with Pool() as pool:
        feats = np.stack(pool.map(histogram_from_path, paths, chunksize=64))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, feats)
    return feats


if __name__ == "__main__":
    cfg = load_config()
    catalog = usable_catalog(cfg)
    manifest = pd.read_csv(cfg.queries_dir / "manifest.csv")

    index = features([cfg.images_dir / n for n in catalog["image_name"]], cfg.features_dir, "baseline_catalog")
    queries = features([cfg.queries_dir / p for p in manifest["query_path"]], cfg.features_dir, "baseline_queries")

    top_idx, _ = search(queries, index, k=10)
    targets = manifest["target"].to_numpy()
    ranks = hit_rank(top_idx, targets)
    name_labels = (catalog["brand"].str.lower() + "|" + catalog["name_perfume"].str.lower()).to_numpy()
    name_ranks = same_label_rank(top_idx, targets, name_labels)

    results: dict[str, dict[str, float]] = {}
    for level in LEVELS:
        m = (manifest["level"] == level).to_numpy()
        results[level] = topk_accuracy(ranks[m])
        results[level].update({f"same_name_{k}": v for k, v in topk_accuracy(name_ranks[m]).items()})
        results[level]["queries"] = int(m.sum())
    results["all"] = topk_accuracy(ranks) | {"queries": len(ranks)}
    report = {"model": "hsv_histogram", "catalog_size": len(catalog), "results": results}

    cfg.reports_dir.mkdir(exist_ok=True)
    out = cfg.reports_dir / "baseline_retrieval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mlflow.set_experiment("retrieval")
    with mlflow.start_run(run_name="baseline_hsv_histogram"):
        mlflow.set_tags(run_tags(cfg))
        mlflow.log_params({"model": "hsv_histogram", "bins": f"{H_BINS}x{S_BINS}x{V_BINS}", "dim": DIM,
                           "catalog_size": len(catalog), "query_items": cfg.query_items})
        for level, metrics in results.items():
            mlflow.log_metrics({f"{level}_{k}": v for k, v in metrics.items() if k != "queries"})
        mlflow.log_artifact(str(out))

    print(json.dumps(report, indent=2))
