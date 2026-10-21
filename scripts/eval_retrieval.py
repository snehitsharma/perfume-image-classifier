"""Evaluate retrieval on the augmented queries and the "not in the catalog" rule, log to MLflow.

Usage: python scripts/eval_retrieval.py --model histogram|clip
"""

import argparse
import json

import mlflow
import numpy as np
import pandas as pd

from perfume_classifier.augment import LEVELS
from perfume_classifier.config import load_config, run_tags
from perfume_classifier.data import usable_catalog
from perfume_classifier.features import MODELS, model_features
from perfume_classifier.retrieval import (
    choose_threshold, hit_rank, open_set_rates, same_label_rank, search, topk_accuracy,
)


def closed_set(index, queries, manifest, labels) -> dict:
    top_idx, _ = search(queries, index, k=10)
    targets = manifest["target"].to_numpy()
    ranks, name_ranks = hit_rank(top_idx, targets), same_label_rank(top_idx, targets, labels)
    out = {}
    for level in LEVELS:
        m = (manifest["level"] == level).to_numpy()
        out[level] = topk_accuracy(ranks[m]) | {f"same_name_{k}": v for k, v in topk_accuracy(name_ranks[m]).items()}
        out[level]["queries"] = int(m.sum())
    out["all"] = topk_accuracy(ranks) | {"queries": len(ranks)}
    return out


def open_set(index, queries, manifest, held_queries, held, seed) -> dict:
    """Index without the held-out items; threshold chosen on half the items, reported on the other half."""
    keep = np.setdiff1d(np.arange(len(index)), held["target"].unique())
    position = np.full(len(index), -1)
    position[keep] = np.arange(len(keep))
    sub = index[keep]

    idx_in, s_in = search(queries, sub, k=1)
    correct_in = idx_in[:, 0] == position[manifest["target"].to_numpy()]
    _, s_out = search(held_queries, sub, k=1)
    s_in, s_out = s_in[:, 0], s_out[:, 0]

    rng = np.random.default_rng(seed)
    def split(df):  # half of the items, with all their levels, go to tuning
        items = df["target"].unique()
        tune = rng.choice(items, size=len(items) // 2, replace=False)
        return df["target"].isin(tune).to_numpy()
    tune_in, tune_out = split(manifest), split(held)

    t = choose_threshold(s_in[tune_in], correct_in[tune_in], s_out[tune_out])
    out = {"threshold": t, "index_size": len(sub), "test_in_queries": int((~tune_in).sum()),
           "test_held_out_queries": int((~tune_out).sum())}
    for level in LEVELS:
        mi = (~tune_in) & (manifest["level"] == level).to_numpy()
        mo = (~tune_out) & (held["level"] == level).to_numpy()
        out[level] = open_set_rates(s_in[mi], correct_in[mi], s_out[mo], t)
    out["all"] = open_set_rates(s_in[~tune_in], correct_in[~tune_in], s_out[~tune_out], t)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()

    cfg = load_config()
    catalog = usable_catalog(cfg)
    manifest = pd.read_csv(cfg.queries_dir / "manifest.csv")
    held = pd.read_csv(cfg.queries_dir / "held_out_manifest.csv")

    index = model_features(args.model, [cfg.images_dir / n for n in catalog["image_name"]], cfg, "catalog")
    queries = model_features(args.model, [cfg.queries_dir / p for p in manifest["query_path"]], cfg, "queries")
    held_queries = model_features(args.model, [cfg.queries_dir / p for p in held["query_path"]], cfg, "held_out")

    labels = (catalog["brand"].str.lower() + "|" + catalog["name_perfume"].str.lower()).to_numpy()
    held_items = held["target"].unique()
    report = {
        "model": args.model if args.model == "histogram" else f"clip {cfg.embed_model}/{cfg.embed_pretrained}",
        "catalog_size": len(catalog),
        "closed_set": closed_set(index, queries, manifest, labels),
        "open_set": open_set(index, queries, manifest, held_queries, held, cfg.seed),
        # Held-out items whose brand + name also appears on another item still in the index.
        "held_out_with_namesake_in_index": int(sum(
            ((labels == labels[i]).sum() > 1) for i in held_items)),
    }

    cfg.reports_dir.mkdir(exist_ok=True)
    out = cfg.reports_dir / f"retrieval_{args.model}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mlflow.set_experiment("retrieval")
    with mlflow.start_run(run_name=f"retrieval_{args.model}"):
        mlflow.set_tags(run_tags(cfg))
        params = {"model": report["model"], "catalog_size": len(catalog),
                  "query_items": cfg.query_items, "held_out_items": cfg.held_out_items}
        mlflow.log_params(params)
        for level, metrics in report["closed_set"].items():
            mlflow.log_metrics({f"{level}_{k}": v for k, v in metrics.items() if k != "queries"})
        for level, metrics in report["open_set"].items():
            if isinstance(metrics, dict):
                mlflow.log_metrics({f"open_{level}_{k}": v for k, v in metrics.items()})
        mlflow.log_metric("open_threshold", report["open_set"]["threshold"])
        mlflow.log_artifact(str(out))

    print(json.dumps(report, indent=2))
