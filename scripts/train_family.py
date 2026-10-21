"""Train and evaluate the fragrance-family classifier on catalog image features, log to MLflow.

Compares: majority class, logistic regression on colour histograms, and on CLIP embeddings,
each with plain and class-balanced weights. The main split keeps every brand on one side.
"""

import json

import mlflow
import numpy as np

from perfume_classifier.classifier import evaluate, logistic, majority, split
from perfume_classifier.config import load_config, run_tags
from perfume_classifier.data import usable_catalog
from perfume_classifier.features import model_features

if __name__ == "__main__":
    cfg = load_config()
    catalog = usable_catalog(cfg)
    paths = [cfg.images_dir / n for n in catalog["image_name"]]
    feats = {kind: model_features(kind, paths, cfg, "catalog") for kind in ("histogram", "clip")}
    y = catalog["family"].to_numpy()
    groups = catalog["brand"].str.lower().to_numpy()
    classes = sorted(set(y))

    report: dict = {"catalog_size": len(catalog), "classes": classes,
                    "clip_model": f"{cfg.embed_model}/{cfg.embed_pretrained}", "splits": {}}
    mlflow.set_experiment("family_classifier")
    for split_name, by_group in (("by_brand", True), ("random", False)):
        train, test = split(y, groups, cfg.clf_test_size, cfg.seed, by_group)
        results = {"train_rows": len(train), "test_rows": len(test),
                   "test_class_counts": {c: int((y[test] == c).sum()) for c in classes}}
        candidates = [("majority", None, majority())]
        for kind in ("histogram", "clip"):
            for balanced in (False, True):
                candidates.append((f"{kind}_logreg{'_balanced' if balanced else ''}", kind,
                                   logistic(cfg.clf_C, balanced, cfg.seed)))
        for name, kind, model in candidates:
            X = feats[kind] if kind else np.zeros((len(y), 1))
            model.fit(X[train], y[train])
            results[name] = evaluate(y[test], model.predict(X[test]), classes)
            with mlflow.start_run(run_name=f"{name}_{split_name}"):
                mlflow.set_tags(run_tags(cfg) | {"split": split_name})
                mlflow.log_params({"model": name, "features": kind or "none", "C": cfg.clf_C,
                                   "test_size": cfg.clf_test_size, "train_rows": len(train), "test_rows": len(test)})
                mlflow.log_metrics({k: results[name][k] for k in ("accuracy", "macro_f1")})
            print(f"{split_name:9s} {name:28s} acc={results[name]['accuracy']:.3f} macroF1={results[name]['macro_f1']:.3f}")
        report["splits"][split_name] = results

    cfg.reports_dir.mkdir(exist_ok=True)
    (cfg.reports_dir / "family_classifier.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
