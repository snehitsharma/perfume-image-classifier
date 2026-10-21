"""Run the data checks, write reports/data_checks.json and log it to MLflow."""

import json

import mlflow

from perfume_classifier import checks
from perfume_classifier.config import load_config, run_tags
from perfume_classifier.data import load_catalog

if __name__ == "__main__":
    cfg = load_config()
    df = load_catalog(cfg.raw_dir / cfg.csv_file)
    images = checks.image_report(df, cfg.images_dir)
    usable = checks.usable_rows(df, cfg.images_dir, images["broken_images"])  # type: ignore[arg-type]
    report = {
        "rows": len(df),
        "usable_rows": int(usable.sum()),
        "missing": checks.missing_counts(df),
        "duplicates": checks.duplicate_counts(df),
        "images": images,
        "balance": checks.balance(df),
    }

    cfg.reports_dir.mkdir(exist_ok=True)
    out = cfg.reports_dir / "data_checks.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    mlflow.set_experiment("data_checks")
    with mlflow.start_run():
        mlflow.set_tags(run_tags(cfg))
        mlflow.log_metrics({
            "rows": report["rows"],
            "usable_rows": report["usable_rows"],
            "broken_images": len(images["broken_images"]),  # type: ignore[arg-type]
        })
        mlflow.log_artifact(str(out))

    print(json.dumps({k: report[k] for k in ("rows", "usable_rows", "missing", "duplicates")}, indent=2))
    print("broken images:", len(images["broken_images"]))  # type: ignore[arg-type]
    print("sizes:", images["most_common_sizes"], "modes:", images["modes"])
