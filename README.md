# Bottle Defect Inspection: project brief

**Status: planned. No code and no results yet.** This file is the plan. Everything below is a target or a design decision, not a finding. Results will be added to this README only after the experiments run, with the commands to reproduce them.

## Goal

Build a computer vision system that looks at a photo of a glass bottle and decides whether it is good or defective, shows where the defect is, and then take it from experiment to a small production service: tracked runs, a served API, tests, CI, and a check that warns when incoming images stop looking like the training images.

This is a stand-in for packaging and product quality inspection (does this bottle look right before it ships?). The dataset is an industrial inspection set, not a consumer product set, and the brief makes no claim about any specific product or brand.

## Why this problem

- It is a verification task: the model has to say "good or defective" and justify it with a location, not only a label.
- Defects are rare in real life, so the training data has only good examples. That makes it an anomaly detection problem, which is harder and more realistic than ordinary classification.
- Small enough to run on one laptop GPU (8 GB), so every experiment can be re-run.
- Public dataset, so anyone can reproduce the work.

## Data

- **Dataset:** the `bottle` category of [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (top-down photos of bottle openings). Training images are defect-free only. The test set has good images and three defect types (large break, small break, contamination), with pixel-level masks showing where each defect is.
- **License:** CC BY-NC-SA 4.0 (non-commercial, share-alike). Fine for a learning and portfolio project. The dataset images are not redistributed in this repo; the README will link to the official download.
- **Size caveat:** the test set is small (under 100 images for this category), so every metric will be reported with a bootstrap confidence interval and read with care.
- **Checks before training:** image counts per split, size and lighting spread, and a visual look at a sample of each defect type.

## Approach

| Step | What | Why |
|---|---|---|
| 1. Baseline | Simple image statistics (color and texture histograms) with scikit-learn anomaly detection | A floor. If a deep model cannot clearly beat this, the extra complexity is not worth it. |
| 2. Image-level model | Pretrained ResNet features from good images only, score a new image by its distance to the nearest good examples | Cheap and strong for anomaly detection, no defect labels needed. |
| 3. Localization | Patch-level features with the same nearest-neighbor idea, to produce a heatmap of where the image looks wrong | Turns "defective" into "defective here". |
| 4. Detection boxes | Turn heatmaps into bounding boxes and compare them with boxes derived from the ground-truth masks | Covers object detection style evaluation. |
| 5. Threshold choice | Pick the score threshold from a stated cost: a missed defect costs more than a false alarm | The threshold is a business decision, so it gets written down and justified. |

## Evaluation plan

- **Metrics:** image-level AUROC, pixel-level AUROC, precision and recall at the chosen threshold, and per-defect-type recall.
- **Error analysis:** look at the false alarms and the missed defects, and say why they happen (small defects, reflections, lighting).
- **Reporting:** bootstrap confidence intervals, and repeated runs where there is randomness. A single run on 80 images is not a result.
- **Target (a hypothesis, not a claim):** the patch-level model beats the image-level model, which beats the baseline. I will say plainly if it does not.

## MLOps part

- **Experiment tracking:** log parameters, metrics, heatmap examples and model files for every run in MLflow.
- **Versioning:** a model registry entry per trained model, with the data version and code version recorded.
- **Serving:** a FastAPI service that takes an image and returns good or defective, the score, and a heatmap. Input validation for file type and size.
- **Packaging:** Docker image for the service.
- **Tests:** pytest for preprocessing, scoring and the API, including bad inputs.
- **CI:** GitHub Actions runs tests and lint on every push.
- **Drift check:** compare the feature distribution of incoming images against the training images with a simple statistic (population stability index or a two-sample test). Prove it works by feeding it deliberately shifted images (different lighting, blur, a different camera angle) and showing it alerts on those and stays quiet on normal ones.
- **Retraining trigger (design first):** document when drift should trigger a refresh of the reference images, and what must be checked before a new model replaces the old one.

## Stack

Python, PyTorch, torchvision, scikit-learn, OpenCV, MLflow, FastAPI, Docker, GitHub Actions, pytest.

## Milestones

1. Data checks, baseline model with results and confidence intervals.
2. Image-level model, then patch-level model with heatmaps.
3. Bounding boxes, threshold choice with a written cost argument.
4. MLflow tracking and model registry in place.
5. FastAPI service, Docker image, tests and CI.
6. Drift check with the shifted-image test, written up.
7. Final README with results, limits, and reproduction steps.

## Known limits (stated up front)

- One object category, controlled studio lighting. Results will not carry over to other products, packaging or real factory conditions without more work.
- Small test set, so conclusions are limited and intervals will be wide.
- The defects are manufactured-style damage and contamination, not the surface or print defects a luxury packaging line would care about.
- A laptop GPU limits model size and the number of runs.
- Drift detection is tested on simulated shifts, not on real production traffic.

## Definition of done

Someone can clone the repo, follow the README, reproduce the reported numbers, run the service in Docker, and see the drift check fire on the shifted images.
