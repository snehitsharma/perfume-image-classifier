# Pet Breed Classifier: project brief

**Status: planned. No code and no results yet.** This file is the plan. Everything below is a target or a design decision, not a finding. Results will be added to this README only after the experiments run, with the commands to reproduce them.

## Goal

Build an image classifier that labels a photo of a cat or dog with one of 37 breeds, then take it from experiment to a small production service: tracked training runs, a served API, tests, CI, and a check that tells me when incoming images no longer look like the training data.

The point is the full path from prototype to production for a computer vision model, with honest evaluation against a simple baseline, not a record-breaking score.

## Why this problem

- 37 breeds, many of them look alike (for example several spaniels and terriers), so it is harder than a cat-versus-dog split.
- Small enough to train on one laptop GPU (8 GB), so every experiment can be re-run.
- Public dataset, so anyone can reproduce the work.

## Data

- **Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), about 7,400 images, 37 breeds, about 200 images per breed. Check the dataset's license terms before publishing any derived files.
- **Split:** fixed stratified train, validation and test split with a stored seed. The test set is touched once, at the end.
- **Checks before training:** class balance, corrupt or duplicate images, image size spread, and a look at a sample of each class.

## Approach

| Step | What | Why |
|---|---|---|
| 1. Baseline | Color-histogram features with scikit-learn logistic regression | A simple floor. If the deep model cannot beat this by a clear margin, the extra complexity is not worth it. |
| 2. Frozen features | Pretrained ResNet features with a linear classifier | Shows how much comes from pretraining alone. |
| 3. Fine-tuning | Fine-tune a pretrained ResNet in PyTorch, with data augmentation | The main model. |
| 4. Ablations | Learning rate, augmentation on or off, number of unfrozen layers | One change at a time, same seed, so the result can be explained. |

## Evaluation plan

- **Metrics:** accuracy and macro-F1 on the test set, per-class precision and recall, and a confusion matrix.
- **Error analysis:** list the most confused breed pairs and look at the actual misclassified images. Report what the model gets wrong, not only the score.
- **Reporting:** one seed per run at first, then repeat the final model over several seeds and report the spread. A single run is not a result.
- **Target (a hypothesis, not a claim):** the fine-tuned model clearly beats both the baseline and the frozen-feature model. I will say plainly if it does not.

## MLOps part

- **Experiment tracking:** log parameters, metrics, confusion matrices and model files for every run in MLflow.
- **Versioning:** a model registry entry per trained model, with the data split and code version recorded.
- **Serving:** a FastAPI service that takes an image and returns the top 3 breeds with scores. Input validation for file type and size.
- **Packaging:** Docker image for the service.
- **Tests:** pytest for the data pipeline, the preprocessing and the API (including bad inputs).
- **CI:** GitHub Actions to run tests and lint on every push.
- **Drift check:** compare the distribution of incoming image features (and prediction confidence) against the training data, using a simple statistic such as population stability index or a two-sample test. Prove it works by feeding it deliberately shifted images (blur, brightness, a different image source) and showing it raises an alert on those and stays quiet on normal images.
- **Retraining trigger (design only at first):** document when drift should trigger a retrain, and what would be checked before a new model replaces the old one.

## Stack

Python, PyTorch, torchvision, scikit-learn, MLflow, FastAPI, Docker, GitHub Actions, pytest.

## Milestones

1. Data checks and fixed split, baseline model with results.
2. Frozen-feature model and fine-tuned model, ablations, error analysis.
3. MLflow tracking and model registry in place.
4. FastAPI service, Docker image, tests and CI.
5. Drift check with the shifted-image test, written up.
6. Final README with results, limits, and reproduction steps.

## Known limits (stated up front)

- One dataset, studio-quality pet photos. Results will not carry over to phone photos, other animals, or other kinds of images without more work.
- Breed labels in this dataset are not perfect; some confusion is label noise, not model error.
- A laptop GPU limits model size and the number of runs.
- Drift detection here is tested on simulated shifts, not on real production traffic.

## Definition of done

Someone can clone the repo, follow the README, reproduce the reported numbers on the same split, run the service in Docker, and see the drift check fire on the shifted images.
