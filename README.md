# Aircraft Classifier: project brief

**Status: planned. No code and no results yet.** This is a small project plan. Everything below is a target, not a finding. Results get added only after the runs, with the commands to reproduce them.

## Goal

A small, finished image classifier: give it a photo of an aircraft and it returns the most likely model variant (for example, which Boeing or Airbus type). Many variants look nearly the same from the side, which makes it a real fine-grained classification problem.

The aim is a clean, complete computer vision project that fits in a couple of weekends: train, compare against a baseline, serve it, test it.

## Data

- **Dataset:** FGVC-Aircraft, loaded with `torchvision.datasets.FGVCAircraft` (100 aircraft variants, with official train, val and test splits).
- **License:** check the terms on the dataset's official page before publishing anything derived from it. Images are not stored in this repo.
- **Checks before training:** class balance, image size spread, and a look at a sample of images from a few classes.

## Approach

| Step | What | Why |
|---|---|---|
| 1. Baseline | Color-histogram features with scikit-learn logistic regression | A simple floor to beat. |
| 2. Transfer learning | Pretrained ResNet in PyTorch, first with frozen layers, then fine-tuned | The main model. |
| 3. Compare | One change at a time (augmentation on or off, layers unfrozen), same seed | So the result can be explained. |

## Evaluation plan

- Accuracy and macro-F1 on the test split, and a confusion matrix.
- A look at the most confused variant pairs and the images the model gets wrong.
- Repeat the final model over a few seeds and report the spread. A single run is not a result.
- **Target (a hypothesis):** the fine-tuned model clearly beats the baseline. I will say so if it does not.

## Small production part

- MLflow to track runs.
- A FastAPI endpoint that takes an image and returns the top 3 variants with scores.
- Docker image.
- pytest for preprocessing and the API, including bad inputs, and a GitHub Actions workflow to run them.

## Stack

Python, PyTorch, torchvision, scikit-learn, MLflow, FastAPI, Docker, GitHub Actions, pytest.

## Milestones

1. Baseline and data checks.
2. Frozen and fine-tuned model, small comparison, error look.
3. MLflow, FastAPI, Docker, tests and CI.
4. README with results, limits, and reproduction steps.

## Known limits

- One curated dataset. Photos from other angles, airports or phones will do worse.
- One seed per run at first; small comparisons are noisy.
- No drift monitoring or retraining in this version.

## Definition of done

Someone can clone the repo, follow the README, reproduce the numbers, and run the service in Docker.
