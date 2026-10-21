# Perfume Image Classifier: instructions for the agent building this project

Read `README.md` first. It is the brief. This file says how to work.

## What this project is

A photo of a perfume bottle goes in. The system finds the bottle in a catalog of about 26,000 perfumes by image similarity, predicts its fragrance family with a classifier on the image embeddings, reads the matched record, and writes a short explanation of its scent profile using only that record. Owner: Snehit Sharma. This is a portfolio project that will be linked from his resume, so what is written about it must be true.

## Honesty rules (the most important part)

- **Never invent a result.** No number goes into the README, a commit message or any resume text unless a script in this repo produced it and the command to reproduce it is written down next to it.
- **Never edit results to look better.** Report failures, weak classes and wrong matches. If the model does not beat the baseline, the README says so.
- **State sample sizes.** Catalog size, number of query images, number of real photos. A small real-photo set means wide uncertainty; say it.
- **No recipes.** The system does not output formulas or "how to make it". A photo cannot reveal one and a language model would invent it.
- **Ground the explanation.** Every note or claim in a generated explanation must come from the retrieved record. If a field is empty, the answer says so.
- If the brief and the data disagree (a column is missing, images are broken, the license is unclear), stop and tell Snehit. Do not work around it silently.

## Data rules

- Catalog: Hugging Face dataset `doevent/perfume` (declared MIT; the collection source is not stated). Check the source and terms before publishing anything derived from it.
- Do **not** commit the images, the database or any large data file. Add them to `.gitignore`. The README links to the download and the code fetches them.
- Keep a fixed random seed in one config file. Record dataset version and code version for every run.

## How to work

- Work through the milestones in the README in order. Finish each with its evaluation before starting the next.
- Plain Python, small modules, type hints. Tests with pytest for preprocessing, retrieval scoring and the API, including bad inputs.
- Track every training or evaluation run in MLflow.
- Keep the repo runnable: a fresh clone plus the README steps must reproduce the reported numbers.
- Prefer simple, explainable choices. Compare one change at a time against the baseline.
- When you finish a milestone, update the README: what was done, the result, the command, the limits.

## Git rules

- Commit small and often with plain messages (what changed).
- **No `Co-Authored-By` lines and no AI attribution of any kind in commits or pull requests.** This is Snehit's instruction for all his repos.
- The repo is public. Keep the README status line true at all times: say what is planned and what is done, and never describe unfinished work as finished.

## When the project is finished

Do not touch his resume. Tell Snehit it is done and list, in a short plain note:

1. The final numbers, each with the command that reproduces it.
2. What the system cannot do.
3. Any claim in the README that you could not verify.

He (or the resume agent) will then add a verified row to his claims ledger and fill in the resume bullets. The pending bullets live in his resume vault at `career-vault/projects/scentlens.md` (project name there is Perfume Image Classifier) with placeholders for the numbers.
