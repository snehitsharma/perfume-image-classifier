# Perfume Scanner: project brief

**Status: planned. No code and no results yet.** This is a plan. Everything below is a target or a design decision, not a finding. Results get added only after the runs, with the commands to reproduce them.

## Goal

Take a photo of a perfume bottle and return what the fragrance is, its scent profile (top, heart and base notes, main accords, in plain words), and a short explanation of what those ingredients are and how fragrances of that type are put together.

## What a photo can and cannot tell you

A photo cannot reveal ingredients or a recipe. Two bottles can look nothing alike and smell the same, and a formula is a trade secret that does not appear in any image. So the system is built in three honest steps, not as a model that "reads" a scent from a picture:

1. **Recognize the product** from the photo (computer vision).
2. **Look up its notes** in a table of public fragrance information (retrieval).
3. **Explain** the notes in plain language, only from what was retrieved (a language model with grounding).

**Out of scope on purpose:** "how to make it". Real formulas are proprietary, cannot be recovered from a photo, and a language model asked for one would invent it. The explanation covers the general structure of a fragrance (what top, heart and base notes are, what ingredient families do) and nothing that pretends to be a formula.

## Data

- **Bottle photos:** start small with a closed set of 15 to 30 products, with photos I take myself from several angles and lighting conditions (my own photos avoid image-rights problems). Add more products later.
- **Notes table:** a public fragrance dataset with note pyramids. Candidates include the Parfumo-based dataset used in the TidyTuesday project. **Check the source and its license before use**; some fragrance sites forbid scraping, and I will not scrape sites that do. If no usable license exists, I build the notes table by hand for the products in the closed set from the manufacturer's published descriptions.
- **Split:** hold out photos taken in different sessions for the test set, so the test measures new lighting and angles, not near-duplicate shots.

## Approach

| Step | What | Why |
|---|---|---|
| 1. Baseline | Color and shape features with scikit-learn | A floor to beat. |
| 2. Recognition | Pretrained vision model, fine-tuned in PyTorch on my photos, with augmentation | Main model for step 1 of the system. |
| 3. Unknown bottle | Reject low-confidence predictions with "not in my list" instead of guessing | A wrong confident answer is worse than "I do not know". |
| 4. Retrieval | Look up the product in the notes table, with embeddings for fuzzy name matches | The notes come from data, not from the model. |
| 5. Explanation | A language model writes the scent profile and explanation using only the retrieved notes, and says so when notes are missing | Keeps it grounded. |

## Evaluation plan

- **Recognition:** top-1 and top-3 accuracy and a confusion matrix on the held-out photo session, plus how often the model correctly says "not in my list" on bottles it was never trained on.
- **Explanation:** check that every note and claim in the answer appears in the retrieved data (grounding and faithfulness checks), and review a sample by hand.
- **Reporting:** a small closed set is easy to look good on, so the README will say how many products and photos were used and show the failures.
- **Target (a hypothesis):** the fine-tuned model clearly beats the baseline, and the explanations contain no note that is not in the table.

## Small production part

MLflow for runs, a FastAPI endpoint that takes a photo and returns the product, the notes and the explanation, a Docker image, pytest with CI. Drift monitoring is a later extension.

## Stack

Python, PyTorch, scikit-learn, embeddings and a vector store, an LLM API, MLflow, FastAPI, Docker, GitHub Actions, pytest.

## Milestones

1. Collect photos for a closed set of products, baseline model.
2. Fine-tuned recognition model with the "not in my list" rejection, evaluated on the held-out session.
3. Notes table and retrieval.
4. Grounded explanation and its checks.
5. FastAPI, Docker, tests and CI.
6. README with results, failures, limits and reproduction steps.

## Known limits

- A closed set of products, not every perfume. Look-alike bottles and re-designed packaging will confuse it.
- Notes tables come from public sources and brand marketing, so they describe the official note list, not the actual chemistry.
- Scent is subjective, so the explanation describes a fragrance, it does not rate it.
- No brand is affiliated with this project.

## Definition of done

Someone can clone the repo, follow the README, reproduce the numbers on the same photo split, and run the service in Docker.
