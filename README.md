# Perfume Image Classifier: project brief

**Status: planned. No code and no results yet.** This is a plan. Everything below is a target or a design decision, not a finding. Results get added only after the runs, with the commands to reproduce them.

## Goal

Take a photo of a perfume bottle and return which perfume it most likely is (image matching against a catalog), a predicted fragrance family (a classifier), its listed ingredients and notes, and a short plain-language description of what it smells like, written only from the retrieved record.

## What a photo can and cannot tell you

A photo cannot reveal ingredients or a recipe. Two bottles can look nothing alike and smell the same, and a formula is a trade secret that never appears in an image. So the system works in three honest steps:

1. **Find the product from the photo** by matching it against a catalog of bottle images (image retrieval).
2. **Read its record**: brand, family, listed notes and ingredients, description (lookup, not guessing).
3. **Explain it** in plain language, using only what the record says, and say so when something is missing (language model with grounding).

**Out of scope on purpose:** "how to make it". Real formulas are proprietary and cannot be recovered from a photo, and a language model asked for one would invent it. The explanation covers what the listed notes and families mean, not a formula.

## Data

- **Catalog:** the Hugging Face dataset [`doevent/perfume`](https://huggingface.co/datasets/doevent/perfume): about 26,000 perfumes (2024 snapshot) with an image, brand, name, family, subfamily, fragrance notes, ingredients, origin, gender, year and description per record. Declared license on the dataset page: MIT. The page does not say where the data was collected, so **check the source terms before publishing anything derived from it**, and do not redistribute the images in this repo.
- **Query photos for testing:** two kinds. (1) Catalog images changed to look like real photos (crop, rotate, blur, lighting change, background change), used for the bulk of the evaluation. (2) A small set of real bottles photographed by me from different angles, matched against the catalog to see how it behaves on real photos.
- **Split:** a fixed seed. Perfumes whose bottles look alike (same brand line) are kept in mind when reading results, since near-duplicates make retrieval look easier than it is.
- **Checks before building:** missing or broken images, duplicate perfumes, how many records have empty notes or descriptions, and brand and family balance.

## Approach

| Step | What | Why |
|---|---|---|
| 1. Baseline | Color-histogram features with nearest-neighbor search | A floor to beat. |
| 2. Retrieval model | Embed bottle images with a pretrained vision model (a CLIP-style model), store the vectors in a vector index, and return the nearest catalog items | One image per product makes classification impractical (26,000 classes). Retrieval works with one example each. |
| 3. Improve | Compare embedding models, add text matching on any visible brand name, and try a light fine-tune of the embedding model on the augmented images | One change at a time, same data. |
| 3b. Family classifier | A simple classifier on the same image embeddings that predicts the fragrance family (for example floral or oriental), compared with a majority-class baseline | Adds a real classification task. The link between a bottle's look and its family may be weak, and the evaluation will say how weak. |
| 4. Unknown bottle | If the best match score is low, answer "not in the catalog" instead of guessing | A wrong confident answer is worse than "I do not know". |
| 5. Lookup | Fetch the matched record | The notes come from data, not from the model. |
| 6. Explanation | A language model writes the profile from the record only, and says what is missing | Keeps it grounded. |

## Evaluation plan

- **Retrieval:** top-1, top-5 and top-10 accuracy on the augmented query set, split by how hard the change is, plus the same on the small set of real photos. Report the rate of correct "not in the catalog" answers on bottles taken out of the index.
- **Family classifier:** accuracy and macro-F1 against the majority-class baseline, with a confusion matrix. If it does not beat the baseline, the README says so.
- **Explanation:** check that every note and claim in the answer appears in the retrieved record (grounding and faithfulness checks), and read a sample by hand. Compare against an answer written with no retrieval to show the difference.
- **Reporting:** state the catalog size, the number of real photos, and show the failures. A small real-photo set means wide uncertainty, and the README will say so.
- **Target (a hypothesis):** the embedding model clearly beats the baseline, and explanations contain no note that is not in the record.

## Small production part

MLflow for runs, a FastAPI endpoint that takes a photo and returns the product, the record and the explanation, a Docker image, pytest with CI. A drift check on incoming photo embeddings against the catalog is a later extension.

## Stack

Python, PyTorch, a CLIP-style embedding model, a vector database, scikit-learn, an LLM API, MLflow, FastAPI, Docker, GitHub Actions, pytest.

## Milestones

1. Data checks, baseline retrieval and the augmented query set.
2. Embedding retrieval and its evaluation, including the "not in the catalog" rule, and the family classifier.
3. Real-photo test with bottles I photograph.
4. Record lookup and the grounded explanation with its checks.
5. FastAPI, Docker, tests and CI.
6. README with results, failures, limits and reproduction steps.

## Known limits

- Catalog photos are clean product shots; real phone photos are harder.
- Look-alike bottles, redesigned packaging and counterfeits will confuse it.
- Notes come from the listing, so they describe the official note list, not the actual chemistry.
- Scent is subjective, so the explanation describes a fragrance, it does not rate it.
- No brand is affiliated with this project.

## Definition of done

Someone can clone the repo, follow the README, reproduce the numbers on the same query set, and run the service in Docker.
