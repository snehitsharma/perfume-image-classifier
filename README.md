# Perfume Image Classifier: project brief

## Goal

Take a photo of a perfume bottle and return which perfume it most likely is (image matching against a catalog), a predicted fragrance family (a classifier), its listed ingredients and notes, and a short plain-language description of what it smells like, written only from the retrieved record.

## Progress

Done: milestone 1 (data checks, query sets, baseline) and a first pass of milestone 2 (CLIP retrieval, the "not in the catalog" rule, the family classifier). Not started: model comparison and fine-tuning, the real-photo test (milestone 3), the explanation (milestone 4), the API, Docker and CI (milestone 5). Every number below comes from synthetic queries made from catalog images; none comes from a real photo.

### Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126   # GPU build
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py  # ~900 MB, pinned to dataset commit 9d2db45 in config.yaml
python scripts/make_queries.py   # builds the query sets with seed 42
pytest
```

Results reported here were run on one RTX 4070 Laptop GPU. Every run is logged in MLflow (`mlflow ui`) with the dataset revision, code commit and seed.

### Data checks

`python scripts/data_checks.py` writes [reports/data_checks.json](reports/data_checks.json).

- 26,319 catalog rows; 26,283 have a readable image and form the catalog used below. 36 rows have no image name; 139 image files have no row; no image failed to decode.
- No field is empty except `origin` (3,295 rows) and `description` (54 rows). Every row has at least one ingredient.
- 71 rows repeat an earlier brand + name. Most are different releases (different year, sometimes a different family), so they are kept as separate items.
- Family balance is uneven: FLORAL 10,774, WOODY 7,044, AMBERY (ORIENTAL) 2,362, AROMATIC FOUGERE 2,330, CITRUS 1,606, CHYPRE 1,133, LEATHER 1,070 (rows in the full CSV).
- 1,263 brands. Images are clean product shots on a white background, almost all 500 x 667 pixels.

### Query sets

`python scripts/make_queries.py`. With seed 42, 1,000 catalog items are sampled as in-catalog queries and a separate 1,000 as held-out items (removed from the index for the "not in the catalog" test). Each item becomes one query per level, so each set has 3,000 queries. Regenerating gives byte-identical files.

- *easy*: crop to 85 to 100%, small brightness, contrast and colour changes, JPEG.
- *medium*: rotation up to 15 degrees, crop to 70 to 100%, larger lighting changes, blur, lower JPEG quality.
- *hard*: the white background is replaced by a random colour, gradient or texture, then rotation up to 25 degrees, crop to 60 to 90%, colour cast, blur, downscale to 160 px and back, and low JPEG quality.

### Retrieval

`python scripts/eval_retrieval.py --model histogram` and `python scripts/eval_retrieval.py --model clip` write [reports/retrieval_histogram.json](reports/retrieval_histogram.json) and [reports/retrieval_clip.json](reports/retrieval_clip.json). Each query is searched against all 26,283 items by exact nearest-neighbour search.

- *Baseline:* a 256-bin HSV colour histogram of the non-white pixels, Bhattacharyya similarity.
- *CLIP:* OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`), pretrained, no fine-tuning, cosine similarity.

| Level (1,000 queries each) | Baseline top-1 | Baseline top-10 | CLIP top-1 | CLIP top-5 | CLIP top-10 |
|---|---|---|---|---|---|
| easy | 72.9% | 87.2% | 94.7% | 98.9% | 99.4% |
| medium | 36.5% | 49.0% | 87.3% | 96.5% | 98.0% |
| hard | 1.6% | 5.4% | 51.1% | 67.9% | 73.5% |

CLIP beats the baseline at every level, most of all once the background and lighting change. On hard queries it still misses the right item in its top 10 about a quarter of the time. Counting any item with the same brand + name as correct gives identical numbers, so namesake releases did not change these results.

### "Not in the catalog"

The 1,000 held-out items are removed from the index (25,283 items left). If the best match scores below a threshold, the answer is "not in the catalog". The threshold is chosen on half of the items to maximise the average of (in-catalog queries answered with the right item) and (held-out queries rejected), and reported on the other half (1,500 queries of each kind).

| Level | Model | In catalog: right item | In catalog: wrong item | In catalog: rejected | Held out: rejected |
|---|---|---|---|---|---|
| easy | CLIP | 91.6% | 4.6% | 3.8% | 53.8% |
| medium | CLIP | 69.0% | 5.8% | 25.2% | 71.6% |
| hard | CLIP | 5.2% | 1.0% | 93.8% | 98.2% |
| all | CLIP | 55.3% | 3.8% | 40.9% | 74.5% |
| all | Baseline | 19.2% | 4.9% | 75.9% | 91.9% |

This rule works poorly. A right match on a hard query scores lower than a wrong match on an easy one, so one threshold either rejects almost every hard query or accepts many bottles that are not in the catalog: on easy queries, 46% of held-out bottles still get a (wrong) answer. A better rule (for example the gap between the first and second match) is the next thing to try. 4 of the 1,000 held-out items have a namesake release still in the index, so a few "wrong" answers there may be the same perfume.

### Family classifier

`python scripts/train_family.py` writes [reports/family_classifier.json](reports/family_classifier.json), including confusion matrices. Logistic regression on the catalog image features predicts one of 7 families. The main split keeps every brand entirely in train (21,565 images) or test (4,718 images), so the model cannot just recognise a brand's bottle style. A random split is shown for comparison.

| Model | Accuracy (by brand) | Macro-F1 (by brand) | Accuracy (random) | Macro-F1 (random) |
|---|---|---|---|---|
| Majority class (always FLORAL) | 39.2% | 0.080 | 40.9% | 0.083 |
| Colour histogram + logistic regression | 45.3% | 0.176 | 47.1% | 0.180 |
| CLIP + logistic regression | 48.6% | 0.265 | 51.4% | 0.286 |
| CLIP + logistic regression, balanced class weights | 30.7% | 0.253 | 32.8% | 0.271 |

The classifier beats the majority baseline, but the margin is small: a bottle's look says little about its fragrance family. Per-class F1 on the brand split (CLIP, plain weights): FLORAL 0.65, WOODY 0.47, AROMATIC FOUGERE 0.26, CITRUS 0.21, LEATHER 0.13, AMBERY (ORIENTAL) 0.10, CHYPRE 0.03. Most errors go to FLORAL or WOODY. C = 1.0 was fixed in advance and has not been tuned.

## What a photo can and cannot tell you

A photo cannot reveal ingredients or a recipe. Two bottles can look nothing alike and smell the same, and a formula is a trade secret that never appears in an image. So the system works in three honest steps:

1. **Find the product from the photo** by matching it against a catalog of bottle images (image retrieval).
2. **Read its record**: brand, family, listed notes and ingredients, description (lookup, not guessing).
3. **Explain it** in plain language, using only what the record says, and say so when something is missing (language model with grounding).

**Out of scope on purpose:** "how to make it". Real formulas are proprietary and cannot be recovered from a photo, and a language model asked for one would invent it. The explanation covers what the listed notes and families mean, not a formula.

## Data

- **Catalog:** the Hugging Face dataset [`doevent/perfume`](https://huggingface.co/datasets/doevent/perfume): about 26,000 perfumes (2024 snapshot) with an image, brand, name, family, subfamily, fragrance notes, ingredients, origin, gender, year and description per record. Declared license on the dataset page: MIT. The page does not say where the data was collected, so **check the source terms before publishing anything derived from it**, and do not redistribute the images in this repo.
- **Other sources checked (October 2026), not used:** FragDB on Hugging Face (`FragDBnet/fragrance-database`) is a paid database scraped from Fragrantica, with a free 10-row sample under CC BY-NC. The Fragrantica dump on Kaggle (`ledecanteur/fragrantica-perfumes`) has notes and accords but no images and no stated license. Small Roboflow perfume sets (about 100 to 400 real photos, listed as CC BY 4.0) label bottles for detection, not by product. No openly licensed catalog of bottle images with notes was found.
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
