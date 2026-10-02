# Anime Recommender System

## Project name
Item-based Collaborative Filtering for Anime Recommendations

## Overview

This project builds a recommender system that suggests anime titles similar to a selected anime using explicit user ratings from the MyAnimeList Kaggle dataset. The goal is to create a practical, interpretable recommendation baseline that works well with real rating data and is easy to understand, evaluate, and extend.


## Why this project exists
The model is intentionally designed as a strong and clear baseline rather than a highly complex production system. This keeps the project realistic for a short timeline while still producing a working recommendation engine that can be evaluated using standard ranking metrics.

The project is built around the idea that explicit ratings are highly informative, and item similarity is one of the most interpretable ways to generate useful recommendations.

---

## Technical details

### 1. Recommender approach: Item-based collaborative filtering
Why this was chosen:
- Chosen over user-based collaborative filtering because item similarity is usually more stable and easier to explain.
- Chosen over a hybrid approach because the project scope is focused on a clean baseline and a 3-day implementation window.
- Chosen over deep learning because the goal is fast, interpretable, and reproducible results rather than a large neural model.

### 2. Similarity metric: Cosine similarity or Pearson similarity
Why this was chosen:
- Cosine similarity is a strong default for sparse rating vectors and is easy to compute.
- Pearson similarity is useful when we want to account for rating bias and relative preference patterns.
- Chosen over a more complex embedding model because this baseline needs to remain transparent and easy to debug.

### 3. Dataset: MyAnimeList Kaggle anime ratings
Why this was chosen:
- The dataset includes explicit 1–10 ratings, which are ideal for collaborative filtering.
- It also includes anime titles, genres, and synopses, which make the project easier to explain and extend later.
- Chosen over implicit-only behavioral data because explicit ratings give us a cleaner and more interpretable preference signal.

### 4. Output behavior: Top-K similar anime for a selected anime
Why this was chosen:
- This matches the item-based collaborative filtering paradigm cleanly.
- It is easier to validate than a full user-personalization system for a short sprint.
- Chosen over a full user-to-anime ranking pipeline because it is simpler, faster, and more aligned with a clear baseline MVP.

### 5. Evaluation metrics: Recall@K and NDCG@K
Why this was chosen:
- Recall@K measures how many relevant anime are retrieved among the top K recommendations.
- NDCG@K measures whether the most relevant anime are ranked near the top.
- Chosen over Precision@K because candidate coverage matters in recommendation, and over CTR because live click data is not available in an offline baseline.

### 6. Data processing pipeline
Why this was chosen:
- Data cleaning and schema validation are necessary to avoid noisy ratings and incorrect joins.
- The user-item matrix is the core input to item similarity calculations.
- This is much simpler and more controlled than building a full content-heavy or deep-learning pipeline at this stage.

### 7. Programming stack
Why this was chosen:
- Python is the default choice for a clean and reproducible recommender workflow.
- pandas is used for tabular data handling and feature preparation.
- scikit-learn is used for similarity and evaluation utilities.
- Chosen over larger ML frameworks because the project is intentionally designed as a lightweight and accessible baseline.

---

## Recommended project flow
1. Load the anime ratings data and metadata.
2. Clean and validate the schema.
3. Build a user-item matrix.
4. Compute item similarities with cosine or Pearson similarity.
5. For a selected anime title, retrieve the top-K similar anime.
6. Evaluate with Recall@K and NDCG@K.
7. Record the strengths and limitations of the baseline.

---

## Current project scope
This repository provides an explainable item-based collaborative recommender with positive-history scoring and a genre-aware reranker. The tested catalog is intentionally limited to popular anime; cold-start items and production-scale serving are out of scope.

---

## Expected outcome
By the end of the project, the system should be able to:
- load the MyAnimeList data,
- compute anime similarity from user rating patterns,
- return top-K similar anime for a selected title,
- evaluate performance using Recall@K and NDCG@K,
- provide a clear baseline that can be improved later.

This project is designed to be understandable, reproducible, and practical for a short sprint while still behaving like a legitimate recommendation system.

## Day 2 baseline

The baseline implementation is in `anime_recommender.py`, with an experiment workflow in `notebooks/anime_item_cf.ipynb`.

### Dataset setup

Install the dependencies, then download the MyAnimeList anime ratings dataset with KaggleHub:

```powershell
python -m pip install -r requirements.txt
python -c "import kagglehub; path = kagglehub.dataset_download('CooperUnion/anime-recommendations-database'); print('Path to dataset files:', path)"
```

Copy these downloaded files into the project `data/` folder:

- `anime.csv` with `anime_id` and `name` (or `title`); `genre` and `synopsis` are optional metadata.
- `rating.csv` with `user_id`, `anime_id`, and `rating`.

Ratings outside 1-10, including MyAnimeList's `-1` unrated marker, are removed. Duplicate user-anime ratings are averaged. The default run keeps at most the 750 most-rated anime with at least 20 valid ratings each; adjust the notebook's `MAX_ITEMS` and `MIN_ITEM_RATINGS` values to change that working subset. Raw dataset files are ignored by Git.

### Run

Install dependencies and run the test suite:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the complete Day 3 comparison from the repository root:

```powershell
python run_evaluation.py
```

The notebook at `notebooks/anime_item_cf.ipynb` runs the same script. The evaluation creates disjoint validation and test positives for the same eligible users. It tunes the genre weight on validation, then compares the original single-seed cosine baseline, positive-history cosine, the genre hybrid, and popularity on the untouched test split. Previously rated anime are excluded from candidates. The default user-facing recommendation functions use the validation-selected genre weight of 0.10.

The evaluation has one relevant held-out anime per user, so Recall@K is whether that anime appears in the top K and NDCG@K rewards placing it nearer the top. The split is randomized rather than temporal because the common Kaggle ratings file has no timestamp. Similarity uses raw explicit ratings; it does not correct for individual rating-scale bias. The item cap and minimum-count filter limit coverage to popular titles, so cold-start anime are not recommended.

### Day 3 results

On the 62,743-user final test split, the genre hybrid improves Recall@10 by 0.007124 and NDCG@10 by 0.002764 over full positive-history cosine. The largest gain comes from aggregating a user's positive history rather than relying on the original single seed.

| Model | Genre weight | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Original single-seed cosine | 0.00 | 0.138517 | 0.078878 |
| Positive-history cosine | 0.00 | 0.262738 | 0.158557 |
| Genre hybrid | 0.10 | 0.269863 | 0.161320 |
| Popularity | 0.00 | 0.136701 | 0.073877 |

The validation search selected genre weight 0.10 from weights 0.00, 0.05, 0.10, 0.20, 0.35, and 0.50. Detailed validation metrics, per-user test rankings, and a run summary are saved as `results/day3_validation_metrics.csv`, `results/day3_metrics.csv`, `results/day3_user_rankings.csv`, and `results/day3_summary.md`. Day 2 artifacts remain separate under their original `baseline_*` names.

Day 3 scores are not a direct replay of the historical Day 2 metrics: this comparison uses disjoint validation/test positives, requires positive training history, and excludes all already-rated titles for every model. It is an offline randomized holdout, not evidence of online user satisfaction. The single-target-per-user protocol measures retrieval of one known positive and does not measure multiple relevant results or temporal preference changes. See `results/day3_summary.md` for the full configuration and interpretation.

### Next steps

- Repeat the evaluation across several random seeds and report metric variability.
- Expand the working catalog and stratify Recall@10 by item popularity to measure long-tail coverage.
- Add temporal or online evaluation when timestamped interactions or feedback become available.
