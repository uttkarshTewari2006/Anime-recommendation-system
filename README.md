# Anime Recommender System

## Project name
Item-based Collaborative Filtering for Anime Recommendations

## Overview
This project builds a recommender system that suggests anime titles similar to a selected anime using explicit user ratings from the MyAnimeList Kaggle dataset. The goal is to create a practical, interpretable recommendation baseline that works well with real rating data and is easy to understand, evaluate, and extend.
---

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
This repository currently targets a strong, explainable, and realistic baseline based on item-based collaborative filtering. It is not intended to be a production-scale recommendation platform yet, but it provides a credible foundation for future upgrades such as metadata-aware reranking, matrix factorization, or hybrid recommendation systems.

---

## Expected outcome
By the end of the project, the system should be able to:
- load the MyAnimeList data,
- compute anime similarity from user rating patterns,
- return top-K similar anime for a selected title,
- evaluate performance using Recall@K and NDCG@K,
- provide a clear baseline that can be improved later.

This project is designed to be understandable, reproducible, and practical for a short sprint while still behaving like a legitimate recommendation system.
