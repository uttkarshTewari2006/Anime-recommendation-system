# Positive Seed-Target Protocol Results

## Configuration

- Dataset: Kaggle `CooperUnion/anime-recommendations-database`.
- Catalog: 750 most-rated anime; minimum 20 valid ratings per title.
- Cleaned interactions: 4,097,576; training interactions: 4,032,132.
- Relevance threshold: rating >= 7; one randomly held-out positive target per eligible user.
- Seed sampling: up to 2 distinct positive training titles per user; pair-level results macro-averaged within user.
- Split seed: 42; seed-sampling seed: 2026.
- Candidate set: all catalog anime not present in that user's training history; same exclusion for both methods.
- Item-CF: raw cosine; popularity: training interaction count.
- Bootstrap: paired user-level percentile intervals, 2000 resamples, seed 31415.
- Eligible users: 65,179; evaluated seed-target pairs: 127,922.

## Results

| model       | recall@10 | recall_ci_low | recall_ci_high | ndcg@10  | ndcg_ci_low | ndcg_ci_high | users_evaluated | seed_target_pairs |
| ----------- | --------- | ------------- | -------------- | -------- | ----------- | ------------ | --------------- | ----------------- |
| item_cosine | 0.157367  | 0.155249      | 0.159561       | 0.093584 | 0.092068    | 0.095070     | 65179           | 127922            |
| popularity  | 0.139692  | 0.136961      | 0.142408       | 0.075972 | 0.074333    | 0.077629     | 65179           | 127922            |

## Paired user-level differences (item cosine minus popularity)

| metric    | item_cosine_minus_popularity | ci_low   | ci_high  |
| --------- | ---------------------------- | -------- | -------- |
| recall@10 | 0.017674                     | 0.014690 | 0.020735 |
| ndcg@10   | 0.017612                     | 0.015587 | 0.019554 |

Each target is excluded from model training by the fixed holdout. Users without a positive training seed are omitted and counted only if they contribute a pair. Pair metrics are averaged within each user before computing macro means and bootstrap intervals, so users with two seeds do not receive twice the overall weight.

The Day 2 single-seed scores are retained as historical results from a flawed, unrelated seed-target protocol and should not be compared numerically with this run.
