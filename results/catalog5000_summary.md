# 5,000-Title Catalog Results

## Configuration

- Catalog: 5,000 most-rated titles; minimum 20 valid ratings; lowest selected item has 56 training-source ratings before holdout.
- Cleaned interactions: 6,264,212; train interactions: 6,199,033; held-out targets: 65,179.
- Reused item-1 protocol pairs exactly: 127,922 pairs across 65,179 users.
- Relevance threshold: rating >= 7; K = 10; same-seen-title exclusion for both methods.
- Model: raw item cosine; baseline: training interaction-count popularity.
- Holdout seed: 42; seed-sampling seed: 2026; bootstrap seed: 31415; resamples: 2000.

## Metrics

| model       | recall@10 | recall_ci_low | recall_ci_high | ndcg@10  | ndcg_ci_low | ndcg_ci_high | users_evaluated | seed_target_pairs |
| ----------- | --------- | ------------- | -------------- | -------- | ----------- | ------------ | --------------- | ----------------- |
| item_cosine | 0.148476  | 0.146458      | 0.150624       | 0.087712 | 0.086255    | 0.089138     | 65179           | 127922            |
| popularity  | 0.139692  | 0.136961      | 0.142408       | 0.075972 | 0.074333    | 0.077629     | 65179           | 127922            |

## Paired differences (item cosine minus popularity)

| metric    | item_cosine_minus_popularity | ci_low   | ci_high  |
| --------- | ---------------------------- | -------- | -------- |
| recall@10 | 0.008784                     | 0.005837 | 0.011814 |
| ndcg@10   | 0.011740                     | 0.009769 | 0.013644 |

## Resource observations

- Similarity matrix: 5,000 x 5,000; 24,325,294 stored entries; 97.3012% density; CSR arrays use 291.9 MB.
- User-item matrix CSR arrays use 74.7 MB.
- Model build: 6.5 seconds; pair evaluation and bootstrap: 19.4 seconds.

This is a catalog-size experiment only. The user-target pairs, sampled seeds, split, candidate exclusion, scoring method, and metrics are held fixed from item 1.
