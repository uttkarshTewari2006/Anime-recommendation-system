# Results Log

## 2026-10-02 - Item 1 Positive Seed-Target Protocol

**Status:** Complete. Protocol approved by the user; no manual download or notebook action was required because the dataset files were already present locally.

**Configuration:** Dataset `CooperUnion/anime-recommendations-database`; 750 most-rated titles with minimum 20 valid ratings; 4,097,576 cleaned ratings; relevance threshold 7; raw item-item cosine compared with popularity ranked by training interaction count; top K = 10.

**Split and protocol:** `split_user_holdout`, interaction holdout seed 42, one target rated >= 7 per eligible user. Up to two distinct positive training titles sampled per user using seed 2026; each seed-target pair evaluated independently, then pair metrics averaged within user. Users without positive training seeds are excluded. Both recommenders exclude every anime in the user's training history. There were 65,179 eligible users and 127,922 seed-target pairs. Model similarity was fitted on the training split only.

**Confidence intervals:** Paired user-level percentile bootstrap, 2,000 resamples, random seed 31415. The interval for each model is over the macro-averaged per-user metric; paired deltas use per-user item-CF minus popularity differences.

| Model | Recall@10 | 95% CI | NDCG@10 | 95% CI |
|---|---:|---:|---:|---:|
| Item cosine CF | 0.157367 | [0.155249, 0.159561] | 0.093584 | [0.092068, 0.095070] |
| Popularity | 0.139692 | [0.136961, 0.142408] | 0.075972 | [0.074333, 0.077629] |

| Paired difference (item cosine - popularity) | Mean | 95% CI |
|---|---:|---:|
| Recall@10 | +0.017674 | [+0.014690, +0.020735] |
| NDCG@10 | +0.017612 | [+0.015587, +0.019554] |

**Reproduce:** `python run_protocol1_evaluation.py`.

**Artifacts:** `results/protocol1_metrics.csv`, `results/protocol1_paired_deltas.csv`, `results/protocol1_seed_target_pairs.csv`, `results/protocol1_user_metrics.csv`, and `results/protocol1_summary.md`.

**Caveat:** Day 2 values (item cosine Recall@10 0.1173 / NDCG@10 0.0663; popularity 0.1075 / 0.0549) are historical results from the flawed first-row-seed protocol. They are not comparable performance estimates for this positive seed-target run.
