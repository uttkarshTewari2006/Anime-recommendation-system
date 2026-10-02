# Day 3 Recommender Results

## Run configuration

- Catalog: 750 anime; genre metadata populated for 750.
- Cleaned interactions: 4,097,576.
- Training interactions: 3,969,103.
- Disjoint validation/test cases: 62,743 each, with one positive target per user.
- Relevance: explicit rating of at least 7; ranking cutoff: 10.
- Genre weight selected on validation: 0.10.

## Validation weight search

| genre_weight | recall@10 | ndcg@10  | users_evaluated |
| ------------ | --------- | -------- | --------------- |
| 0.000000     | 0.262468  | 0.158877 | 62743           |
| 0.050000     | 0.265336  | 0.160499 | 62743           |
| 0.100000     | 0.267153  | 0.160577 | 62743           |
| 0.200000     | 0.259806  | 0.154831 | 62743           |
| 0.350000     | 0.237413  | 0.139443 | 62743           |
| 0.500000     | 0.202541  | 0.118006 | 62743           |

## Final test comparison

| model              | genre_weight | recall@10 | ndcg@10  | users_evaluated |
| ------------------ | ------------ | --------- | -------- | --------------- |
| single_seed_cosine | 0.000000     | 0.138517  | 0.078878 | 62743           |
| history_cosine     | 0.000000     | 0.262738  | 0.158557 | 62743           |
| genre_hybrid       | 0.100000     | 0.269863  | 0.161320 | 62743           |
| popularity         | 0.000000     | 0.136701  | 0.073877 | 62743           |

The genre hybrid's Recall@10 change over the original single-seed baseline is +0.131345. The weight was selected on validation and then evaluated once on the disjoint test set. Genre and collaborative scores are normalized per user before blending; already-rated anime are excluded from every ranked list.

## Interpretation and limitations

The test set contains one held-out positive per eligible user, so Recall@10 is the share of users whose held-out title appears in the top 10. NDCG@10 rewards higher placement. This is a randomized implicit-preference proxy, not a temporal evaluation or an online user study. Results apply to the 750-title popular-item catalog and do not measure cold-start quality.

These Day 3 values are not a direct replay of the historical Day 2 metrics: Day 3 requires disjoint validation and test positives plus positive training history, and every method excludes all titles already rated by each user.

For this dataset and catalog, the validation-selected genre weight of 0.10 is the user-facing recommendation default. If the dataset or catalog changes, rerun validation and update `DEFAULT_GENRE_WEIGHT` to the new selected value. The CSV artifacts preserve per-user ranks for reproducibility.

## Next steps

- Repeat the disjoint split with several random seeds and report mean and spread to check that the hybrid gain is stable.
- Increase the catalog size and report Recall@10 by item popularity to quantify long-tail coverage.
- When timestamped interactions or online feedback are available, add a temporal or user-study evaluation before production use.
