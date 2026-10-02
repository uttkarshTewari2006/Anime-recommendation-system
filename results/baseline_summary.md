# Day 2 Baseline Results

## Run configuration

- Dataset: Kaggle `CooperUnion/anime-recommendations-database`, downloaded with KaggleHub.
- Working catalog: 750 most-rated anime with at least 20 valid ratings each.
- Cleaned interactions: 4,097,576 ratings across 750 anime and 68,852 users.
- Holdout: one rating of at least 7 per eligible user; 65,444 users evaluated.
- Similarity: cosine similarity over explicit ratings, computed from the training split.

## Ranking metrics

| Metric | Result |
|---|---:|
| Recall@10 | 0.117337 |
| NDCG@10 | 0.066312 |

Each evaluation case has one relevant held-out anime. Recall@10 is the fraction of cases where it appears in the top ten; NDCG@10 additionally rewards higher rank positions.

## Popularity comparison

A popularity-only recommender ranks anime by training-set rating count, removes the query seed, and returns the next ten titles. It was evaluated on the same users and holdouts.

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity | 0.107527 | 0.054898 |
| Item cosine similarity | 0.117337 | 0.066312 |
| Absolute difference | +0.009810 | +0.011414 |

The collaborative model is better, but the gain over popularity is modest. Recall@10 is not unusually low against a uniform 10-of-750 candidate expectation (about 0.0133), yet it leaves substantial room for improvement.

## Recommendation example

Seed: `Highschool of the Dead`

| Rank | Anime | Cosine similarity |
|---:|---|---:|
| 1 | Sword Art Online | 0.5643 |
| 2 | Angel Beats! | 0.5613 |
| 3 | High School DxD | 0.5590 |
| 4 | Highschool of the Dead: Drifters of the Dead | 0.5528 |
| 5 | Mirai Nikki (TV) | 0.5354 |
| 6 | Deadman Wonderland | 0.5206 |
| 7 | Elfen Lied | 0.5181 |
| 8 | Shingeki no Kyojin | 0.5163 |
| 9 | Death Note | 0.5120 |
| 10 | Kore wa Zombie Desu ka? | 0.5050 |

## Strengths and limitations

The item-based approach is interpretable, reproducible, and produces plausible neighbors from shared rating patterns. Sparse matrices keep the selected 750-item working set practical, and the positive per-user holdout evaluates whether a selected anime retrieves another title that user rated highly.

The evaluation has a proxy mismatch: for each user, the seed is simply the first row in the retained training history, while the relevant target is one other anime rated at least 7. Those titles may be unrelated, so this is a noisy test of item-to-item similarity. It is not a test of personalized recommendations from the user's full taste profile. A future evaluation should form multiple seed-target pairs from positive co-ratings, or score candidates from a user profile, and compare those methods on the same split.

The randomized split is not temporal; the public file has no timestamps. Raw-rating cosine does not normalize individual rating habits. The 750-item cap favors popular titles and excludes cold-start recommendations. The dataset has no synopsis column, so this run uses collaborative filtering only. The same-split popularity comparison shows the current model's gain is real but small, so this remains a baseline rather than evidence of a strong ranking system.
