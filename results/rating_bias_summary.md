# Rating-Habit Bias Experiment

## Configuration

- Catalog: 5,000 titles; minimum 20 ratings.
- Reused targets/users: 65,179; seed-target pairs: 127,922.
- Holdout and pair seeds: fixed from item 1 (42 / 2026); bootstrap seed 31415, 2000 resamples.
- Adjusted cosine: subtract each user's mean observed training rating before item-item cosine.
- Candidate exclusions, top K = 10, and user-level pair averaging are unchanged.
- Neighbor review: 8 fixed seed titles, raw and adjusted top-10 lists in `results/rating_bias_neighbors.md`.

## Metrics

| model           | recall@10 | recall_ci_low | recall_ci_high | ndcg@10  | ndcg_ci_low | ndcg_ci_high | users_evaluated |
| --------------- | --------- | ------------- | -------------- | -------- | ----------- | ------------ | --------------- |
| raw_cosine      | 0.148476  | 0.146458      | 0.150624       | 0.087712 | 0.086255    | 0.089138     | 65179           |
| adjusted_cosine | 0.109314  | 0.107542      | 0.111117       | 0.067506 | 0.066254    | 0.068882     | 65179           |
| popularity      | 0.139692  | 0.137130      | 0.142423       | 0.075972 | 0.074410    | 0.077604     | 65179           |

## Paired adjusted-minus-raw change

| metric    | adjusted_minus_raw | ci_low    | ci_high   |
| --------- | ------------------ | --------- | --------- |
| recall@10 | -0.039161          | -0.040819 | -0.037474 |
| ndcg@10   | -0.020206          | -0.021078 | -0.019330 |

Raw model plus pair evaluation took 20.0 seconds; adjusted model plus pair evaluation took 20.2 seconds. The popularity reference is unchanged because the training split and user histories are the same.
