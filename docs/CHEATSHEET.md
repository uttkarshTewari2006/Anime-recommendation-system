# Project Cheatsheet

## Current approved work

- Item 1 complete: one held-out rating >= 7 per eligible user; sample up to two positive training seeds; macro-average seed-pair metrics within user.
- All training-seen titles excluded for both item cosine and popularity.
- Item 2 complete: 5,000 titles, minimum 20; exact item-1 pairs reused. Item cosine Recall@10 / NDCG@10: 0.148476 / 0.087712; popularity: 0.139692 / 0.075972.
- Keep Day 2 metrics labeled as historical / flawed-protocol results.
- No rating-bias or neighbor-shrinkage experiment is authorized until its approval gate.

## Reproducibility seeds

- Holdout: 42
- Seed sampling: 2026
- User-level bootstrap: 31415, 2,000 resamples
- Eligible users: 65,179
- Seed-target pairs: 127,922
- Item cosine Recall@10 / NDCG@10: 0.157367 / 0.093584
- Popularity Recall@10 / NDCG@10: 0.139692 / 0.075972

## Commands

- Tests: `python -m unittest discover -s tests -v`
- Existing Day 3 runner (not the approved item 1 rerun): `python run_evaluation.py`
- Approved positive seed-target protocol: `python run_protocol1_evaluation.py`
- Approved 5,000-title catalog comparison: `python run_catalog_size_evaluation.py`

Item 1 artifacts: `results/protocol1_summary.md` and `results/protocol1_*.csv`.
Item 2 artifacts: `results/catalog5000_summary.md` and `results/catalog5000_*.csv`.
