from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from anime_recommender import build_similarity_model, load_dataset
from run_protocol1_evaluation import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    HOLDOUT_SEED,
    K,
    SEED_SAMPLING_SEED,
    _markdown_table,
    evaluate_seed_target_pairs,
    summarize_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
CATALOG_SIZE = 5000
MIN_ITEM_RATINGS = 20


def apply_fixed_targets(
    ratings: pd.DataFrame,
    seed_target_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the item-1 train/test split from its fixed user-target IDs."""
    targets = seed_target_pairs[["user_id", "target_id"]].drop_duplicates()
    if targets.groupby("user_id")["target_id"].nunique().gt(1).any():
        raise ValueError("Expected exactly one held-out target per user.")

    rating_keys = pd.MultiIndex.from_frame(ratings[["user_id", "anime_id"]])
    target_keys = pd.MultiIndex.from_frame(
        targets.rename(columns={"target_id": "anime_id"})[["user_id", "anime_id"]]
    )
    test_mask = rating_keys.isin(target_keys)
    train = ratings.loc[~test_mask].reset_index(drop=True)
    test = ratings.loc[test_mask].reset_index(drop=True)
    if len(test) != len(targets):
        raise ValueError(
            f"Could not restore every held-out target: expected {len(targets)}, found {len(test)}."
        )
    if not test["rating"].ge(7).all():
        raise ValueError("Restored held-out targets must all have ratings of at least 7.")
    return train, test


def main() -> None:
    ratings = load_dataset(
        PROJECT_ROOT / "data" / "anime.csv",
        PROJECT_ROOT / "data" / "rating.csv",
        max_items=CATALOG_SIZE,
        min_item_ratings=MIN_ITEM_RATINGS,
    )
    seed_target_pairs = pd.read_csv(RESULTS_DIR / "protocol1_seed_target_pairs.csv")
    train, test = apply_fixed_targets(ratings, seed_target_pairs)
    if len(test) != seed_target_pairs["user_id"].nunique():
        raise ValueError("Expanded catalog did not preserve the item-1 target users.")

    catalog = ratings[["anime_id", "title"]].drop_duplicates("anime_id")
    fit_start = perf_counter()
    model = build_similarity_model(train, catalog)
    fit_seconds = perf_counter() - fit_start

    evaluation_start = perf_counter()
    scored_pairs = evaluate_seed_target_pairs(model, train, seed_target_pairs, k=K)
    metrics, deltas = summarize_pairs(
        scored_pairs,
        random_state=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES,
    )
    evaluation_seconds = perf_counter() - evaluation_start
    user_metrics = scored_pairs.groupby("user_id", sort=False).agg(
        cf_recall_at_k=("cf_recall_at_k", "mean"),
        cf_ndcg_at_k=("cf_ndcg_at_k", "mean"),
        popularity_recall_at_k=("popularity_recall_at_k", "mean"),
        popularity_ndcg_at_k=("popularity_ndcg_at_k", "mean"),
        seed_target_pairs=("seed_id", "size"),
    ).reset_index()

    similarity_bytes = (
        model.similarities.data.nbytes
        + model.similarities.indices.nbytes
        + model.similarities.indptr.nbytes
    )
    user_item_bytes = (
        model.user_item_matrix.data.nbytes
        + model.user_item_matrix.indices.nbytes
        + model.user_item_matrix.indptr.nbytes
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "catalog5000_metrics.csv", index=False)
    deltas.to_csv(RESULTS_DIR / "catalog5000_paired_deltas.csv", index=False)
    scored_pairs.to_csv(RESULTS_DIR / "catalog5000_seed_target_pairs.csv", index=False)
    user_metrics.to_csv(RESULTS_DIR / "catalog5000_user_metrics.csv", index=False)

    summary = f"""# 5,000-Title Catalog Results

## Configuration

- Catalog: {len(catalog):,} most-rated titles; minimum {MIN_ITEM_RATINGS} valid ratings; lowest selected item has {int(ratings.groupby('anime_id').size().min())} training-source ratings before holdout.
- Cleaned interactions: {len(ratings):,}; train interactions: {len(train):,}; held-out targets: {len(test):,}.
- Reused item-1 protocol pairs exactly: {len(scored_pairs):,} pairs across {scored_pairs['user_id'].nunique():,} users.
- Relevance threshold: rating >= 7; K = {K}; same-seen-title exclusion for both methods.
- Model: raw item cosine; baseline: training interaction-count popularity.
- Holdout seed: {HOLDOUT_SEED}; seed-sampling seed: {SEED_SAMPLING_SEED}; bootstrap seed: {BOOTSTRAP_SEED}; resamples: {BOOTSTRAP_RESAMPLES}.

## Metrics

{_markdown_table(metrics)}

## Paired differences (item cosine minus popularity)

{_markdown_table(deltas)}

## Resource observations

- Similarity matrix: {model.similarities.shape[0]:,} x {model.similarities.shape[1]:,}; {model.similarities.nnz:,} stored entries; {model.similarities.nnz / (model.similarities.shape[0] ** 2):.4%} density; CSR arrays use {similarity_bytes / 1_000_000:.1f} MB.
- User-item matrix CSR arrays use {user_item_bytes / 1_000_000:.1f} MB.
- Model build: {fit_seconds:.1f} seconds; pair evaluation and bootstrap: {evaluation_seconds:.1f} seconds.

This is a catalog-size experiment only. The user-target pairs, sampled seeds, split, candidate exclusion, scoring method, and metrics are held fixed from item 1.
"""
    (RESULTS_DIR / "catalog5000_summary.md").write_text(summary, encoding="utf-8")
    print(f"Catalog: {len(catalog):,}; interactions: {len(ratings):,}")
    print(f"Fixed targets/users: {len(test):,}; seed-target pairs: {len(scored_pairs):,}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(deltas.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Similarity CSR memory: {similarity_bytes / 1_000_000:.1f} MB")
    print(f"Model build: {fit_seconds:.1f}s; evaluation: {evaluation_seconds:.1f}s")


if __name__ == "__main__":
    main()