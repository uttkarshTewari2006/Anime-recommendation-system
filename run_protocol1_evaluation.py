from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anime_recommender import build_similarity_model, load_dataset, split_user_holdout


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
HOLDOUT_SEED = 42
SEED_SAMPLING_SEED = 2026
BOOTSTRAP_SEED = 31415
BOOTSTRAP_RESAMPLES = 2000
MAX_SEEDS_PER_USER = 2
K = 10


def make_seed_target_pairs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    random_state: int = SEED_SAMPLING_SEED,
    max_seeds_per_user: int = MAX_SEEDS_PER_USER,
) -> pd.DataFrame:
    """Sample up to N positive training seeds for each positive held-out target."""
    if max_seeds_per_user < 1:
        raise ValueError("max_seeds_per_user must be at least 1")

    positive_by_user = {
        user_id: np.sort(group.loc[group["rating"] >= 7, "anime_id"].astype(int).unique())
        for user_id, group in train.groupby("user_id", sort=False)
    }
    title_by_id = (
        train.drop_duplicates("anime_id", keep="first")
        .set_index("anime_id")["title"]
        .to_dict()
    )
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    for target in test.itertuples(index=False):
        positive_ids = positive_by_user.get(target.user_id, np.array([], dtype=int))
        if not len(positive_ids):
            continue
        sample_size = min(max_seeds_per_user, len(positive_ids))
        seed_ids = np.sort(rng.choice(positive_ids, size=sample_size, replace=False))
        for seed_number, seed_id in enumerate(seed_ids, start=1):
            rows.append(
                {
                    "user_id": target.user_id,
                    "target_id": int(target.anime_id),
                    "target_title": target.title,
                    "seed_id": int(seed_id),
                    "seed_title": str(title_by_id[int(seed_id)]),
                    "seed_number": seed_number,
                }
            )
    if not rows:
        raise ValueError("No held-out targets have positive training seeds.")
    return pd.DataFrame(rows)


def _rank_target(
    scores: np.ndarray,
    target_index: int,
    seen_indices: np.ndarray,
    anime_ids: np.ndarray,
    k: int,
) -> int | None:
    candidate_mask = scores > 0
    candidate_mask[seen_indices] = False
    if not candidate_mask[target_index]:
        return None
    target_score = scores[target_index]
    precedes_target = (scores > target_score) | (
        (scores == target_score) & (anime_ids < anime_ids[target_index])
    )
    rank = int(np.count_nonzero(candidate_mask & precedes_target) + 1)
    return rank if rank <= k else None


def evaluate_seed_target_pairs(
    model: Any,
    train: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    k: int = K,
) -> pd.DataFrame:
    """Score item-cosine and popularity for identical positive seed-target pairs."""
    if k < 1:
        raise ValueError("k must be at least 1")
    user_to_row = {user_id: index for index, user_id in enumerate(model.user_ids)}
    item_ids = np.asarray(model.anime_ids)
    pairs = pairs[pairs["user_id"].isin(user_to_row)].reset_index(drop=True).copy()
    pairs = pairs[
        pairs["seed_id"].isin(model.id_to_index)
        & pairs["target_id"].isin(model.id_to_index)
    ].reset_index(drop=True)
    seed_indices = pairs["seed_id"].map(model.id_to_index).to_numpy(dtype=np.int32)
    target_indices = pairs["target_id"].map(model.id_to_index).to_numpy(dtype=np.int32)
    user_rows = pairs["user_id"].map(user_to_row).to_numpy(dtype=np.int32)

    popular_ids = (
        train.groupby("anime_id").size()
        .sort_values(ascending=False, kind="stable")
        .index.astype(int)
        .tolist()
    )
    popularity_positions = {
        anime_id: position for position, anime_id in enumerate(popular_ids, start=1)
    }
    popularity_ranks: dict[Any, int | None] = {}
    for user_id, user_pairs in pairs.groupby("user_id", sort=False):
        row_index = user_to_row[user_id]
        start = model.user_item_matrix.indptr[row_index]
        stop = model.user_item_matrix.indptr[row_index + 1]
        seen_indices = model.user_item_matrix.indices[start:stop]
        seen_popularity_positions = np.fromiter(
            (
                popularity_positions[model.anime_ids[index]]
                for index in seen_indices
                if model.anime_ids[index] in popularity_positions
            ),
            dtype=np.int32,
        )
        seen_popularity_positions.sort()
        for target_id in user_pairs["target_id"].unique():
            target_position = popularity_positions.get(int(target_id))
            if target_position is None:
                popularity_ranks[(user_id, int(target_id))] = None
                continue
            excluded_before = np.searchsorted(
                seen_popularity_positions, target_position, side="left"
            )
            excluded_target = np.searchsorted(
                seen_popularity_positions, target_position, side="right"
            ) > excluded_before
            rank = target_position - int(excluded_before)
            popularity_ranks[(user_id, int(target_id))] = (
                rank if not excluded_target and rank <= k else None
            )

    pair_rows: list[dict[str, Any]] = []
    batch_size = 512
    for batch_start in range(0, len(pairs), batch_size):
        batch_stop = min(batch_start + batch_size, len(pairs))
        batch_scores = model.similarities[seed_indices[batch_start:batch_stop]].toarray()
        for offset, pair in enumerate(pairs.iloc[batch_start:batch_stop].itertuples(index=False)):
            row_index = user_rows[batch_start + offset]
            seen_start = model.user_item_matrix.indptr[row_index]
            seen_stop = model.user_item_matrix.indptr[row_index + 1]
            seen_indices = model.user_item_matrix.indices[seen_start:seen_stop]
            cf_rank = _rank_target(
                batch_scores[offset],
                target_indices[batch_start + offset],
                seen_indices,
                item_ids,
                k,
            )
            pop_rank = popularity_ranks[(pair.user_id, int(pair.target_id))]
            pair_rows.append(
                {
                    **pair._asdict(),
                    "cf_rank": cf_rank,
                    "cf_recall_at_k": int(cf_rank is not None),
                    "cf_ndcg_at_k": 1 / np.log2(cf_rank + 1) if cf_rank is not None else 0.0,
                    "popularity_rank": pop_rank,
                    "popularity_recall_at_k": int(pop_rank is not None),
                    "popularity_ndcg_at_k": 1 / np.log2(pop_rank + 1) if pop_rank is not None else 0.0,
                }
            )
    return pd.DataFrame(pair_rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    random_state: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Return a percentile bootstrap 95% CI for a user-level macro mean."""
    rng = np.random.default_rng(random_state)
    sample_count = len(values)
    bootstrap_means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = rng.integers(0, sample_count, size=sample_count)
        bootstrap_means[index] = values[sample].mean()
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def summarize_pairs(
    pairs: pd.DataFrame,
    *,
    random_state: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_metrics = pairs.groupby("user_id", sort=False).agg(
        cf_recall_at_k=("cf_recall_at_k", "mean"),
        cf_ndcg_at_k=("cf_ndcg_at_k", "mean"),
        popularity_recall_at_k=("popularity_recall_at_k", "mean"),
        popularity_ndcg_at_k=("popularity_ndcg_at_k", "mean"),
        seed_pairs=("seed_id", "size"),
    )
    rng = np.random.default_rng(random_state)
    rows = []
    for model_name, recall_column, ndcg_column in (
        ("item_cosine", "cf_recall_at_k", "cf_ndcg_at_k"),
        ("popularity", "popularity_recall_at_k", "popularity_ndcg_at_k"),
    ):
        recall = user_metrics[recall_column].to_numpy()
        ndcg = user_metrics[ndcg_column].to_numpy()
        recall_ci = bootstrap_mean_ci(recall, random_state=int(rng.integers(0, 2**32)), resamples=resamples)
        ndcg_ci = bootstrap_mean_ci(ndcg, random_state=int(rng.integers(0, 2**32)), resamples=resamples)
        rows.append(
            {
                "model": model_name,
                "recall@10": recall.mean(),
                "recall_ci_low": recall_ci[0],
                "recall_ci_high": recall_ci[1],
                "ndcg@10": ndcg.mean(),
                "ndcg_ci_low": ndcg_ci[0],
                "ndcg_ci_high": ndcg_ci[1],
                "users_evaluated": len(user_metrics),
                "seed_target_pairs": len(pairs),
            }
        )
    user_metrics["recall_difference"] = (
        user_metrics["cf_recall_at_k"] - user_metrics["popularity_recall_at_k"]
    )
    user_metrics["ndcg_difference"] = (
        user_metrics["cf_ndcg_at_k"] - user_metrics["popularity_ndcg_at_k"]
    )
    recall_delta_ci = bootstrap_mean_ci(
        user_metrics["recall_difference"].to_numpy(),
        random_state=int(rng.integers(0, 2**32)),
        resamples=resamples,
    )
    ndcg_delta_ci = bootstrap_mean_ci(
        user_metrics["ndcg_difference"].to_numpy(),
        random_state=int(rng.integers(0, 2**32)),
        resamples=resamples,
    )
    deltas = pd.DataFrame(
        [
            {"metric": "recall@10", "item_cosine_minus_popularity": user_metrics["recall_difference"].mean(), "ci_low": recall_delta_ci[0], "ci_high": recall_delta_ci[1]},
            {"metric": "ndcg@10", "item_cosine_minus_popularity": user_metrics["ndcg_difference"].mean(), "ci_low": ndcg_delta_ci[0], "ci_high": ndcg_delta_ci[1]},
        ]
    )
    return pd.DataFrame(rows), deltas


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[f"{value:.6f}" if isinstance(value, float) else str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[column]), *(len(row[column]) for row in rows)) for column in range(len(headers))]
    render = lambda row: "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
    return "\n".join([render(headers), render(["-" * width for width in widths])] + [render(row) for row in rows])


def main() -> None:
    ratings = load_dataset(
        PROJECT_ROOT / "data" / "anime.csv",
        PROJECT_ROOT / "data" / "rating.csv",
        max_items=750,
        min_item_ratings=20,
    )
    catalog = ratings[["anime_id", "title"]].drop_duplicates("anime_id")
    train, test = split_user_holdout(
        ratings,
        random_state=HOLDOUT_SEED,
        min_user_ratings=2,
        relevance_threshold=7,
    )
    model = build_similarity_model(train, catalog)
    pairs = make_seed_target_pairs(train, test)
    pair_results = evaluate_seed_target_pairs(model, train, pairs, k=K)
    metrics, deltas = summarize_pairs(pair_results)
    user_metrics = pair_results.groupby("user_id", sort=False).agg(
        cf_recall_at_k=("cf_recall_at_k", "mean"),
        cf_ndcg_at_k=("cf_ndcg_at_k", "mean"),
        popularity_recall_at_k=("popularity_recall_at_k", "mean"),
        popularity_ndcg_at_k=("popularity_ndcg_at_k", "mean"),
        seed_target_pairs=("seed_id", "size"),
    ).reset_index()

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "protocol1_metrics.csv", index=False)
    deltas.to_csv(RESULTS_DIR / "protocol1_paired_deltas.csv", index=False)
    pair_results.to_csv(RESULTS_DIR / "protocol1_seed_target_pairs.csv", index=False)
    user_metrics.to_csv(RESULTS_DIR / "protocol1_user_metrics.csv", index=False)

    summary = f"""# Positive Seed-Target Protocol Results

## Configuration

- Dataset: Kaggle `CooperUnion/anime-recommendations-database`.
- Catalog: 750 most-rated anime; minimum 20 valid ratings per title.
- Cleaned interactions: {len(ratings):,}; training interactions: {len(train):,}.
- Relevance threshold: rating >= 7; one randomly held-out positive target per eligible user.
- Seed sampling: up to {MAX_SEEDS_PER_USER} distinct positive training titles per user; pair-level results macro-averaged within user.
- Split seed: {HOLDOUT_SEED}; seed-sampling seed: {SEED_SAMPLING_SEED}.
- Candidate set: all catalog anime not present in that user's training history; same exclusion for both methods.
- Item-CF: raw cosine; popularity: training interaction count.
- Bootstrap: paired user-level percentile intervals, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}.
- Eligible users: {user_metrics['user_id'].nunique():,}; evaluated seed-target pairs: {len(pair_results):,}.

## Results

{_markdown_table(metrics)}

## Paired user-level differences (item cosine minus popularity)

{_markdown_table(deltas)}

Each target is excluded from model training by the fixed holdout. Users without a positive training seed are omitted and counted only if they contribute a pair. Pair metrics are averaged within each user before computing macro means and bootstrap intervals, so users with two seeds do not receive twice the overall weight.

The Day 2 single-seed scores are retained as historical results from a flawed, unrelated seed-target protocol and should not be compared numerically with this run.
"""
    (RESULTS_DIR / "protocol1_summary.md").write_text(summary, encoding="utf-8")
    print(f"Eligible users: {user_metrics['user_id'].nunique():,}")
    print(f"Seed-target pairs: {len(pair_results):,}")
    print(f"Split seed: {HOLDOUT_SEED}; seed sampling: {SEED_SAMPLING_SEED}; bootstrap: {BOOTSTRAP_SEED}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("Paired differences (item cosine - popularity):")
    print(deltas.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()