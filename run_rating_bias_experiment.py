from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from anime_recommender import build_similarity_model, load_dataset, recommend_for_anime
from run_catalog_size_evaluation import CATALOG_SIZE, MIN_ITEM_RATINGS, apply_fixed_targets
from run_protocol1_evaluation import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    K,
    bootstrap_mean_ci,
    evaluate_seed_target_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
SEED_TITLES = (
    "Fullmetal Alchemist: Brotherhood",
    "Naruto",
    "Death Note",
    "Sword Art Online",
    "Neon Genesis Evangelion",
    "Clannad",
    "Highschool of the Dead",
    "Trigun",
)


def _user_metrics(scored_pairs: pd.DataFrame) -> pd.DataFrame:
    return scored_pairs.groupby("user_id", sort=False).agg(
        cf_recall_at_k=("cf_recall_at_k", "mean"),
        cf_ndcg_at_k=("cf_ndcg_at_k", "mean"),
        popularity_recall_at_k=("popularity_recall_at_k", "mean"),
        popularity_ndcg_at_k=("popularity_ndcg_at_k", "mean"),
    )


def _metric_row(
    name: str,
    recall: np.ndarray,
    ndcg: np.ndarray,
    *,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    recall_ci = bootstrap_mean_ci(
        recall,
        random_state=int(rng.integers(0, 2**32)),
        resamples=BOOTSTRAP_RESAMPLES,
    )
    ndcg_ci = bootstrap_mean_ci(
        ndcg,
        random_state=int(rng.integers(0, 2**32)),
        resamples=BOOTSTRAP_RESAMPLES,
    )
    return {
        "model": name,
        "recall@10": recall.mean(),
        "recall_ci_low": recall_ci[0],
        "recall_ci_high": recall_ci[1],
        "ndcg@10": ndcg.mean(),
        "ndcg_ci_low": ndcg_ci[0],
        "ndcg_ci_high": ndcg_ci[1],
        "users_evaluated": len(recall),
    }


def _neighbor_rows(model: object, model_name: str) -> list[dict[str, object]]:
    rows = []
    for seed_title in SEED_TITLES:
        recommendations = recommend_for_anime(
            model,
            seed_title,
            k=10,
            genre_weight=0.0,
        )
        for rank, recommendation in enumerate(recommendations, start=1):
            rows.append(
                {
                    "model": model_name,
                    "seed_title": seed_title,
                    "rank": rank,
                    "neighbor_title": recommendation["title"],
                    "similarity": recommendation["similarity"],
                }
            )
    return rows


def _neighbor_markdown(neighbors: pd.DataFrame) -> str:
    sections = ["# Fixed-Title Neighbor Review", ""]
    for seed_title in SEED_TITLES:
        raw = neighbors[
            (neighbors["seed_title"] == seed_title)
            & (neighbors["model"] == "raw_cosine")
        ].sort_values("rank")
        adjusted = neighbors[
            (neighbors["seed_title"] == seed_title)
            & (neighbors["model"] == "adjusted_cosine")
        ].sort_values("rank")
        rows = []
        raw_by_rank = {row.rank: row for row in raw.itertuples(index=False)}
        adjusted_by_rank = {row.rank: row for row in adjusted.itertuples(index=False)}
        for rank in range(1, 11):
            raw_row = raw_by_rank.get(rank)
            adjusted_row = adjusted_by_rank.get(rank)
            raw_display = (
                f"{raw_row.neighbor_title} ({raw_row.similarity:.4f})"
                if raw_row is not None
                else ""
            )
            adjusted_display = (
                f"{adjusted_row.neighbor_title} ({adjusted_row.similarity:.4f})"
                if adjusted_row is not None
                else ""
            )
            rows.append(
                f"| {rank} | {raw_display} | {adjusted_display} |"
            )
        sections.extend(
            [
                f"## {seed_title}",
                "",
                "| Rank | Raw cosine | Adjusted cosine |",
                "|---:|---|---|",
                *rows,
                "",
            ]
        )
    return "\n".join(sections)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [
        [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    render = lambda row: "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(row)
    ) + " |"
    return "\n".join(
        [render(headers), render(["-" * width for width in widths])]
        + [render(row) for row in rows]
    )


def main() -> None:
    ratings = load_dataset(
        PROJECT_ROOT / "data" / "anime.csv",
        PROJECT_ROOT / "data" / "rating.csv",
        max_items=CATALOG_SIZE,
        min_item_ratings=MIN_ITEM_RATINGS,
    )
    seed_target_pairs = pd.read_csv(RESULTS_DIR / "protocol1_seed_target_pairs.csv")
    train, test = apply_fixed_targets(ratings, seed_target_pairs)
    catalog = ratings[["anime_id", "title"]].drop_duplicates("anime_id")
    available_titles = set(catalog["title"].astype(str))
    missing_titles = set(SEED_TITLES) - available_titles
    if missing_titles:
        raise ValueError(f"Fixed neighbor-review titles missing from catalog: {sorted(missing_titles)}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    raw_start = perf_counter()
    raw_model = build_similarity_model(
        train, catalog, similarity_method="raw_cosine"
    )
    raw_scored = evaluate_seed_target_pairs(raw_model, train, seed_target_pairs, k=K)
    raw_seconds = perf_counter() - raw_start

    adjusted_start = perf_counter()
    adjusted_model = build_similarity_model(
        train, catalog, similarity_method="adjusted_cosine"
    )
    adjusted_scored = evaluate_seed_target_pairs(
        adjusted_model, train, seed_target_pairs, k=K
    )
    adjusted_seconds = perf_counter() - adjusted_start

    raw_users = _user_metrics(raw_scored)
    adjusted_users = _user_metrics(adjusted_scored)
    user_metrics = raw_users.join(
        adjusted_users[["cf_recall_at_k", "cf_ndcg_at_k"]],
        lsuffix="_raw",
        rsuffix="_adjusted",
        how="inner",
    )
    user_metrics["popularity_recall_at_k"] = raw_users["popularity_recall_at_k"]
    user_metrics["popularity_ndcg_at_k"] = raw_users["popularity_ndcg_at_k"]
    user_metrics = user_metrics.rename_axis("user_id").reset_index()

    raw_recall = user_metrics["cf_recall_at_k_raw"].to_numpy()
    raw_ndcg = user_metrics["cf_ndcg_at_k_raw"].to_numpy()
    adjusted_recall = user_metrics["cf_recall_at_k_adjusted"].to_numpy()
    adjusted_ndcg = user_metrics["cf_ndcg_at_k_adjusted"].to_numpy()
    popularity_recall = user_metrics["popularity_recall_at_k"].to_numpy()
    popularity_ndcg = user_metrics["popularity_ndcg_at_k"].to_numpy()

    metric_rows = [
        _metric_row("raw_cosine", raw_recall, raw_ndcg, rng=rng),
        _metric_row("adjusted_cosine", adjusted_recall, adjusted_ndcg, rng=rng),
        _metric_row("popularity", popularity_recall, popularity_ndcg, rng=rng),
    ]
    deltas = []
    for metric_name, adjusted_values, raw_values in (
        ("recall@10", adjusted_recall, raw_recall),
        ("ndcg@10", adjusted_ndcg, raw_ndcg),
    ):
        difference = adjusted_values - raw_values
        interval = bootstrap_mean_ci(
            difference,
            random_state=int(rng.integers(0, 2**32)),
            resamples=BOOTSTRAP_RESAMPLES,
        )
        deltas.append(
            {
                "metric": metric_name,
                "adjusted_minus_raw": difference.mean(),
                "ci_low": interval[0],
                "ci_high": interval[1],
            }
        )
    metric_table = pd.DataFrame(metric_rows)
    delta_table = pd.DataFrame(deltas)

    pair_details = raw_scored[
        ["user_id", "target_id", "seed_id", "seed_number", "cf_rank", "cf_recall_at_k", "cf_ndcg_at_k", "popularity_rank"]
    ].merge(
        adjusted_scored[
            ["user_id", "target_id", "seed_id", "seed_number", "cf_rank", "cf_recall_at_k", "cf_ndcg_at_k"]
        ],
        on=["user_id", "target_id", "seed_id", "seed_number"],
        suffixes=("_raw", "_adjusted"),
        validate="one_to_one",
    )
    neighbor_rows = _neighbor_rows(raw_model, "raw_cosine") + _neighbor_rows(
        adjusted_model, "adjusted_cosine"
    )
    neighbors = pd.DataFrame(neighbor_rows)
    neighbors_markdown = _neighbor_markdown(neighbors)

    RESULTS_DIR.mkdir(exist_ok=True)
    metric_table.to_csv(RESULTS_DIR / "rating_bias_metrics.csv", index=False)
    delta_table.to_csv(RESULTS_DIR / "rating_bias_deltas.csv", index=False)
    pair_details.to_csv(RESULTS_DIR / "rating_bias_pair_details.csv", index=False)
    user_metrics.to_csv(RESULTS_DIR / "rating_bias_user_metrics.csv", index=False)
    neighbors.to_csv(RESULTS_DIR / "rating_bias_neighbors.csv", index=False)
    (RESULTS_DIR / "rating_bias_neighbors.md").write_text(
        neighbors_markdown, encoding="utf-8"
    )

    summary = f"""# Rating-Habit Bias Experiment

## Configuration

- Catalog: {len(catalog):,} titles; minimum {MIN_ITEM_RATINGS} ratings.
- Reused targets/users: {len(test):,}; seed-target pairs: {len(raw_scored):,}.
- Holdout and pair seeds: fixed from item 1 (42 / 2026); bootstrap seed {BOOTSTRAP_SEED}, {BOOTSTRAP_RESAMPLES} resamples.
- Adjusted cosine: subtract each user's mean observed training rating before item-item cosine.
- Candidate exclusions, top K = {K}, and user-level pair averaging are unchanged.
- Neighbor review: {len(SEED_TITLES)} fixed seed titles, raw and adjusted top-10 lists in `results/rating_bias_neighbors.md`.

## Metrics

{_markdown_table(metric_table)}

## Paired adjusted-minus-raw change

{_markdown_table(delta_table)}

Raw model plus pair evaluation took {raw_seconds:.1f} seconds; adjusted model plus pair evaluation took {adjusted_seconds:.1f} seconds. The popularity reference is unchanged because the training split and user histories are the same.
"""
    (RESULTS_DIR / "rating_bias_summary.md").write_text(summary, encoding="utf-8")
    print(metric_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(delta_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Neighbor review: {RESULTS_DIR / 'rating_bias_neighbors.md'}")


if __name__ == "__main__":
    main()