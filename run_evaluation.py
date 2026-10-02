from __future__ import annotations

from pathlib import Path

import pandas as pd

from anime_recommender import (
    build_similarity_model,
    evaluate_holdout,
    evaluate_popularity_holdout,
    load_dataset,
    split_user_holdout,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
GENRE_WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)
K = 10


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rows.append(
            [
                f"{value:.6f}" if isinstance(value, float) else str(value)
                for value in values
            ]
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    format_row = lambda values: "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(values)
    ) + " |"
    return "\n".join(
        [format_row(headers), format_row(["-" * width for width in widths])]
        + [format_row(row) for row in rows]
    )


def split_validation_test(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create disjoint validation and test positives with one shared train set."""
    training_pool, test = split_user_holdout(ratings, random_state=42)
    train, validation = split_user_holdout(training_pool, random_state=43)

    positive_train_users = set(train.loc[train["rating"] >= 7, "user_id"])
    eligible_users = (
        set(validation["user_id"])
        & set(test["user_id"])
        & positive_train_users
    )
    validation = validation[validation["user_id"].isin(eligible_users)].reset_index(drop=True)
    test = test[test["user_id"].isin(eligible_users)].reset_index(drop=True)
    if validation.empty or test.empty:
        raise ValueError("Not enough users with positive train, validation, and test ratings.")
    return train, validation, test


def main() -> None:
    ratings = load_dataset(
        PROJECT_ROOT / "data" / "anime.csv",
        PROJECT_ROOT / "data" / "rating.csv",
        max_items=750,
        min_item_ratings=20,
    )
    catalog = ratings[["anime_id", "title", "genre"]].drop_duplicates("anime_id")
    train, validation, test = split_validation_test(ratings)
    model = build_similarity_model(train, catalog)

    validation_rows = []
    for genre_weight in GENRE_WEIGHTS:
        metrics, _ = evaluate_holdout(
            model,
            train,
            validation,
            k=K,
            genre_weight=genre_weight,
        )
        validation_rows.append({"genre_weight": genre_weight, **metrics})
    validation_metrics = pd.DataFrame(validation_rows)
    selected_row = max(
        validation_rows,
        key=lambda row: (row[f"recall@{K}"], row[f"ndcg@{K}"], -row["genre_weight"]),
    )
    selected_weight = float(selected_row["genre_weight"])

    collaborative_metrics, collaborative_details = evaluate_holdout(
        model, train, test, k=K, genre_weight=0.0
    )
    baseline_metrics, baseline_details = evaluate_holdout(
        model, train, test, k=K, genre_weight=0.0, use_all_history=False
    )
    hybrid_metrics, hybrid_details = evaluate_holdout(
        model, train, test, k=K, genre_weight=selected_weight
    )
    popularity_metrics, popularity_details = evaluate_popularity_holdout(
        train, test, k=K
    )

    metrics_table = pd.DataFrame(
        [
            {"model": "single_seed_cosine", "genre_weight": 0.0, **baseline_metrics},
            {"model": "history_cosine", "genre_weight": 0.0, **collaborative_metrics},
            {"model": "genre_hybrid", "genre_weight": selected_weight, **hybrid_metrics},
            {"model": "popularity", "genre_weight": 0.0, **popularity_metrics},
        ]
    )
    ranking_details = []
    for label, weight, details in (
        ("single_seed_cosine", 0.0, baseline_details),
        ("history_cosine", 0.0, collaborative_details),
        ("genre_hybrid", selected_weight, hybrid_details),
        ("popularity", 0.0, popularity_details),
    ):
        details = details.copy()
        details.insert(0, "genre_weight", weight)
        details.insert(0, "model", label)
        ranking_details.append(details)
    ranking_table = pd.concat(ranking_details, ignore_index=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    validation_metrics.to_csv(RESULTS_DIR / "day3_validation_metrics.csv", index=False)
    metrics_table.to_csv(RESULTS_DIR / "day3_metrics.csv", index=False)
    ranking_table.to_csv(RESULTS_DIR / "day3_user_rankings.csv", index=False)

    metric_gain = hybrid_metrics[f"recall@{K}"] - baseline_metrics[f"recall@{K}"]
    summary = f"""# Day 3 Recommender Results

## Run configuration

- Catalog: {len(catalog):,} anime; genre metadata populated for {catalog['genre'].fillna('').str.strip().ne('').sum():,}.
- Cleaned interactions: {len(ratings):,}.
- Training interactions: {len(train):,}.
- Disjoint validation/test cases: {len(validation):,} each, with one positive target per user.
- Relevance: explicit rating of at least 7; ranking cutoff: {K}.
- Genre weight selected on validation: {selected_weight:.2f}.

## Validation weight search

{markdown_table(validation_metrics)}

## Final test comparison

{markdown_table(metrics_table)}

The genre hybrid's Recall@{K} change over the original single-seed baseline is {metric_gain:+.6f}. The weight was selected on validation and then evaluated once on the disjoint test set. Genre and collaborative scores are normalized per user before blending; already-rated anime are excluded from every ranked list.

## Interpretation and limitations

The test set contains one held-out positive per eligible user, so Recall@{K} is the share of users whose held-out title appears in the top {K}. NDCG@{K} rewards higher placement. This is a randomized implicit-preference proxy, not a temporal evaluation or an online user study. Results apply to the 750-title popular-item catalog and do not measure cold-start quality.

These Day 3 values are not a direct replay of the historical Day 2 metrics: Day 3 requires disjoint validation and test positives plus positive training history, and every method excludes all titles already rated by each user.

For this dataset and catalog, the validation-selected genre weight of {selected_weight:.2f} is the user-facing recommendation default. If the dataset or catalog changes, rerun validation and update `DEFAULT_GENRE_WEIGHT` to the new selected value. The CSV artifacts preserve per-user ranks for reproducibility.

## Next steps

- Repeat the disjoint split with several random seeds and report mean and spread to check that the hybrid gain is stable.
- Increase the catalog size and report Recall@10 by item popularity to quantify long-tail coverage.
- When timestamped interactions or online feedback are available, add a temporal or user-study evaluation before production use.
"""
    (RESULTS_DIR / "day3_summary.md").write_text(summary, encoding="utf-8")

    print(f"Cleaned interactions: {len(ratings):,}")
    print(f"Training interactions: {len(train):,}")
    print(f"Validation/test users: {len(test):,}")
    print(f"Selected genre weight: {selected_weight:.2f}")
    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Day 3 artifacts saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()