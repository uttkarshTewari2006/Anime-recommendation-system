from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ItemSimilarityModel:
    anime_ids: list[int]
    titles: dict[int, str]
    title_to_ids: dict[str, list[int]]
    user_ids: list[Any]
    user_item_matrix: csr_matrix
    similarities: csr_matrix
    id_to_index: dict[int, int]


def load_dataset(
    anime_path: str | Path = "data/anime.csv",
    ratings_path: str | Path = "data/rating.csv",
    *,
    max_items: int = 750,
    min_item_ratings: int = 20,
) -> pd.DataFrame:
    """Load, validate, clean, and popularity-limit MyAnimeList data."""
    if max_items < 1:
        raise ValueError("max_items must be at least 1")
    if min_item_ratings < 1:
        raise ValueError("min_item_ratings must be at least 1")

    anime_path = Path(anime_path)
    ratings_path = Path(ratings_path)
    if not anime_path.is_file() or not ratings_path.is_file():
        raise FileNotFoundError(
            "Dataset files not found. Place anime.csv and rating.csv in data/ "
            f"or pass their paths (checked: {anime_path}, {ratings_path})."
        )

    anime = pd.read_csv(anime_path)
    ratings = pd.read_csv(ratings_path, usecols=lambda column: column in {"user_id", "anime_id", "rating"})
    if "title" not in anime.columns and "name" in anime.columns:
        anime = anime.rename(columns={"name": "title"})

    required_anime = {"anime_id", "title"}
    required_ratings = {"user_id", "anime_id", "rating"}
    missing_anime = required_anime - set(anime.columns)
    missing_ratings = required_ratings - set(ratings.columns)
    if missing_anime or missing_ratings:
        raise ValueError(
            f"Invalid dataset schema. Missing anime columns: {sorted(missing_anime)}; "
            f"missing rating columns: {sorted(missing_ratings)}."
        )

    for column in ("genre", "synopsis"):
        if column not in anime.columns:
            anime[column] = ""
    anime = anime[["anime_id", "title", "genre", "synopsis"]].copy()
    anime["anime_id"] = pd.to_numeric(anime["anime_id"], errors="coerce")
    anime["title"] = anime["title"].astype("string").str.strip()
    anime = anime.dropna(subset=["anime_id", "title"])
    anime = anime[anime["title"].ne("")]
    anime["anime_id"] = anime["anime_id"].astype("int64")
    anime = anime.drop_duplicates("anime_id", keep="first")

    ratings["anime_id"] = pd.to_numeric(ratings["anime_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "anime_id", "rating"])
    ratings["anime_id"] = ratings["anime_id"].astype("int64")
    ratings = ratings[ratings["rating"].between(1, 10)]
    ratings = ratings[ratings["anime_id"].isin(anime["anime_id"])]
    ratings = ratings.groupby(["user_id", "anime_id"], as_index=False)["rating"].mean()

    item_counts = ratings.groupby("anime_id").size()
    popular_ids = (
        item_counts[item_counts >= min_item_ratings]
        .sort_values(ascending=False, kind="stable")
        .head(max_items)
        .index
    )
    ratings = ratings[ratings["anime_id"].isin(popular_ids)]
    if ratings.empty:
        raise ValueError(
            "No ratings remain after cleaning. Check the source schema and lower "
            "min_item_ratings if the dataset is small."
        )

    return ratings.merge(anime, on="anime_id", how="inner", validate="many_to_one")


def split_user_holdout(
    ratings: pd.DataFrame,
    *,
    random_state: int = 42,
    min_user_ratings: int = 2,
    relevance_threshold: float = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out one positive interaction for each eligible user."""
    if min_user_ratings < 2:
        raise ValueError("min_user_ratings must be at least 2")
    rng = np.random.default_rng(random_state)
    held_out_indices: list[Any] = []
    for _, group in ratings.groupby("user_id", sort=False):
        if len(group) >= min_user_ratings:
            positive_ratings = group[group["rating"] >= relevance_threshold]
            if not positive_ratings.empty:
                held_out_indices.append(rng.choice(positive_ratings.index.to_numpy()))

    test = ratings.loc[held_out_indices].copy()
    train = ratings.drop(index=held_out_indices).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_similarity_model(
    train_ratings: pd.DataFrame,
    catalog: pd.DataFrame | None = None,
) -> ItemSimilarityModel:
    """Build sparse item-item cosine similarities from explicit ratings."""
    if train_ratings.empty:
        raise ValueError("train_ratings must contain at least one interaction")

    if catalog is None:
        catalog = train_ratings[["anime_id", "title"]].drop_duplicates("anime_id")
    else:
        catalog = catalog[["anime_id", "title"]].drop_duplicates("anime_id")

    anime_ids = catalog["anime_id"].astype(int).tolist()
    id_to_index = {anime_id: index for index, anime_id in enumerate(anime_ids)}
    known_ratings = train_ratings[train_ratings["anime_id"].isin(id_to_index)].copy()
    user_codes, user_ids = pd.factorize(known_ratings["user_id"], sort=True)
    item_codes = known_ratings["anime_id"].map(id_to_index).to_numpy()
    user_item = csr_matrix(
        (known_ratings["rating"].astype(float), (user_codes, item_codes)),
        shape=(len(pd.unique(known_ratings["user_id"])), len(anime_ids)),
    )
    item_user = user_item.T.tocsr()
    similarities = csr_matrix(cosine_similarity(item_user, dense_output=False))

    titles = dict(zip(catalog["anime_id"].astype(int), catalog["title"].astype(str)))
    title_to_ids: dict[str, list[int]] = {}
    for anime_id, title in titles.items():
        title_to_ids.setdefault(title.casefold(), []).append(anime_id)

    return ItemSimilarityModel(
        anime_ids=anime_ids,
        titles=titles,
        title_to_ids=title_to_ids,
        user_ids=user_ids.tolist(),
        user_item_matrix=user_item,
        similarities=similarities,
        id_to_index=id_to_index,
    )


def recommend_for_anime(
    model: ItemSimilarityModel,
    title: str,
    *,
    k: int = 10,
    exclude_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Return the highest-similarity anime for an exact title match."""
    if k < 1:
        raise ValueError("k must be at least 1")
    matches = model.title_to_ids.get(title.strip().casefold(), [])
    if not matches:
        raise KeyError(f"Anime title not found: {title}")

    source_id = matches[0]
    source_index = model.id_to_index[source_id]
    row = model.similarities.getrow(source_index)
    excluded = set(exclude_ids or ()) | {source_id}
    ranked = sorted(
        (
            (model.anime_ids[index], float(score))
            for index, score in zip(row.indices, row.data)
            if model.anime_ids[index] not in excluded
        ),
        key=lambda item: (-item[1], item[0]),
    )[:k]
    return [
        {"anime_id": anime_id, "title": model.titles[anime_id], "similarity": score}
        for anime_id, score in ranked
    ]


def evaluate_holdout(
    model: ItemSimilarityModel,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    *,
    k: int = 10,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate one seeded item recommendation per held-out user interaction."""
    if k < 1:
        raise ValueError("k must be at least 1")

    train_by_user = {user_id: group for user_id, group in train_ratings.groupby("user_id", sort=False)}
    rows: list[dict[str, Any]] = []
    for held_out in test_ratings.itertuples(index=False):
        user_history = train_by_user.get(held_out.user_id)
        if user_history is None or user_history.empty:
            continue
        seed = user_history.iloc[0]
        recommendations = recommend_for_anime(model, seed.title, k=k)
        recommended_ids = [result["anime_id"] for result in recommendations]
        relevant_id = int(str(held_out.anime_id))
        try:
            rank = recommended_ids.index(relevant_id) + 1
        except ValueError:
            rank = None
        rows.append(
            {
                "user_id": held_out.user_id,
                "seed_title": seed.title,
                "held_out_title": held_out.title,
                "rank": rank,
                "recall_at_k": int(rank is not None),
                "ndcg_at_k": 1.0 / np.log2(rank + 1) if rank is not None else 0.0,
            }
        )

    details = pd.DataFrame(rows)
    if details.empty:
        metrics: dict[str, float | int] = {f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, "users_evaluated": 0}
    else:
        metrics = {
            f"recall@{k}": float(details["recall_at_k"].mean()),
            f"ndcg@{k}": float(details["ndcg_at_k"].mean()),
            "users_evaluated": int(len(details)),
        }
    return metrics, details


def evaluate_popularity_holdout(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    *,
    k: int = 10,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate top-rated-count anime for the same seed/target holdout cases."""
    if k < 1:
        raise ValueError("k must be at least 1")

    popular_ids = (
        train_ratings.groupby("anime_id").size()
        .sort_values(ascending=False, kind="stable")
        .index.astype(int)
        .tolist()
    )
    train_by_user = {user_id: group for user_id, group in train_ratings.groupby("user_id", sort=False)}
    rows: list[dict[str, Any]] = []
    for held_out in test_ratings.itertuples(index=False):
        user_history = train_by_user.get(held_out.user_id)
        if user_history is None or user_history.empty:
            continue
        seed = user_history.iloc[0]
        candidates = [anime_id for anime_id in popular_ids if anime_id != int(seed.anime_id)][:k]
        relevant_id = int(str(held_out.anime_id))
        try:
            rank = candidates.index(relevant_id) + 1
        except ValueError:
            rank = None
        rows.append(
            {
                "user_id": held_out.user_id,
                "seed_title": seed.title,
                "held_out_title": held_out.title,
                "rank": rank,
                "recall_at_k": int(rank is not None),
                "ndcg_at_k": 1.0 / np.log2(rank + 1) if rank is not None else 0.0,
            }
        )

    details = pd.DataFrame(rows)
    if details.empty:
        metrics: dict[str, float | int] = {f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, "users_evaluated": 0}
    else:
        metrics = {
            f"recall@{k}": float(details["recall_at_k"].mean()),
            f"ndcg@{k}": float(details["ndcg_at_k"].mean()),
            "users_evaluated": int(len(details)),
        }
    return metrics, details