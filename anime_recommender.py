from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

DEFAULT_GENRE_WEIGHT = 0.1


@dataclass
class ItemSimilarityModel:
    anime_ids: list[int]
    titles: dict[int, str]
    title_to_ids: dict[str, list[int]]
    user_ids: list[Any]
    user_item_matrix: csr_matrix
    similarities: csr_matrix
    id_to_index: dict[int, int]
    genre_similarities: csr_matrix | None = None


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
    *,
    similarity_method: str = "raw_cosine",
) -> ItemSimilarityModel:
    """Build item cosine similarities, optionally centering each user's ratings."""
    if train_ratings.empty:
        raise ValueError("train_ratings must contain at least one interaction")
    if similarity_method not in {"raw_cosine", "adjusted_cosine"}:
        raise ValueError("similarity_method must be 'raw_cosine' or 'adjusted_cosine'")

    if catalog is None:
        catalog_columns = ["anime_id", "title"]
        if "genre" in train_ratings.columns:
            catalog_columns.append("genre")
        catalog = train_ratings[catalog_columns].drop_duplicates("anime_id")
    else:
        catalog_columns = ["anime_id", "title"]
        if "genre" in catalog.columns:
            catalog_columns.append("genre")
        catalog = catalog[catalog_columns].drop_duplicates("anime_id")

    anime_ids = catalog["anime_id"].astype(int).tolist()
    id_to_index = {anime_id: index for index, anime_id in enumerate(anime_ids)}
    known_ratings = train_ratings[train_ratings["anime_id"].isin(id_to_index)].copy()
    user_codes, user_ids = pd.factorize(known_ratings["user_id"], sort=True)
    item_codes = known_ratings["anime_id"].map(id_to_index).to_numpy()
    user_item = csr_matrix(
        (known_ratings["rating"].astype(float), (user_codes, item_codes)),
        shape=(len(pd.unique(known_ratings["user_id"])), len(anime_ids)),
    )
    if similarity_method == "adjusted_cosine":
        user_means = known_ratings.groupby("user_id", sort=False)["rating"].transform("mean")
        centered_ratings = known_ratings["rating"].astype(float) - user_means.astype(float)
        similarity_user_item = csr_matrix(
            (centered_ratings.to_numpy(), (user_codes, item_codes)),
            shape=user_item.shape,
        )
        similarity_user_item.eliminate_zeros()
    else:
        similarity_user_item = user_item
    item_user = similarity_user_item.T.tocsr()
    similarities = csr_matrix(cosine_similarity(item_user, dense_output=False))

    genre_similarities = None
    if "genre" in catalog.columns:
        genres = catalog["genre"].fillna("").astype(str).map(
            lambda value: [genre.strip().casefold() for genre in value.split(",") if genre.strip()]
        )
        genre_matrix = csr_matrix(MultiLabelBinarizer().fit_transform(genres))
        if genre_matrix.shape[1]:
            genre_similarities = csr_matrix(cosine_similarity(genre_matrix, dense_output=False))

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
        genre_similarities=genre_similarities,
    )


def _scores_from_sources(
    model: ItemSimilarityModel,
    source_ids: list[int],
    *,
    include_genres: bool,
) -> tuple[np.ndarray, np.ndarray]:
    collaborative_scores = np.zeros(len(model.anime_ids), dtype=float)
    genre_scores = np.zeros(len(model.anime_ids), dtype=float)
    for anime_id in source_ids:
        source_index = model.id_to_index.get(anime_id)
        if source_index is None:
            continue
        row = model.similarities.getrow(source_index)
        collaborative_scores[row.indices] += row.data
        if include_genres and model.genre_similarities is not None:
            genre_row = model.genre_similarities.getrow(source_index)
            genre_scores[genre_row.indices] += genre_row.data
    return collaborative_scores, genre_scores


def _combined_scores(
    model: ItemSimilarityModel,
    source_ids: list[int],
    genre_weight: float,
) -> np.ndarray:
    if not np.isfinite(genre_weight) or not 0 <= genre_weight <= 1:
        raise ValueError("genre_weight must be between 0 and 1")
    collaborative_scores, genre_scores = _scores_from_sources(
        model, source_ids, include_genres=genre_weight > 0
    )

    if genre_weight and genre_scores.max() > 0 and collaborative_scores.max() > 0:
        return (
            (1 - genre_weight) * collaborative_scores / collaborative_scores.max()
            + genre_weight * genre_scores / genre_scores.max()
        )
    if genre_scores.max() > 0 and collaborative_scores.max() == 0:
        return genre_scores
    return collaborative_scores


def _recommend_from_sources(
    model: ItemSimilarityModel,
    source_ids: list[int],
    *,
    k: int,
    exclude_ids: set[int],
    genre_weight: float,
) -> list[dict[str, Any]]:
    if k < 1:
        raise ValueError("k must be at least 1")
    scores = _combined_scores(model, source_ids, genre_weight)

    candidate_indices = np.flatnonzero(scores > 0)
    for anime_id in exclude_ids:
        index = model.id_to_index.get(anime_id)
        if index is not None:
            candidate_indices = candidate_indices[candidate_indices != index]
    if len(candidate_indices) > k:
        cutoff = np.partition(scores[candidate_indices], -k)[-k]
        above_cutoff = candidate_indices[scores[candidate_indices] > cutoff]
        tied_at_cutoff = candidate_indices[scores[candidate_indices] == cutoff]
        tie_order = np.argsort(np.asarray(model.anime_ids)[tied_at_cutoff])
        candidate_indices = np.concatenate(
            [above_cutoff, tied_at_cutoff[tie_order[: k - len(above_cutoff)]]]
        )
    ranked_indices = sorted(
        candidate_indices.tolist(),
        key=lambda index: (-scores[index], model.anime_ids[index]),
    )
    return [
        {
            "anime_id": model.anime_ids[index],
            "title": model.titles[model.anime_ids[index]],
            "similarity": float(scores[index]),
        }
        for index in ranked_indices
    ]


def recommend_for_anime(
    model: ItemSimilarityModel,
    title: str,
    *,
    k: int = 10,
    exclude_ids: set[int] | None = None,
    genre_weight: float = DEFAULT_GENRE_WEIGHT,
) -> list[dict[str, Any]]:
    """Return the highest-similarity anime for an exact title match."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not np.isfinite(genre_weight) or not 0 <= genre_weight <= 1:
        raise ValueError("genre_weight must be between 0 and 1")
    matches = model.title_to_ids.get(title.strip().casefold(), [])
    if not matches:
        raise KeyError(f"Anime title not found: {title}")

    source_id = matches[0]
    excluded = set(exclude_ids or ()) | {source_id}
    return _recommend_from_sources(
        model,
        [source_id],
        k=k,
        exclude_ids=excluded,
        genre_weight=genre_weight,
    )


def recommend_from_history(
    model: ItemSimilarityModel,
    user_history: pd.DataFrame,
    *,
    k: int = 10,
    relevance_threshold: float = 7,
    genre_weight: float = DEFAULT_GENRE_WEIGHT,
) -> list[dict[str, Any]]:
    """Rank unseen anime by summed similarity to positively rated history."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not np.isfinite(genre_weight) or not 0 <= genre_weight <= 1:
        raise ValueError("genre_weight must be between 0 and 1")
    positive_history = user_history[user_history["rating"] >= relevance_threshold]
    if positive_history.empty:
        return []

    return _recommend_from_sources(
        model,
        positive_history["anime_id"].astype(int).tolist(),
        k=k,
        exclude_ids=set(user_history["anime_id"].astype(int)),
        genre_weight=genre_weight,
    )


def evaluate_holdout(
    model: ItemSimilarityModel,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    *,
    k: int = 10,
    genre_weight: float = 0.0,
    use_all_history: bool = True,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate history-based recommendations with batched sparse scoring."""
    if k < 1:
        raise ValueError("k must be at least 1")

    known_train = train_ratings[
        train_ratings["anime_id"].isin(model.id_to_index)
    ].drop_duplicates(["user_id", "anime_id"], keep="first")
    user_ids = pd.unique(known_train["user_id"])
    user_to_row = {user_id: row for row, user_id in enumerate(user_ids)}
    row_indices = known_train["user_id"].map(user_to_row).to_numpy(dtype=np.int32)
    column_indices = known_train["anime_id"].map(model.id_to_index).to_numpy(dtype=np.int32)
    shape = (len(user_ids), len(model.anime_ids))
    seen_matrix = csr_matrix(
        (np.ones(len(known_train), dtype=np.float32), (row_indices, column_indices)),
        shape=shape,
    )

    positive_train = known_train[known_train["rating"] >= 7]
    first_train = known_train.drop_duplicates("user_id", keep="first")
    source_records = positive_train if use_all_history else first_train
    seed_records = (
        positive_train.drop_duplicates("user_id", keep="first")
        if use_all_history
        else first_train
    )
    source_rows = source_records["user_id"].map(user_to_row).to_numpy(dtype=np.int32)
    source_columns = source_records["anime_id"].map(model.id_to_index).to_numpy(dtype=np.int32)
    source_matrix = csr_matrix(
        (np.ones(len(source_records), dtype=np.float32), (source_rows, source_columns)),
        shape=shape,
    )

    known_test = test_ratings[
        test_ratings["user_id"].isin(user_to_row)
        & test_ratings["anime_id"].isin(model.id_to_index)
    ].reset_index(drop=True)
    test_user_rows = known_test["user_id"].map(user_to_row).to_numpy(dtype=np.int32)
    target_indices = known_test["anime_id"].map(model.id_to_index).to_numpy(dtype=np.int32)
    seed_titles = seed_records.set_index("user_id")["title"].to_dict()
    fallback_titles = (
        known_train.drop_duplicates("user_id").set_index("user_id")["title"].to_dict()
    )
    anime_ids = np.asarray(model.anime_ids)
    rows: list[dict[str, Any]] = []
    batch_size = 512

    for start in range(0, len(known_test), batch_size):
        stop = min(start + batch_size, len(known_test))
        user_rows = test_user_rows[start:stop]
        scores = (source_matrix[user_rows] @ model.similarities).toarray()
        if genre_weight and model.genre_similarities is not None:
            genre_scores = (
                source_matrix[user_rows] @ model.genre_similarities
            ).toarray()
            collaborative_max = scores.max(axis=1, keepdims=True)
            genre_max = genre_scores.max(axis=1, keepdims=True)
            both_available = (collaborative_max > 0) & (genre_max > 0)
            blended = np.zeros_like(scores)
            np.divide(scores, collaborative_max, out=blended, where=collaborative_max > 0)
            blended *= 1 - genre_weight
            normalized_genres = np.zeros_like(genre_scores)
            np.divide(genre_scores, genre_max, out=normalized_genres, where=genre_max > 0)
            blended += genre_weight * normalized_genres
            scores = np.where(both_available, blended, scores)

        for offset, test_row in enumerate(known_test.iloc[start:stop].itertuples(index=False)):
            target_index = target_indices[start + offset]
            candidate_mask = scores[offset] > 0
            seen_indices = seen_matrix.indices[
                seen_matrix.indptr[user_rows[offset]] : seen_matrix.indptr[user_rows[offset] + 1]
            ]
            candidate_mask[seen_indices] = False
            target_score = scores[offset, target_index]
            if not candidate_mask[target_index]:
                rank = None
            else:
                precedes_target = (scores[offset] > target_score) | (
                    (scores[offset] == target_score) & (anime_ids < int(test_row.anime_id))
                )
                rank = int(np.count_nonzero(candidate_mask & precedes_target) + 1)
                if rank > k:
                    rank = None
            rows.append(
                {
                    "user_id": test_row.user_id,
                    "seed_title": seed_titles.get(
                        test_row.user_id, fallback_titles.get(test_row.user_id, "")
                    ),
                    "held_out_title": test_row.title,
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
        positive_history = user_history[user_history["rating"] >= 7]
        seed = positive_history.iloc[0] if not positive_history.empty else user_history.iloc[0]
        seen_ids = set(user_history["anime_id"].astype(int))
        candidates = [anime_id for anime_id in popular_ids if anime_id not in seen_ids][:k]
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