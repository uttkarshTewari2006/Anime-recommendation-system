import tempfile
import unittest
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix

from anime_recommender import (
    build_similarity_model,
    evaluate_holdout,
    evaluate_popularity_holdout,
    load_dataset,
    recommend_for_anime,
    recommend_from_history,
    split_user_holdout,
)
from run_evaluation import split_validation_test
from run_catalog_size_evaluation import apply_fixed_targets
from run_protocol1_evaluation import evaluate_seed_target_pairs, make_seed_target_pairs


class AnimeRecommenderTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame(
            {
                "anime_id": [1, 2, 3, 4],
                "title": ["Alpha", "Beta", "Gamma", "Delta"],
                "genre": ["Action, Fantasy", "Action", "Action, Fantasy", "Sports"],
            }
        )
        self.ratings = pd.DataFrame(
            [
                ("u1", 1, 10), ("u1", 2, 9), ("u1", 3, 2),
                ("u2", 1, 9), ("u2", 2, 8), ("u2", 4, 3),
                ("u3", 1, 8), ("u3", 2, 10), ("u3", 3, 4),
                ("u4", 3, 9), ("u4", 4, 8), ("u4", 1, 2),
            ],
            columns=["user_id", "anime_id", "rating"],
        ).merge(self.catalog, on="anime_id")

    def test_load_dataset_cleans_invalid_and_duplicate_ratings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"anime_id": [1, 2], "name": [" Alpha ", "Beta"]}).to_csv(root / "anime.csv", index=False)
            pd.DataFrame(
                [(1, 1, 8), (1, 1, 10), (2, 1, -1), (3, 2, 11)],
                columns=["user_id", "anime_id", "rating"],
            ).to_csv(root / "rating.csv", index=False)

            result = load_dataset(root / "anime.csv", root / "rating.csv", min_item_ratings=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["rating"], 9)
        self.assertEqual(result.iloc[0]["title"], "Alpha")

    def test_similarity_and_recommendation_order(self):
        model = build_similarity_model(self.ratings, self.catalog)
        recommendations = recommend_for_anime(model, "alpha", k=2)

        self.assertEqual(model.user_item_matrix.shape, (4, 4))
        self.assertEqual(model.similarities.shape, (4, 4))
        self.assertEqual(recommendations[0]["title"], "Beta")
        self.assertNotIn("Alpha", [result["title"] for result in recommendations])
        self.assertGreaterEqual(recommendations[0]["similarity"], recommendations[1]["similarity"])

    def test_model_uses_genres_from_ratings_without_explicit_catalog(self):
        model = build_similarity_model(self.ratings)

        self.assertIsNotNone(model.genre_similarities)

    def test_history_recommendations_sum_similarities_and_exclude_seen_items(self):
        model = build_similarity_model(self.ratings, self.catalog)
        model.similarities = csr_matrix(
            [
                [1.0, 0.0, 0.6, 0.9],
                [0.0, 1.0, 0.6, 0.0],
                [0.6, 0.6, 1.0, 0.0],
                [0.9, 0.9, 0.0, 1.0],
            ]
        )
        history = self.ratings[
            (self.ratings["user_id"] == "u1") & self.ratings["anime_id"].isin([1, 2])
        ]

        recommendations = recommend_from_history(model, history, k=2, genre_weight=0.0)

        self.assertEqual(recommendations[0]["anime_id"], 3)
        self.assertAlmostEqual(recommendations[0]["similarity"], 1.2)
        self.assertNotIn(1, [result["anime_id"] for result in recommendations])
        self.assertNotIn(2, [result["anime_id"] for result in recommendations])

    def test_genre_blend_reranks_collaborative_candidates(self):
        model = build_similarity_model(self.ratings, self.catalog)
        model.similarities = csr_matrix(
            [
                [1.0, 0.9, 0.7, 0.0],
                [0.9, 1.0, 0.0, 0.0],
                [0.7, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        baseline = recommend_for_anime(model, "Alpha", k=2)
        hybrid = recommend_for_anime(model, "Alpha", k=2, genre_weight=0.5)

        self.assertEqual(baseline[0]["anime_id"], 2)
        self.assertEqual(hybrid[0]["anime_id"], 3)

    def test_holdout_evaluation_is_bounded(self):
        train, test = split_user_holdout(self.ratings, random_state=7)
        model = build_similarity_model(train, self.catalog)
        metrics, details = evaluate_holdout(model, train, test, k=2)

        self.assertEqual(len(test), 4)
        self.assertTrue(test["rating"].ge(7).all())
        self.assertEqual(metrics["users_evaluated"], 4)
        self.assertGreaterEqual(metrics["recall@2"], 0.0)
        self.assertLessEqual(metrics["recall@2"], 1.0)
        self.assertGreaterEqual(metrics["ndcg@2"], 0.0)
        self.assertLessEqual(metrics["ndcg@2"], 1.0)
        self.assertEqual(len(details), 4)

    def test_history_evaluation_recovers_target_missed_by_first_seed(self):
        model = build_similarity_model(self.ratings, self.catalog)
        model.similarities = csr_matrix(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.8, 0.0],
                [0.0, 0.8, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        train = self.ratings[
            (self.ratings["user_id"] == "u1") & self.ratings["anime_id"].isin([1, 2])
        ].reset_index(drop=True)
        test = self.ratings[
            (self.ratings["user_id"] == "u1") & (self.ratings["anime_id"] == 3)
        ].reset_index(drop=True)
        test.loc[:, "rating"] = 8

        full_history, _ = evaluate_holdout(model, train, test, k=1)
        first_seed, _ = evaluate_holdout(model, train, test, k=1, use_all_history=False)

        self.assertEqual(full_history["recall@1"], 1.0)
        self.assertEqual(first_seed["recall@1"], 0.0)

    def test_popularity_baseline_uses_same_holdout_users(self):
        train, test = split_user_holdout(self.ratings, random_state=7)
        metrics, details = evaluate_popularity_holdout(train, test, k=2)

        self.assertEqual(metrics["users_evaluated"], len(test))
        self.assertEqual(len(details), len(test))
        self.assertGreaterEqual(metrics["recall@2"], 0.0)
        self.assertLessEqual(metrics["recall@2"], 1.0)
        self.assertGreaterEqual(metrics["ndcg@2"], 0.0)
        self.assertLessEqual(metrics["ndcg@2"], 1.0)

    def test_validation_and_test_splits_are_disjoint(self):
        ratings = self.ratings.copy()
        ratings["rating"] = 8
        train, validation, test = split_validation_test(ratings)
        validation_pairs = set(zip(validation["user_id"], validation["anime_id"]))
        test_pairs = set(zip(test["user_id"], test["anime_id"]))

        self.assertFalse(validation_pairs & test_pairs)
        self.assertEqual(set(validation["user_id"]), set(test["user_id"]))
        self.assertTrue(train.groupby("user_id")["rating"].max().ge(7).all())

    def test_positive_seed_target_pairs_score_known_neighbors(self):
        train = self.ratings[
            (self.ratings["user_id"] == "u1") & self.ratings["anime_id"].isin([1, 2])
        ].reset_index(drop=True)
        test = self.ratings[
            (self.ratings["user_id"] == "u1") & (self.ratings["anime_id"] == 3)
        ].reset_index(drop=True)
        test.loc[:, "rating"] = 8
        model = build_similarity_model(train, self.catalog)
        model.similarities = csr_matrix(
            [
                [1.0, 0.0, 0.8, 0.0],
                [0.0, 1.0, 0.8, 0.0],
                [0.8, 0.8, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        pairs = make_seed_target_pairs(train, test, random_state=1, max_seeds_per_user=2)
        scored = evaluate_seed_target_pairs(model, train, pairs, k=1)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(scored["cf_recall_at_k"].tolist(), [1, 1])

    def test_fixed_catalog_targets_recreate_same_split(self):
        ratings = self.ratings.copy()
        target = ratings[(ratings["user_id"] == "u1") & (ratings["anime_id"] == 3)].copy()
        target.loc[:, "rating"] = 8
        ratings.loc[target.index, "rating"] = 8
        seed_target_pairs = pd.DataFrame(
            {"user_id": ["u1", "u1"], "target_id": [3, 3]}
        )

        train, test = apply_fixed_targets(ratings, seed_target_pairs)

        self.assertEqual(len(test), 1)
        self.assertEqual(test.iloc[0]["anime_id"], 3)
        self.assertFalse(((train["user_id"] == "u1") & (train["anime_id"] == 3)).any())

    def test_popularity_pair_rank_accounts_for_all_seen_titles(self):
        catalog = pd.DataFrame(
            {"anime_id": [1, 2, 3, 4], "title": ["Alpha", "Beta", "Gamma", "Delta"]}
        )
        train = pd.DataFrame(
            [
                ("u1", 1, 10, "Alpha"), ("u1", 2, 9, "Beta"),
                ("u2", 3, 10, "Gamma"), ("u3", 3, 9, "Gamma"), ("u4", 3, 8, "Gamma"),
            ],
            columns=["user_id", "anime_id", "rating", "title"],
        )
        model = build_similarity_model(train, catalog)
        model.similarities = csr_matrix(
            [
                [1.0, 0.0, 0.8, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.8, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        pairs = pd.DataFrame(
            [{"user_id": "u1", "target_id": 3, "target_title": "Gamma", "seed_id": 1, "seed_title": "Alpha", "seed_number": 1}]
        )

        scored = evaluate_seed_target_pairs(model, train, pairs, k=1)

        self.assertEqual(scored.iloc[0]["cf_rank"], 1)
        self.assertEqual(scored.iloc[0]["popularity_rank"], 1)


if __name__ == "__main__":
    unittest.main()