import tempfile
import unittest
from pathlib import Path

import pandas as pd

from anime_recommender import (
    build_similarity_model,
    evaluate_holdout,
    evaluate_popularity_holdout,
    load_dataset,
    recommend_for_anime,
    split_user_holdout,
)


class AnimeRecommenderTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame(
            {
                "anime_id": [1, 2, 3, 4],
                "title": ["Alpha", "Beta", "Gamma", "Delta"],
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

    def test_popularity_baseline_uses_same_holdout_users(self):
        train, test = split_user_holdout(self.ratings, random_state=7)
        metrics, details = evaluate_popularity_holdout(train, test, k=2)

        self.assertEqual(metrics["users_evaluated"], len(test))
        self.assertEqual(len(details), len(test))
        self.assertGreaterEqual(metrics["recall@2"], 0.0)
        self.assertLessEqual(metrics["recall@2"], 1.0)
        self.assertGreaterEqual(metrics["ndcg@2"], 0.0)
        self.assertLessEqual(metrics["ndcg@2"], 1.0)


if __name__ == "__main__":
    unittest.main()