import unittest
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import MovieLensDataLoader
from src.svd_matrix_factorization import RegularizedSVDRecommender
from src.ranking_evaluator import RecommenderRankingEvaluator


class TestRecommenderEngine(unittest.TestCase):
    def setUp(self):
        self.loader = MovieLensDataLoader()
        self.df_ratings, self.df_items = self.loader.load_data()
        self.train_df, self.test_df = train_test_split(self.df_ratings, test_size=0.20, random_state=42)
        self.model = RegularizedSVDRecommender(n_factors=15, n_epochs=5, lr=0.01, reg=0.05, random_state=42)
        self.model.fit(self.train_df)

    def test_data_integrity(self):
        self.assertEqual(len(self.df_ratings), 100000)
        self.assertEqual(self.df_ratings["user_id"].nunique(), 943)
        self.assertEqual(self.df_ratings["item_id"].nunique(), 1682)
        self.assertTrue((self.df_ratings["rating"] >= 1.0).all() and (self.df_ratings["rating"] <= 5.0).all())

    def test_prediction_bounds(self):
        pred = self.model.predict_pair(user_id=196, item_id=242)
        self.assertGreaterEqual(pred, 1.0)
        self.assertLessEqual(pred, 5.0)

    def test_rmse_benchmark(self):
        metrics = self.model.evaluate_rmse(self.test_df)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        # RMSE must be well below 1.0
        self.assertLess(metrics["rmse"], 1.0)

    def test_top_k_recommendations(self):
        top_recs = self.model.recommend_top_k(user_id=196, df_items=self.df_items, top_k=5)
        self.assertEqual(len(top_recs), 5)
        self.assertIn("title", top_recs.columns)
        self.assertIn("predicted_rating", top_recs.columns)


if __name__ == '__main__':
    unittest.main()
