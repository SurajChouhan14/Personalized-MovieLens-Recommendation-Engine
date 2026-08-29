"""
Automated Unit Test Suite for Personalized Hybrid Recommendation Engine (SVD + NCF).
Verifies Data Ingestion, SVD Factorization, PyTorch/NumPy NCF Embeddings, and Top-K Ranking Metrics.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import MovieLensDataLoader
from src.svd_matrix_factorization import RegularizedSVDRecommender
from src.ncf_model import NeuralCollaborativeFilteringEngine
from src.ranking_evaluator import RecommenderRankingEvaluator


class TestRecommendationEngine(unittest.TestCase):
    """
    Unit test cases for hybrid SVD and PyTorch NCF recommendation.
    """

    def setUp(self):
        self.loader = MovieLensDataLoader()
        self.ratings_df, self.movies_df = self.loader.load_data()

    def test_movielens_data_ingestion(self):
        """Verify MovieLens 100K ratings and items are ingested correctly."""
        self.assertEqual(len(self.ratings_df), 100000)
        self.assertEqual(self.ratings_df["user_id"].nunique(), 943)
        self.assertEqual(self.ratings_df["item_id"].nunique(), 1682)
        self.assertIn("rating", self.ratings_df.columns)

    def test_svd_matrix_factorization(self):
        """Verify SVD learns user/item latent matrices and produces bounded rating predictions."""
        sample_df = self.ratings_df.iloc[:2000]
        svd = RegularizedSVDRecommender(n_factors=10, n_epochs=5)
        svd.fit(sample_df)

        pred = svd.predict_pair(user_id=196, item_id=242)
        self.assertGreaterEqual(pred, 1.0)
        self.assertLessEqual(pred, 5.0)

    def test_ncf_model_training(self):
        """Verify NCF NeuMF engine trains and outputs probabilities in [0, 1]."""
        sample_df = self.ratings_df.iloc[:3000]
        ncf = NeuralCollaborativeFilteringEngine(latent_dim=8, batch_size=128)
        res = ncf.fit(sample_df, num_users=100, num_items=200, epochs=2)

        self.assertIn("final_loss", res)
        self.assertLess(res["final_loss"], 2.0)

    def test_top_k_ranking_evaluation(self):
        """Verify Hit Rate@10 and NDCG@10 metrics are within valid bounds [0, 1]."""
        sample_df = self.ratings_df.iloc[:3000]
        ncf = NeuralCollaborativeFilteringEngine(latent_dim=8, batch_size=128)
        ncf.fit(sample_df, num_users=50, num_items=100, epochs=2)

        test_items = {0: 10, 1: 15, 2: 20}
        metrics = ncf.evaluate_leave_one_out(test_items, top_k=10)

        self.assertIn("hit_rate@10", metrics)
        self.assertIn("ndcg@10", metrics)
        self.assertGreater(metrics["hit_rate@10"], 0.50)
        self.assertGreater(metrics["ndcg@10"], 0.30)


if __name__ == '__main__':
    unittest.main()
