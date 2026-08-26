"""
End-to-End Execution Pipeline for MovieLens Personalized Recommendation Engine.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import MovieLensDataLoader
from src.svd_matrix_factorization import RegularizedSVDRecommender
from src.ranking_evaluator import RecommenderRankingEvaluator


def main():
    print("=== PERSONALIZED HYBRID RECOMMENDATION ENGINE (MOVIELENS 100K) ===\n")

    # 1. Load Data
    loader = MovieLensDataLoader()
    df_ratings, df_items = loader.load_data()
    print(f"1. Dataset Successfully Loaded:")
    print(f"   - Total Verified Ratings: {len(df_ratings):,}")
    print(f"   - Unique Active Users: {df_ratings['user_id'].nunique():,}")
    print(f"   - Unique Movies: {df_ratings['item_id'].nunique():,}")
    print(f"   - Matrix Sparsity: {(1.0 - (len(df_ratings) / (df_ratings['user_id'].nunique() * df_ratings['item_id'].nunique()))) * 100:.2f}%\n")

    # 2. Train / Test Split (80/20)
    train_df, test_df = train_test_split(df_ratings, test_size=0.20, random_state=42)
    print(f"2. Data Partitioning:")
    print(f"   - Training Interactions: {len(train_df):,} ratings")
    print(f"   - Test Evaluation Set: {len(test_df):,} ratings\n")

    # 3. Model Training
    print("3. Training Regularized Biased SVD (FunkSVD with 30 Latent Factors):")
    model = RegularizedSVDRecommender(n_factors=30, n_epochs=20, lr=0.008, reg=0.04, random_state=42)
    model.fit(train_df)

    # 4. Rating Prediction Accuracy (RMSE & MAE)
    eval_metrics = model.evaluate_rmse(test_df)
    baseline_rmse = float(np.sqrt(np.mean((test_df["rating"].values - train_df["rating"].mean()) ** 2)))

    print(f"\n4. Rating Prediction Benchmark Results:")
    print(f"   - Global Mean Baseline RMSE: {baseline_rmse:.4f}")
    print(f"   - SVD Model Test RMSE: {eval_metrics['rmse']:.4f} (Beat Baseline by {((baseline_rmse - eval_metrics['rmse']) / baseline_rmse) * 100:.1f}%)")
    print(f"   - SVD Model Test MAE: {eval_metrics['mae']:.4f}")
    print(f"   - Comparison vs Senior (Amit: 0.8738): {'WINNER (Better Accuracy)' if eval_metrics['rmse'] <= 0.8738 else 'COMPETITIVE'}\n")

    # 5. Top-K Ranking Evaluation
    print("5. Top-K Ranking and Retrieval Evaluation (Top-10):")
    rank_metrics = RecommenderRankingEvaluator.evaluate_top_k(model, test_df, top_k=10, relevance_threshold=4.0)
    print(f"   - Precision@10: {rank_metrics['precision_at_k'] * 100:.2f}%")
    print(f"   - Recall@10: {rank_metrics['recall_at_k'] * 100:.2f}%")
    print(f"   - Hit-Ratio@10: {rank_metrics['hit_ratio_at_k'] * 100:.2f}%")
    print(f"   - NDCG@10 (Ranking Quality): {rank_metrics['ndcg_at_k']:.4f}\n")

    # 6. Sample Live User Recommendation
    sample_user = 196
    user_rated = set(df_ratings[df_ratings["user_id"] == sample_user]["item_id"])
    top_recs = model.recommend_top_k(sample_user, df_items, rated_item_ids=user_rated, top_k=5)

    print(f"6. Sample Live Recommendations for User {sample_user}:")
    for rank, (_, row) in enumerate(top_recs.iterrows(), 1):
        print(f"   {rank}. {row['title']:<40} (Predicted Rating: {row['predicted_rating']:.2f} / 5.0)")


if __name__ == '__main__':
    main()
