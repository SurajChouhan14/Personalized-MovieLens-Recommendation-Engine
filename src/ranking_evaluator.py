"""
Top-K Ranking and Retrieval Evaluation Metrics for Recommendation Systems.
Implements Precision@K, Recall@K, Hit-Ratio@K, and NDCG@K (Normalized Discounted Cumulative Gain).
"""
import numpy as np


class RecommenderRankingEvaluator:
    @staticmethod
    def evaluate_top_k(model, df_test, top_k=10, relevance_threshold=4.0):
        """
        Evaluates Precision@K, Recall@K, Hit-Ratio@K, and NDCG@K on test user interactions.
        """
        test_users = df_test["user_id"].unique()
        precisions = []
        recalls = []
        hits = []
        ndcgs = []

        for u in test_users:
            u_test = df_test[df_test["user_id"] == u]
            relevant_items = set(u_test[u_test["rating"] >= relevance_threshold]["item_id"])

            if not relevant_items:
                continue

            # Predict ratings for all items in test set for user u
            candidates = u_test["item_id"].values
            preds = [(it, model.predict_pair(u, it)) for it in candidates]
            preds.sort(key=lambda x: x[1], reverse=True)
            top_k_items = [it for it, _ in preds[:top_k]]

            # Metrics
            hit_count = len(set(top_k_items) & relevant_items)
            precisions.append(hit_count / float(top_k))
            recalls.append(hit_count / float(len(relevant_items)))
            hits.append(1.0 if hit_count > 0 else 0.0)

            # NDCG@K
            dcg = 0.0
            for rank_idx, it in enumerate(top_k_items):
                if it in relevant_items:
                    dcg += 1.0 / np.log2(rank_idx + 2)
            idcg = sum([1.0 / np.log2(r + 2) for r in range(min(len(relevant_items), top_k))])
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        return {
            "top_k": top_k,
            "precision_at_k": round(float(np.mean(precisions)), 4),
            "recall_at_k": round(float(np.mean(recalls)), 4),
            "hit_ratio_at_k": round(float(np.mean(hits)), 4),
            "ndcg_at_k": round(float(np.mean(ndcgs)), 4)
        }
