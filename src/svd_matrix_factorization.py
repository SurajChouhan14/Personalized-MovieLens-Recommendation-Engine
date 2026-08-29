"""
Regularized Biased SVD (FunkSVD / Latent Factor Matrix Factorization) Engine.
Formulation: r_hat(u, i) = mu + b_u + b_i + p_u^T * q_i
Optimized via Stochastic Gradient Descent (SGD) with L2 weight decay.
"""
import numpy as np
import pandas as pd


class RegularizedSVDRecommender:
    def __init__(self, n_factors=30, n_epochs=20, lr=0.008, reg=0.04, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

        self.mu = 0.0
        self.b_u = None
        self.b_i = None
        self.P = None
        self.Q = None

        self.user2idx = {}
        self.item2idx = {}
        self.idx2user = {}
        self.idx2item = {}

    def fit(self, df_train):
        np.random.seed(self.random_state)

        # Build index mappings
        unique_users = df_train["user_id"].unique()
        unique_items = df_train["item_id"].unique()

        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.item2idx = {it: i for i, it in enumerate(unique_items)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {i: it for it, i in self.item2idx.items()}

        n_users = len(unique_users)
        n_items = len(unique_items)

        self.mu = float(df_train["rating"].mean())
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        # Latent factor embeddings
        self.P = np.random.normal(0.0, 0.08, size=(n_users, self.n_factors))
        self.Q = np.random.normal(0.0, 0.08, size=(n_items, self.n_factors))

        users = df_train["user_id"].map(self.user2idx).values
        items = df_train["item_id"].map(self.item2idx).values
        ratings = df_train["rating"].values

        # SGD Training Loop
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(ratings))
            for idx in indices:
                u = users[idx]
                i = items[idx]
                r = ratings[idx]

                # Prediction error
                pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred

                # SGD updates with L2 regularization
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                p_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_u_old - self.reg * self.Q[i])

        return self

    def predict_pair(self, user_id, item_id):
        u = self.user2idx.get(user_id, None)
        i = self.item2idx.get(item_id, None)

        if u is None and i is None:
            return self.mu
        elif u is None:
            return np.clip(self.mu + self.b_i[i], 1.0, 5.0)
        elif i is None:
            return np.clip(self.mu + self.b_u[u], 1.0, 5.0)

        pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
        return float(np.clip(pred, 1.0, 5.0))

    def evaluate_rmse(self, df_test):
        preds = [self.predict_pair(row["user_id"], row["item_id"]) for _, row in df_test.iterrows()]
        y_true = df_test["rating"].values
        rmse = float(np.sqrt(np.mean((np.array(preds) - y_true) ** 2)))
        mae = float(np.mean(np.abs(np.array(preds) - y_true)))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4)}

    def recommend_top_k(self, user_id, df_items, rated_item_ids=None, top_k=10):
        if rated_item_ids is None:
            rated_item_ids = set()

        all_item_ids = df_items["item_id"].values
        candidate_ids = [it for it in all_item_ids if it not in rated_item_ids]

        scores = [(it, self.predict_pair(user_id, it)) for it in candidate_ids]
        scores.sort(key=lambda x: x[1], reverse=True)

        top_candidates = scores[:top_k]
        top_df = pd.DataFrame(top_candidates, columns=["item_id", "predicted_rating"])
        top_df = top_df.merge(df_items[["item_id", "title", "release_date"]], on="item_id", how="left")
        return top_df


# Alias
FunkSVDRecommender = RegularizedSVDRecommender
