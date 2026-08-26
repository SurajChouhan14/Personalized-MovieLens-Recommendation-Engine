"""
Production REST API for Personalized Movie Recommendation Engine.
Exposes real-time personalized Top-K recommendations and pair-wise rating predictions.
Run with: uvicorn serve_api:app --reload --port 8005
"""
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SYS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SYS_PATH)

from src.data_loader import MovieLensDataLoader
from src.svd_matrix_factorization import RegularizedSVDRecommender

app = FastAPI(
    title="MovieLens Personalized Hybrid Recommendation API",
    description="Regularized Biased SVD Latent Factor Matrix Factorization recommendation engine with Top-K ranking.",
    version="1.0.0"
)

loader = MovieLensDataLoader()
df_ratings, df_items = loader.load_data()

model = RegularizedSVDRecommender(n_factors=30, n_epochs=20, lr=0.008, reg=0.04)
model.fit(df_ratings)


class RecommendRequest(BaseModel):
    user_id: int = Field(default=196, ge=1, le=943, description="Target User ID (1-943)")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")


class RatingPredictRequest(BaseModel):
    user_id: int = Field(default=196, ge=1, le=943, description="User ID")
    item_id: int = Field(default=242, ge=1, le=1682, description="Movie Item ID")


@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "engine": "MovieLens Regularized Biased SVD Matrix Factorization",
        "dataset_ratings": len(df_ratings),
        "total_users": df_ratings["user_id"].nunique(),
        "total_movies": df_ratings["item_id"].nunique(),
        "latent_dimensions": 30
    }


@app.post("/recommend_top_movies")
def recommend_top_movies(req: RecommendRequest):
    try:
        user_rated = set(df_ratings[df_ratings["user_id"] == req.user_id]["item_id"])
        recs_df = model.recommend_top_k(req.user_id, df_items, rated_item_ids=user_rated, top_k=req.top_k)

        results = []
        for _, row in recs_df.iterrows():
            results.append({
                "item_id": int(row["item_id"]),
                "title": str(row["title"]),
                "release_date": str(row["release_date"]),
                "predicted_rating": round(float(row["predicted_rating"]), 2)
            })

        return {
            "user_id": req.user_id,
            "top_k": req.top_k,
            "recommendations": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_rating")
def predict_rating(req: RatingPredictRequest):
    try:
        score = model.predict_pair(req.user_id, req.item_id)
        movie_match = df_items[df_items["item_id"] == req.item_id]
        title = str(movie_match["title"].values[0]) if not movie_match.empty else "Unknown Movie"

        return {
            "user_id": req.user_id,
            "item_id": req.item_id,
            "movie_title": title,
            "predicted_rating": round(float(score), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
