"""
Data Loader and Parser for the Official GroupLens MovieLens-100K Benchmark.
Loads 100,000 ratings across 943 users and 1,682 movies with metadata.
"""
import os
import pandas as pd


class MovieLensDataLoader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, "data", "ml-100k")
        else:
            self.data_dir = data_dir

    def load_data(self):
        ratings_path = os.path.join(self.data_dir, "u.data")
        items_path = os.path.join(self.data_dir, "u.item")

        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"MovieLens u.data not found at {ratings_path}")

        # 1. Load Ratings (100,000 rows)
        df_ratings = pd.read_csv(
            ratings_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python"
        )

        # 2. Load Movie Metadata (1,682 movies)
        genre_cols = [
            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols

        df_items = pd.read_csv(
            items_path,
            sep="|",
            names=item_cols,
            encoding="latin-1",
            engine="python"
        )[["item_id", "title", "release_date"] + genre_cols]

        return df_ratings, df_items
