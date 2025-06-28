from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import load_data
from .content_embedder import generate_content_embeddings

# Load data once for simplicity
_ratings_df, _metadata_df = load_data()
_embeddings = generate_content_embeddings(_metadata_df)


def _compute_cf_scores(user_id: int) -> pd.Series:
    """Compute collaborative filtering scores for all items."""
    user_item = _ratings_df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)
    similarity = cosine_similarity(user_item.T)
    similarity_df = pd.DataFrame(similarity, index=user_item.columns, columns=user_item.columns)
    user_ratings = user_item.loc[user_id]

    scores = {}
    for item in user_item.columns:
        if user_ratings[item] == 0:
            sims = similarity_df[item]
            score = np.dot(sims, user_ratings) / (np.abs(sims).sum() + 1e-8)
            scores[item] = score
    return pd.Series(scores)


def _compute_content_scores(user_id: int) -> pd.Series:
    """Compute content similarity scores for all items."""
    user_history = _ratings_df[_ratings_df["user_id"] == user_id]
    liked = user_history[user_history["rating"] > 3]["item_id"].tolist()

    if liked:
        liked_idx = [
            _metadata_df.index[_metadata_df["item_id"] == i][0]
            for i in liked
            if i in _metadata_df["item_id"].values
        ]
        liked_embs = _embeddings[liked_idx]
        similarity = cosine_similarity(_embeddings, liked_embs).mean(axis=1)
    else:
        similarity = np.zeros(len(_metadata_df))

    scores = {}
    rated_items = set(user_history["item_id"].tolist())
    for idx, item_id in enumerate(_metadata_df["item_id"]):
        if item_id not in rated_items:
            scores[item_id] = float(similarity[idx])
    return pd.Series(scores)


def hybrid_recommend(user_id: int, weights: Tuple[float, float] = (0.5, 0.5), top_n: int = 10) -> List[int]:
    """Return top-N item IDs using a weighted hybrid approach."""
    cf_scores = _compute_cf_scores(user_id)
    content_scores = _compute_content_scores(user_id)

    common_items = cf_scores.index.intersection(content_scores.index)
    combined = (
        cf_scores.loc[common_items] * weights[0]
        + content_scores.loc[common_items] * weights[1]
    )
    recommended = combined.sort_values(ascending=False).head(top_n).index.tolist()
    return recommended


def get_item_scores(
    user_id: int, item_id: int, weights: Tuple[float, float] = (0.5, 0.5)
) -> Tuple[float, float, float]:
    """Utility to fetch score components for explanation."""
    cf_scores = _compute_cf_scores(user_id)
    content_scores = _compute_content_scores(user_id)
    cf = cf_scores.get(item_id, 0.0)
    content = content_scores.get(item_id, 0.0)
    combined = weights[0] * cf + weights[1] * content
    return cf, content, combined
