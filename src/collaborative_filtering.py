import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


def recommend_cf(user_id: int, ratings_df: pd.DataFrame, top_n: int = 10) -> List[int]:
    """Recommend items for a user using item-based collaborative filtering.

    Parameters
    ----------
    user_id : int
        Target user identifier.
    ratings_df : pd.DataFrame
        Ratings data with columns ["user_id", "item_id", "rating"].
    top_n : int, optional
        Number of recommendations to return.

    Returns
    -------
    List[int]
        Recommended item IDs sorted by predicted rating.
    """
    # Create user-item rating matrix
    user_item = ratings_df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)

    # Compute item-item similarity matrix
    similarity = cosine_similarity(user_item.T)
    similarity_df = pd.DataFrame(similarity, index=user_item.columns, columns=user_item.columns)

    # Ratings from the target user
    user_ratings = user_item.loc[user_id]

    scores = {}
    for item in user_item.columns:
        if user_ratings[item] == 0:
            sims = similarity_df[item]
            score = np.dot(sims, user_ratings) / (np.abs(sims).sum() + 1e-8)
            scores[item] = score

    recommended = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return recommended
