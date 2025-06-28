import numpy as np
import shap
from typing import Tuple

from .hybrid_recommender import get_item_scores


def explain_recommendation(
    user_id: int, item_id: int, weights: Tuple[float, float] = (0.5, 0.5)
) -> str:
    """Generate a textual explanation for a recommendation using SHAP.

    Parameters
    ----------
    user_id : int
        Identifier for the target user.
    item_id : int
        Identifier for the recommended item.
    weights : tuple of float, optional
        Weights used in the hybrid recommender.

    Returns
    -------
    str
        Human-readable explanation string.
    """
    cf_score, content_score, combined_score = get_item_scores(user_id, item_id, weights)

    def model(X: np.ndarray) -> np.ndarray:
        return X.dot(np.array(weights))

    explainer = shap.Explainer(model, np.zeros((1, 2)))
    shap_values = explainer(np.array([[cf_score, content_score]]))
    cf_contrib, content_contrib = shap_values.values[0]

    explanation = (
        f"Recommendation score for item {item_id}: {combined_score:.3f}\n"
        f" - Collaborative filtering contribution: {cf_contrib:.3f}\n"
        f" - Content similarity contribution: {content_contrib:.3f}"
    )
    return explanation
