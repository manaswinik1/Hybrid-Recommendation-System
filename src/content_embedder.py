import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def generate_content_embeddings(metadata_df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings for item descriptions using sentence-transformers.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame containing item metadata with a ``description`` column.
    model_name : str, optional
        Pretrained model name for ``SentenceTransformer``.

    Returns
    -------
    np.ndarray
        Embeddings array aligned with ``metadata_df`` index.
    """
    model = SentenceTransformer(model_name)
    descriptions = metadata_df["description"].fillna("").tolist()
    embeddings = model.encode(descriptions, show_progress_bar=False)
    return embeddings
