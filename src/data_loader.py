import pandas as pd
from typing import Tuple


def load_data(
    ratings_path: str = "data/raw/movielens_ratings.csv",
    metadata_path: str = "data/raw/movielens_metadata.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and metadata CSV files.

    The function cleans basic issues such as missing values and
    duplicates before returning DataFrames.

    Parameters
    ----------
    ratings_path : str, optional
        Path to the ratings CSV file.
    metadata_path : str, optional
        Path to the metadata CSV file.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing the ratings DataFrame and metadata DataFrame.
    """
    ratings_df = pd.read_csv(ratings_path)
    metadata_df = pd.read_csv(metadata_path)

    # Drop rows with missing essential values
    ratings_df = ratings_df.dropna(subset=["user_id", "item_id", "rating"])
    metadata_df = metadata_df.dropna(subset=["item_id", "title", "description"])

    # Ensure correct dtypes
    ratings_df["user_id"] = ratings_df["user_id"].astype(int)
    ratings_df["item_id"] = ratings_df["item_id"].astype(int)
    metadata_df["item_id"] = metadata_df["item_id"].astype(int)

    # Remove duplicates
    ratings_df = ratings_df.drop_duplicates()
    metadata_df = metadata_df.drop_duplicates("item_id")

    return ratings_df, metadata_df
