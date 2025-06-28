import streamlit as st

from src.data_loader import load_data
from src.hybrid_recommender import hybrid_recommend
from src.explainer import explain_recommendation

ratings_df, metadata_df = load_data()

st.title("Explainable Hybrid Recommendation System")

user_ids = sorted(ratings_df["user_id"].unique())
user_id = st.selectbox("Select user ID", user_ids)

if st.button("Get Recommendations"):
    rec_items = hybrid_recommend(user_id)
    for item_id in rec_items:
        item_row = metadata_df[metadata_df["item_id"] == item_id].iloc[0]
        with st.expander(item_row["title"]):
            st.write(item_row["description"])
            explanation = explain_recommendation(user_id, item_id)
            st.markdown("**Explanation**")
            st.write(explanation)
