# Explainable Hybrid Recommendation System

This project demonstrates a prototype hybrid recommender that combines collaborative filtering and content-based methods using sentence-transformer embeddings. Explanations are generated with SHAP to help users understand why items were recommended. A simple Streamlit interface lets you explore the recommendations interactively.

## Features
- Personalized hybrid recommendations
- Embedding-based content similarity
- SHAP-based explanation of scoring
- Interactive Streamlit web interface

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Folder Structure
```
hybrid-recommender-xai/
├── data/
│   └── raw/
│       ├── movielens_ratings.csv
│       └── movielens_metadata.csv
├── models/
├── src/
│   ├── data_loader.py
│   ├── collaborative_filtering.py
│   ├── content_embedder.py
│   ├── hybrid_recommender.py
│   ├── explainer.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Example
Screenshot placeholder.

## Disclaimer
This repository is a prototype for demonstration purposes only and is not intended for production use.
