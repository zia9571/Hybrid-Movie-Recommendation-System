import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import requests

TMDB_API_KEY = "d2715da28b169ce7d0f24a87b7f11077" 

movies = pd.read_csv("movies.csv")

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
movie_tfidf = tfidf.fit_transform(movies["title"] + " " + movies["genres"])
cosine_sim = cosine_similarity(movie_tfidf, movie_tfidf)

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_movie_model.json")

def fetch_movie_from_tmdb(query):
    """Search TMDB for a movie and return details."""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
    response = requests.get(url).json()
    if response["results"]:
        return response["results"][0]  # best match
    return None

def hybrid_recommend_from_tmdb(query, top_n=10):
    """Find recommendations based on TMDB query + MovieLens hybrid model."""
    movie_data = fetch_movie_from_tmdb(query)
    if not movie_data:
        return pd.DataFrame({"title": [f"'{query}' not found in TMDB"], "genres": [""], "pred_rating": ["-"]})
    
    genres = " ".join([g["name"] for g in movie_data.get("genre_ids", [])]) if "genre_ids" in movie_data else ""
    overview = movie_data.get("overview", "")
    query_text = movie_data["title"] + " " + genres + " " + overview

    # TF-IDF transform
    query_vec = tfidf.transform([query_text])

    # Similarity with MovieLens dataset
    sim_scores = cosine_similarity(query_vec, movie_tfidf).flatten()
    top_idx = sim_scores.argsort()[-top_n:][::-1]

    recs = movies.iloc[top_idx].copy()
    tfidf_input = tfidf.transform(recs["title"] + " " + recs["genres"])
    recs["pred_rating"] = xgb_model.predict(tfidf_input)

    return recs.sort_values("pred_rating", ascending=False)[["title", "genres", "pred_rating"]]

st.title("🎬 Hybrid Movie Recommendation System (TMDB + MovieLens)")

user_query = st.text_input("Enter any movie title:", "The Materialist")

if st.button("Recommend"):
    recs = hybrid_recommend_from_tmdb(user_query, top_n=10)
    st.dataframe(recs)
