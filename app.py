import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import requests

TMDB_API_KEY = "d2715da28b169ce7d0f24a87b7f11077"

def get_tmdb_genre_mapping():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url).json()
    return {g["id"]: g["name"] for g in response["genres"]}

tmdb_genres = get_tmdb_genre_mapping()

movies = pd.read_csv("movies.csv")
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
movie_tfidf = tfidf.fit_transform(movies["title"] + " " + movies["genres"])
cosine_sim = cosine_similarity(movie_tfidf, movie_tfidf)

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_movie_model.json")

def fetch_movie_from_tmdb(query):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
    response = requests.get(url).json()
    if response["results"]:
        return response["results"][0]  # take first match
    return None

def hybrid_recommend_from_tmdb(query, top_n=10):
    movie_data = fetch_movie_from_tmdb(query)
    if not movie_data:
        return pd.DataFrame({"title": [f"'{query}' not found in TMDB"],
                             "genres": [""],
                             "pred_rating": ["-"],
                             "poster": [""]})

    genre_ids = movie_data.get("genre_ids", [])
    genres = " ".join([tmdb_genres.get(g, "") for g in genre_ids])
    overview = movie_data.get("overview", "")
    query_text = movie_data["title"] + " " + genres + " " + overview

    # TF-IDF similarity
    query_vec = tfidf.transform([query_text])
    sim_scores = cosine_similarity(query_vec, movie_tfidf).flatten()
    top_idx = sim_scores.argsort()[-top_n:][::-1]

    recs = movies.iloc[top_idx].copy()
    tfidf_input = tfidf.transform(recs["title"] + " " + recs["genres"])
    recs["pred_rating"] = xgb_model.predict(tfidf_input)

    # Fetch posters from TMDB for each recommended movie
    poster_urls = []
    for title in recs["title"]:
        tmdb_result = fetch_movie_from_tmdb(title)
        if tmdb_result and tmdb_result.get("poster_path"):
            poster_urls.append(f"https://image.tmdb.org/t/p/w200{tmdb_result['poster_path']}")
        else:
            poster_urls.append("")  # empty if not found

    recs["poster"] = poster_urls

    return recs.sort_values("pred_rating", ascending=False)[["title", "genres", "pred_rating", "poster"]]


st.title("🎬 Hybrid Movie Recommendation System (TMDB + MovieLens)")

user_query = st.text_input("Enter any movie title:", "The Materialist")

if st.button("Recommend"):
    recs = hybrid_recommend_from_tmdb(user_query, top_n=10)
    
    for idx, row in recs.iterrows():
        cols = st.columns([1,3])  # small column for poster, big for info
        with cols[0]:
            if row["poster"]:
                st.image(row["poster"])
            else:
                st.text("No poster")
        with cols[1]:
            st.markdown(f"**{row['title']}**")
            st.markdown(f"Genres: {row['genres']}")
            st.markdown(f"Predicted Rating: {row['pred_rating']:.2f}")
        st.markdown("---")


