import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb

movies = pd.read_csv("movies.csv")

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
movie_tfidf = tfidf.fit_transform(movies["title"] + " " + movies["genres"])
cosine_sim = cosine_similarity(movie_tfidf, movie_tfidf)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_movie_model.json")

def hybrid_recommend(movie_title, top_n=10):
    if movie_title not in indices:
        return pd.DataFrame({"title": [f"Movie '{movie_title}' not found"], "genres": [""], "pred_rating": ["-"]})
    
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    recs = movies.iloc[movie_indices].copy()
    tfidf_input = tfidf.transform(recs["title"] + " " + recs["genres"])
    recs["pred_rating"] = xgb_model.predict(tfidf_input)
    
    return recs.sort_values("pred_rating", ascending=False)[["title", "genres", "pred_rating"]]

st.title("🎬 Hybrid Movie Recommendation System")
movie_name = st.text_input("Enter a movie title:", "Toy Story (1995)")

if st.button("Recommend"):
    recs = hybrid_recommend(movie_name, top_n=10)
    st.dataframe(recs)
