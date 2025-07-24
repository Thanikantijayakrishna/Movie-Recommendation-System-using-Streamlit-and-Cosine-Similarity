import streamlit as st
import pandas as pd
import joblib

# ─── Load Preprocessed Data ──────────────────────────────────────────────
df = joblib.load('movie_df.pkl')           # DataFrame with movie metadata
cosine_sim = joblib.load('cosine_sim.pkl') # Cosine similarity matrix
indices = joblib.load('indices.pkl')       # title ➜ index mapping

# ─── Recommend Function ───────────────────────────────────────────────────
def recommend_movie(title, num_recommendations=5):
    title = title.lower()
    if title not in indices:
        return []
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # Skip the movie itself

    recommended_movies = [df.iloc[i[0]]['title'] for i in sim_scores]
    return recommended_movies

# ─── Streamlit UI ─────────────────────────────────────────────────────────
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="centered")
st.title("🎬 Movie Recommendation System")
st.write("Get movie suggestions based on your favorite film using cosine similarity.")

# ─── Input Box ────────────────────────────────────────────────────────────
movie_input = st.text_input("Enter a movie title:")

if movie_input:
    recommendations = recommend_movie(movie_input)
    if recommendations:
        st.subheader("You may also like:")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("Movie not found. Please check spelling or try a different title.")
