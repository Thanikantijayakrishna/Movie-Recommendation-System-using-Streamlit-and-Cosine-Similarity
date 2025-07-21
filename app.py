import streamlit as st
import pandas as pd
import joblib

# ─── Load Data ─────────────────────────────────────────
df = joblib.load('movie_df.pkl')          # DataFrame with all movie metadata
cosine_sim = joblib.load('cosine_sim.pkl')  # Cosine similarity matrix
indices = joblib.load('indices.pkl')        # title → df index mapping (lowercase)

# ─── Page Config + Styling ────────────────────────────
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.markdown(
    """
    <style>
        body {
            background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
            color:white;
        }
        .movie-card{
            background:rgba(255,255,255,.05);
            padding:1rem;border-radius:10px;margin:10px;text-align:center;
        }
        .movie-detail{
            background:rgba(255,255,255,.10);
            padding:1rem;border-radius:10px;
        }
        img{border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Recommender Logic ────────────────────────────────
def recommend_movies_by_index(index: int, num: int = 10) -> pd.DataFrame:
    sim_scores = cosine_sim[index]
    similar_indices = sim_scores.argsort()[::-1][1:num + 1]
    return df.iloc[similar_indices]

def get_index_by_title(title: str) -> int:
    title = title.lower()
    if title not in indices:
        return -1
    idx_val = indices[title]
    return idx_val.iloc[0] if isinstance(idx_val, pd.Series) else idx_val

# ─── Read Query Param (when a poster is clicked) ──────
qp = st.query_params
selected_idx = int(qp.get("selected", ["-1"])[0])

# ─── Search Box ───────────────────────────────────────
st.title("🎬 Movie Recommendation System")
movie_name = st.text_input("Enter a movie name:", placeholder="e.g., The Godfather")

if st.button("Get Recommendations") and movie_name.strip():
    idx = get_index_by_title(movie_name)
    if idx == -1:
        st.error("Movie not found in the database.")
    else:
        recs = recommend_movies_by_index(idx)
        st.session_state["initial_recs"] = recs
        st.session_state["selected_movie_idx"] = idx
        st.query_params.clear()
        selected_idx = -1  # reset poster click

# ─── Show Recommendations ─────────────────────────────
if "initial_recs" in st.session_state:
    recs = st.session_state["initial_recs"]
    st.subheader("🎯 Top 10 Similar Movies")
    cols = st.columns(5)
    for i, (row_idx, row) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            if pd.notna(row["poster_path"]):
                poster = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
                st.markdown(
                    f'<a href="?selected={row_idx}"><img src="{poster}" width="100%"></a>',
                    unsafe_allow_html=True,
                )
            st.markdown(f"**🎞️ {row['original_title']}**")
            st.markdown("</div>", unsafe_allow_html=True)

# ─── Poster Clicked → Show Movie Detail + Recs ────────
if selected_idx != -1 and selected_idx in df.index:
    movie = df.loc[selected_idx]
    st.markdown("---")
    st.subheader(f"🎬 {movie['original_title']}")

    left, right = st.columns([1, 2])
    with left:
        if pd.notna(movie['poster_path']):
            st.image(f"https://image.tmdb.org/t/p/w500{movie['poster_path']}",
                     use_container_width=True)

    with right:
        st.markdown('<div class="movie-detail">', unsafe_allow_html=True)
        st.markdown(f"📅 **Release Date:** {movie['release_date']}")
        st.markdown(f"🗣️ **Language:** {movie['original_language']}")
        st.markdown(f"🎭 **Genres:** {movie['genres']}")
        overview = movie['overview'] if pd.notna(movie['overview']) else "No overview available."
        st.markdown(f"📝 **Overview:** {overview}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Re-recommendations based on selected movie
    more_recs = recommend_movies_by_index(selected_idx)
    if not more_recs.empty:
        st.subheader(f"🎬 Movies similar to {movie['original_title']}")
        sub_cols = st.columns(5)
        for i, (ridx, rrow) in enumerate(more_recs.iterrows()):
            with sub_cols[i % 5]:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                if pd.notna(rrow['poster_path']):
                    poster = f"https://image.tmdb.org/t/p/w500{rrow['poster_path']}"
                    st.markdown(
                        f'<a href="?selected={ridx}"><img src="{poster}" width="100%"></a>',
                        unsafe_allow_html=True,
                    )
                st.markdown(f"**🎞️ {rrow['original_title']}**")
                st.markdown("</div>", unsafe_allow_html=True)
