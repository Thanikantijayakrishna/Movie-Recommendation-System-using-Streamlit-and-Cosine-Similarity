import streamlit as st
import pandas as pd
import joblib

# ─── Load Data ─────────────────────────────────────────
df = joblib.load('movie_df.pkl')          # DataFrame with all movie meta
cosine_sim = joblib.load('cosine_sim.pkl')
indices     = joblib.load('indices.pkl')   # title ➜ df‑index mapping (lower‑case)

# ─── Page config + CSS ─────────────────────────────────
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

# ─── Recommender helper ─────────────────────────────────

def recommend_movies(title: str) -> pd.DataFrame:
    """Return top‑10 similar movies as DataFrame (empty if not found)."""
    title = title.lower()
    if title not in indices:
        return pd.DataFrame()
    idx_entry = indices[title]
    idx = idx_entry.iloc[0] if isinstance(idx_entry, pd.Series) else idx_entry
    sims = cosine_sim[idx]
    top_10 = sims.argsort()[::-1][1:11]  # skip the movie itself
    return df.iloc[top_10]

# ─── Read query‑param if user clicked a poster ──────────
qp = st.query_params
selected_idx = int(qp.get("selected", ["-1"])[0])  # -1 ➜ nothing selected

# ─── Input field & search button ───────────────────────
st.title("🎬 Movie Recommendation System")
movie_name = st.text_input("Enter a movie name:", placeholder="e.g., The Godfather")

if st.button("Get Recommendations") and movie_name.strip():
    recs = recommend_movies(movie_name)
    if recs.empty:
        st.error("Movie not found in database.")
    else:
        # rewrite query‑params to clear previous selection
        st.query_params.clear()
        st.session_state["initial_recs"] = recs
        selected_idx = -1  # reset selection

# ─── Show initial recommendations grid ─────────────────
if "initial_recs" in st.session_state:
    recs: pd.DataFrame = st.session_state["initial_recs"]
    st.subheader("Top 10 Similar Movies")
    cols = st.columns(5)
    for i, (row_idx, row) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            if pd.notna(row["poster_path"]):
                poster = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
                # clickable image ➜ adds ?selected=row_idx to URL, no new tab
                st.markdown(
                    f'<a href="?selected={row_idx}"><img src="{poster}" width="100%"></a>',
                    unsafe_allow_html=True,
                )
            st.markdown(f"**🎞️ {row['original_title']}**")
            st.markdown("</div>", unsafe_allow_html=True)

# ─── If user clicked a poster, show details + 10 more ──
if selected_idx != -1 and selected_idx in df.index:
    movie = df.loc[selected_idx]

    st.markdown("---")
    st.subheader(f"🎬 {movie['original_title']}")
    detail_left, detail_right = st.columns([1,2])

    with detail_left:
        if pd.notna(movie['poster_path']):
            st.image(f"https://image.tmdb.org/t/p/w500{movie['poster_path']}",
                     use_container_width=True)

    with detail_right:
        st.markdown('<div class="movie-detail">', unsafe_allow_html=True)
        st.markdown(f"📅 **Release Date:** {movie['release_date']}")
        st.markdown(f"🗣️ **Language:** {movie['original_language']}")
        st.markdown(f"🎭 **Genres:** {movie['genres']}")
        overview = movie['overview'] if pd.notna(movie['overview']) else "No overview available."
        st.markdown(f"📝 **Overview:** {overview}")
        st.markdown('</div>', unsafe_allow_html=True)

    # secondary recommendations (from clicked movie)
    sub_recs = recommend_movies(movie['original_title'])
    if not sub_recs.empty:
        st.subheader(f"Movies similar to {movie['original_title']}")
        sub_cols = st.columns(5)
        for i, (ridx, rrow) in enumerate(sub_recs.iterrows()):
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
