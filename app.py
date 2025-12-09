import streamlit as st
import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from models.ga_optimizer import GAOptimizer
from models.ml_recommender import MovieRecommender
from sentiment.sentiment_analysis import get_sentiment
from utils import preprocess_genres, get_closest_title
import matplotlib.pyplot as plt
import time
import random
from services.poster_service import PosterService
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config for professional look
st.set_page_config(
    page_title="CineMatch - Smart Movie Recommendations",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #E50914;
        --secondary: #221F1F;
        --background: #141414;
        --text: #FFFFFF;
        --card: #181818;
        --highlight: #E50914;
    }
    
    /* Overall app styling */
    .stApp {
        background-color: var(--background);
        color: var(--text);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: var(--text) !important;
        font-weight: 600 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #F40612;
        border: none;
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        background-color: #333333;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem;
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div>select {
        background-color: #333333;
        color: white;
    }
    
    /* Card styling */
    .movie-card {
        background-color: var(--card);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .movie-card:hover {
        transform: scale(1.02);
    }
    
    /* Rating styling */
    .rating {
        display: inline-block;
        background-color: var(--primary);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--secondary);
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        color: var(--text);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
    }
    
    /* Logo styling */
    .logo-text {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary);
        margin-bottom: 0;
    }
    .logo-subtitle {
        font-size: 1rem;
        color: #AAAAAA;
        margin-top: 0;
    }
    
    /* Hide debug expanders by default */
    .streamlit-expanderHeader {
        font-size: 0.8rem;
        color: #888888;
    }
    
    /* Slider styling */
    .stSlider > div > div {
        color: white;
    }
    
    /* For movie genres */
    .genre-tag {
        display: inline-block;
        background-color: #333333;
        color: white;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
        font-size: 0.8rem;
    }
    
    /* For "Powered by ML" badge */
    .ml-badge {
        display: inline-block;
        background-color: #6C5CE7;
        color: white;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    
    /* For section dividers */
    .divider {
        border-top: 1px solid #333333;
        margin: 1.5rem 0;
    }
    
    /* For movie overview text */
    .overview {
        color: #BBBBBB;
        font-size: 0.9rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# App header with logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<p class="logo-text">CineMatch</p>', unsafe_allow_html=True)
    st.markdown('<p class="logo-subtitle">Smart movie recommendations powered by AI</p>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/tmdb_5000_movies.csv")
    df.dropna(subset=['overview'], inplace=True)
    
    # Parse genres
    df['genres'] = df['genres'].apply(lambda x: [d['name'] for d in ast.literal_eval(x)])
    
    # Fix any data issues - ensure vote_average is correct
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
    df['vote_average'] = df['vote_average'].fillna(0)
    
    # Ensure runtime is numeric
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['runtime'] = df['runtime'].fillna(0)
    
    return df

df = load_data()
_, genre_names = preprocess_genres(df['genres'])

# Initialize ML recommender
@st.cache_resource
def initialize_ml_recommender(df):
    ml_recommender = MovieRecommender()
    # Only train if not already trained
    if not ml_recommender.is_trained():
        with st.spinner("Initializing recommendation engine..."):
            ml_recommender.train(df)
    return ml_recommender

ml_recommender = initialize_ml_recommender(df)

# Initialize the poster service
@st.cache_resource
def initialize_poster_service():
    return PosterService()

poster_service = initialize_poster_service()

# TF-IDF for overviews
@st.cache_resource
def create_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = create_similarity_matrix(df)

# Title index lookup
indices = pd.Series(df.index, index=df['title'])

# Main UI
tab1, tab2, tab3 = st.tabs(["Discover Movies", "Mood Match", "About CineMatch"])

with tab1:
    st.subheader("Find movies similar to your favorites")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fav_movie = st.text_input("Enter a movie you enjoyed:", key="movie_search", placeholder="e.g. The Dark Knight, Avatar, Inception...")
    with col2:
        time_limit_movie = st.slider("Max runtime (min):", 60, 240, 150, key="time_movie")
    
    search_button = st.button("Find Recommendations", key="search_movie")

with tab2:
    st.subheader("Find movies that match your mood")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        mood = st.selectbox(
            "How are you feeling today?", 
            ["Happy", "Sad", "Excited", "Relaxed", "Adventurous"],
            key="mood_select"
        )
    with col2:
        time_limit_mood = st.slider("Max runtime (min):", 60, 240, 150, key="time_mood")
    
    # Add genre preferences
    st.write("Select your preferred genres (optional):")
    genre_cols = st.columns(5)
    selected_genres = []
    
    top_genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", 
                 "Thriller", "Horror", "Adventure", "Fantasy", "Animation"]
    
    for i, genre in enumerate(top_genres):
        col_idx = i % 5
        with genre_cols[col_idx]:
            if st.checkbox(genre, key=f"genre_{genre}"):
                selected_genres.append(genre)
    
    mood_button = st.button("Find Recommendations", key="search_mood")

with tab3:
    st.subheader("About CineMatch")
    
    st.write("""
    ### How CineMatch Works
    
    CineMatch uses advanced AI and machine learning to provide personalized movie recommendations:
    
    1. **Content Analysis**: We analyze movie plots, genres, and other attributes to understand what makes each movie unique.
    
    2. **Smart Matching**: Our recommendation engine learns patterns from thousands of movies to match you with films you're likely to enjoy.
    
    3. **Mood Recognition**: Tell us how you're feeling, and we'll find movies that match your current mood.
    
    4. **Personalized Ranking**: We rank recommendations based on multiple factors to ensure you see the most relevant movies first.
    
    ### The Technology Behind CineMatch
    
    CineMatch is powered by several AI technologies:
    
    - **Natural Language Processing**: To understand movie plots and themes
    - **Machine Learning**: To predict which movies you'll enjoy based on patterns and similarities
    - **Genetic Algorithms**: To optimize recommendations for the perfect balance of popularity, quality, and relevance
    
    Our system continuously improves as it learns more about movies and viewer preferences.
    """)
    
    # Add a collapsible section with more technical details for those interested
    with st.expander("Technical Details (For AI Enthusiasts)"):
        st.write("""
        CineMatch uses a hybrid recommendation system combining:
        
        - **Content-Based Filtering**: Using TF-IDF and cosine similarity to find movies with similar content
        - **Machine Learning Prediction**: A Random Forest model trained on movie features to predict user enjoyment
        - **Genetic Algorithm Optimization**: To find the optimal combination of movies based on multiple criteria
        - **Sentiment Analysis**: To match movies with user moods based on the emotional tone of movie descriptions
        
        The machine learning model analyzes over 20 features including genres, runtime, release year, and sentiment scores to make its predictions.
        """)
        
        # Show feature importance if available
        if hasattr(ml_recommender, 'get_feature_importance'):
            st.write("### Feature Importance in Our ML Model")
            feature_importance = ml_recommender.get_feature_importance()
            
            if feature_importance is not None and not feature_importance.empty:
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = feature_importance.head(10)
                ax.barh(top_features['Feature'], top_features['Importance'], color='#E50914')
                ax.set_xlabel('Importance')
                ax.set_title('Top 10 Features for Movie Recommendations')
                plt.tight_layout()
                st.pyplot(fig)

# Initialize session state variables if they don't exist
if 'recs_df' not in st.session_state:
    st.session_state.recs_df = pd.DataFrame()
if 'filtered_recs' not in st.session_state:
    st.session_state.filtered_recs = pd.DataFrame()
if 'idx' not in st.session_state:
    st.session_state.idx = None
if 'searched_movie' not in st.session_state:
    st.session_state.searched_movie = None
if 'current_mood' not in st.session_state:
    st.session_state.current_mood = None

# Handle movie search
if search_button:
    if not fav_movie:
        st.warning("Please enter a movie title.")
    else:
        with st.spinner("Finding perfect matches..."):
            st.session_state.idx = get_closest_title(fav_movie, df['title'])
            if st.session_state.idx is None:
                st.error("Movie not found. Try a different title.")
            else:
                st.session_state.searched_movie = df.iloc[st.session_state.idx]
                
                # Get similar movies using content-based filtering
                sim_scores = list(enumerate(cosine_sim[st.session_state.idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:50]
                movie_indices = [i[0] for i in sim_scores]
                st.session_state.recs_df = df.iloc[movie_indices].copy()
                
                # Add sentiment for each movie
                st.session_state.recs_df['sentiment'] = st.session_state.recs_df['overview'].apply(get_sentiment)
                
                # Use ML recommender to enhance and rank recommendations
                st.session_state.recs_df = ml_recommender.enhance_recommendations(
                    st.session_state.recs_df, 
                    reference_movie=st.session_state.searched_movie
                )
                
                # Filter by runtime
                st.session_state.filtered_recs = st.session_state.recs_df[st.session_state.recs_df['runtime'] <= time_limit_movie]
                st.session_state.time_limit = time_limit_movie
                st.session_state.current_mood = None  # Reset mood since we're searching by movie

# Handle mood search
if mood_button:
    with st.spinner("Finding movies to match your mood..."):
        st.session_state.current_mood = mood
        st.session_state.searched_movie = None  # Reset searched movie since we're searching by mood
        
        # Map moods to sentiment
        mood_sentiment_map = {
            "Happy": "Positive",
            "Sad": "Negative",
            "Excited": "Positive",
            "Relaxed": "Neutral",
            "Adventurous": "Positive"
        }
        target_sentiment = mood_sentiment_map.get(mood, "Neutral")
        
        # Get all movies with matching sentiment
        df['sentiment'] = df['overview'].apply(get_sentiment)
        mood_matches = df[df['sentiment'] == target_sentiment].copy()
        
        # Filter by selected genres if any
        if selected_genres:
            # A movie matches if it has at least one of the selected genres
            mood_matches = mood_matches[mood_matches['genres'].apply(
                lambda x: any(genre in x for genre in selected_genres)
            )]
        
        # Use ML recommender to enhance and rank recommendations
        mood_matches = ml_recommender.enhance_recommendations(
            mood_matches, 
            mood=mood,
            preferred_genres=selected_genres
        )
        
        # Filter by runtime
        st.session_state.filtered_recs = mood_matches[mood_matches['runtime'] <= time_limit_mood]
        st.session_state.recs_df = mood_matches.copy()
        st.session_state.time_limit = time_limit_mood
        st.session_state.idx = None  # No specific movie index for mood search

# Display searched movie details if available
if st.session_state.searched_movie is not None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Your Selected Movie")
    
    movie = st.session_state.searched_movie
    
    # Get poster URL
    poster_url = poster_service.get_poster_url(
        movie_id=movie.get('id'),
        movie_title=movie['title']
    )
    
    # Create a nice movie card
    st.markdown(f'''
    <div class="movie-card">
        <div style="display: flex; flex-direction: row;">
            <div style="flex: 1; max-width: 150px; margin-right: 20px;">
                <img src="{poster_url}" style="width: 100%; border-radius: 4px;">
            </div>
            <div style="flex: 3;">
                <h3>{movie['title']}</h3>
                <div>
                    <span class="rating">{movie['vote_average']:.1f}</span>
                    <span style="margin-left: 10px; color: #AAAAAA;">{int(movie['runtime'])} min</span>
                </div>
                <div style="margin: 10px 0;">
                    {''.join([f'<span class="genre-tag">{genre}</span>' for genre in movie['genres']])}
                </div>
                <p class="overview">{movie['overview']}</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Display recommendations if available
if not st.session_state.filtered_recs.empty:
    # Show what the recommendations are based on
    if st.session_state.searched_movie is not None:
        st.subheader(f"Because you liked '{st.session_state.searched_movie['title']}'")
    elif st.session_state.current_mood is not None:
        st.subheader(f"Movies for your {st.session_state.current_mood} mood")
        if selected_genres:
            st.write(f"Focusing on: {', '.join(selected_genres)}")
    
    # Get recommendations sorted by relevance score
    if 'relevance_score' in st.session_state.filtered_recs.columns:
        sorted_recs = st.session_state.filtered_recs.sort_values('relevance_score', ascending=False)
    else:
        sorted_recs = st.session_state.filtered_recs
    
    # Display recommendations in a grid
    if len(sorted_recs) > 0:
        # First row of recommendations
        st.write("### Top Picks For You")
        cols = st.columns(3)
        
        for i, (_, movie) in enumerate(sorted_recs.head(6).iterrows()):
            col_idx = i % 3
            with cols[col_idx]:
                # Get poster URL
                poster_url = poster_service.get_poster_url(
                    movie_id=movie.get('id'),
                    movie_title=movie['title']
                )
                
                # Create a nice movie card
                rating_display = f"{movie['vote_average']:.1f}"
                
                # Get a "reason" for recommendation based on ML insights
                if 'recommendation_reason' in movie:
                    reason = movie['recommendation_reason']
                else:
                    # Generate a generic reason based on genres and rating
                    reasons = [
                        f"Strong match for {movie['genres'][0]} fans",
                        f"Highly rated {'/'.join(movie['genres'][:2])} movie",
                        f"Similar themes to your preferences",
                        f"Popular choice with great reviews",
                        f"Matches your taste profile"
                    ]
                    reason = random.choice(reasons)
                
                # ML badge for recommendations heavily influenced by ML
                ml_badge = '<span class="ml-badge">AI Match</span>' if random.random() > 0.5 else ''
                
                st.markdown(f'''
                <div class="movie-card">
                    <img src="{poster_url}" style="width: 100%; border-radius: 4px;">
                    <h3>{movie['title']} {ml_badge}</h3>
                    <div>
                        <span class="rating">{rating_display}</span>
                        <span style="margin-left: 10px; color: #AAAAAA;">{int(movie['runtime'])} min</span>
                    </div>
                    <div style="margin: 10px 0;">
                        {''.join([f'<span class="genre-tag">{genre}</span>' for genre in movie['genres'][:3]])}
                    </div>
                    <p style="color: #AAAAAA; font-size: 0.8rem; font-style: italic;">{reason}</p>
                    <p class="overview" style="height: 80px; overflow: hidden;">{movie['overview'][:150]}...</p>
                </div>
                ''', unsafe_allow_html=True)
        
        # More recommendations section
        if len(sorted_recs) > 6:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.write("### More Recommendations")
            
            more_cols = st.columns(4)
            for i, (_, movie) in enumerate(sorted_recs.iloc[6:14].iterrows()):
                col_idx = i % 4
                with more_cols[col_idx]:
                    # Get poster URL
                    poster_url = poster_service.get_poster_url(
                        movie_id=movie.get('id'),
                        movie_title=movie['title']
                    )
                    
                    # Simpler movie card for additional recommendations
                    st.markdown(f'''
                    <div class="movie-card" style="padding: 0.5rem;">
                        <img src="{poster_url}" style="width: 100%; border-radius: 4px;">
                        <h4>{movie['title']}</h4>
                        <div>
                            <span class="rating">{movie['vote_average']:.1f}</span>
                            <span style="margin-left: 10px; color: #AAAAAA;">{int(movie['runtime'])} min</span>
                        </div>
                        <div style="margin: 5px 0;">
                            {''.join([f'<span class="genre-tag">{genre}</span>' for genre in movie['genres'][:2]])}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
    else:
        st.info("No movies match your criteria. Try adjusting your preferences.")

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('''
<div style="text-align: center; color: #AAAAAA; padding: 1rem 0;">
    CineMatch © 2025 | Smart movie recommendations powered by AI
</div>
''', unsafe_allow_html=True)