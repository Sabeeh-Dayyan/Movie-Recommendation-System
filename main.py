import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sentiment.sentiment_analysis import get_sentiment
from models.ga_optimizer import genetic_algorithm
from utils import preprocess_genres, get_cluster_for_mood


# Load datasets
movies_df = pd.read_csv('data/tmdb_5000_movies.csv')
credits_df = pd.read_csv('data/tmdb_5000_credits.csv')

# Merge datasets
movies_df = movies_df.merge(credits_df, on='title')

# Preprocessing
movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['keywords'] = movies_df['keywords'].fillna('')
movies_df['description'] = movies_df['overview'] + ' ' + movies_df['keywords']
movies_df['genres'] = movies_df['genres'].apply(lambda x: [d['name'] for d in ast.literal_eval(x)])

# One-hot encode genres and cluster
genre_matrix, genre_names = preprocess_genres(movies_df['genres'])
kmeans = KMeans(n_clusters=5, random_state=42)
movies_df['genre_cluster'] = kmeans.fit_predict(genre_matrix)

# TF-IDF similarity
tfidf = TfidfVectorizer(stop_words='english')

::contentReference[oaicite:142]{index=142}
 
