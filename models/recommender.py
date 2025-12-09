#models/recommender.py
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from models.ga_optimizer import GAOptimizer
from utils import get_cluster_for_mood
import numpy as np

def recommend_movies(user_mood, fav_movie, watch_time_limit, movies_df, kmeans, genre_names):
    # Step 1: Get favorite movie index
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    fav_idx = indices.get(fav_movie)
    if fav_idx is None:
        raise ValueError("Favorite movie not found!")

    # Step 2: Get similar movies using TF-IDF + cosine similarity
    tfidf_matrix = tfidf.fit_transform(movies_df['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[fav_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_movie_indices = [i[0] for i in sim_scores[1:51]]  # Top 50 similar

    # Step 3: Get mood clusters
    clusters = get_cluster_for_mood(user_mood, kmeans, genre_names)

    # Step 4: Get GA recommendations
    optimizer = GAOptimizer(movies_df, chrom_length=min(5, len(movies_df)))  # Avoid chromosome length larger than dataset
    ga_recs = optimizer.run(top_n=5)  # Get top 5 GA recommendations
    ga_indices = ga_recs.index.tolist()

    # Step 5: Final filtering of recommendations
    final_candidates = movies_df.iloc[list(set(sim_movie_indices) & set(ga_indices))]
    final_candidates = final_candidates[final_candidates['genre_cluster'].isin(clusters)]

    # Step 6: Time constraint
    final_movies = final_candidates[final_candidates['runtime'] <= watch_time_limit]

    return final_movies[['title', 'vote_average', 'popularity', 'runtime']]
