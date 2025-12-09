import numpy as np
import ast
from difflib import get_close_matches

# Define mood to genre preferences
mood_genre_pref = {
    "happy": ["Comedy", "Family", "Adventure"],
    "sad": ["Drama", "Romance"],
    "thrilling": ["Action", "Thriller", "Mystery"],
    "romantic": ["Romance", "Drama"],
    "adventurous": ["Adventure", "Fantasy", "Science Fiction"]
}

def parse_genre_string(genre_str):
    try:
        genre_list = ast.literal_eval(genre_str)
        return [g['name'] for g in genre_list if 'name' in g]
    except:
        return []

def preprocess_genres(genres_series):
    if isinstance(genres_series.iloc[0], list):
        # Already processed
        parsed = genres_series
    else:
        # Need to parse
        parsed = genres_series.apply(parse_genre_string)
    
    all_genres = sorted(set(g for sub in parsed for g in sub))
    genre_to_index = {genre: idx for idx, genre in enumerate(all_genres)}
    matrix = np.zeros((len(parsed), len(all_genres)))
    for i, genres in enumerate(parsed):
        for genre in genres:
            if genre in genre_to_index:
                matrix[i][genre_to_index[genre]] = 1
    return matrix, all_genres

def get_cluster_for_mood(mood, kmeans_model, genre_names):
    mood = mood.lower()
    genres = mood_genre_pref.get(mood, [])
    clusters = set()
    for genre in genres:
        vec = np.zeros((1, len(genre_names)))
        if genre in genre_names:
            vec[0][genre_names.index(genre)] = 1
        cluster = kmeans_model.predict(vec)[0]
        clusters.add(cluster)
    return list(clusters)

def get_closest_title(input_title, all_titles):
    matches = get_close_matches(input_title, all_titles, n=1, cutoff=0.6)
    if matches:
        return all_titles[all_titles == matches[0]].index[0]
    return None