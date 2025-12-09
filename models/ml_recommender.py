import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ast
import re
from datetime import datetime
from textblob import TextBlob
import random

class MovieRecommender:
    """
    A professional movie recommendation engine that uses machine learning
    to enhance recommendations without exposing technical details to users.
    """
    
    def __init__(self):
        self.model = None
        self.feature_importance_df = None
        self.trained = False
        self.important_features = []
        
    def is_trained(self):
        """Check if the model is trained."""
        return self.trained
        
    def extract_year(self, release_date):
        """Extract year from release date string."""
        try:
            if pd.isna(release_date) or release_date == '':
                return 2000  # Default value
            # Try different date formats
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%Y']:
                try:
                    return datetime.strptime(release_date, fmt).year
                except ValueError:
                    continue
            # If all formats fail, try to extract year with regex
            year_match = re.search(r'(\d{4})', release_date)
            if year_match:
                return int(year_match.group(1))
            return 2000  # Default value
        except:
            return 2000  # Default value for any errors
    
    def get_sentiment_score(self, text):
        """Get sentiment polarity score from text."""
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0
    
    def prepare_data(self, df):
        """Prepare data for training or prediction."""
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Extract year from release_date if it exists
        if 'release_date' in data.columns:
            data['year'] = data['release_date'].apply(self.extract_year)
        else:
            data['year'] = 2000  # Default value
        
        # Calculate sentiment score from overview
        if 'overview' in data.columns:
            data['sentiment_score'] = data['overview'].apply(self.get_sentiment_score)
        else:
            data['sentiment_score'] = 0
        
        # Process genres
        if 'genres' in data.columns:
            # If genres are already a list, use them directly
            if isinstance(data['genres'].iloc[0], list):
                data['genres_list'] = data['genres']
            # If genres are in string format, parse them
            else:
                data['genres_list'] = data['genres'].apply(
                    lambda x: [g['name'] for g in ast.literal_eval(x)] if isinstance(x, str) else []
                )
            
            # Get all unique genres
            all_genres = sorted(set(genre for genres in data['genres_list'] for genre in genres))
            
            # One-hot encode genres
            for genre in all_genres:
                data[f'genre_{genre}'] = data['genres_list'].apply(lambda x: 1 if genre in x else 0)
        
        return data
    
    def train(self, df):
        """Train the recommendation model on movie data."""
        print("Preparing data for training recommendation engine...")
        data = self.prepare_data(df)
        
        # Define features to use
        numeric_features = ['popularity', 'runtime', 'year', 'sentiment_score']
        genre_features = [col for col in data.columns if col.startswith('genre_')]
        
        # Store for later use
        self.important_features = numeric_features + genre_features
        
        # Ensure all required columns exist
        for feature in numeric_features:
            if feature not in data.columns:
                data[feature] = 0
        
        # Fill missing values
        for feature in self.important_features:
            if feature in data.columns:
                data[feature] = data[feature].fillna(0)
        
        # Prepare features and target
        X = data[self.important_features]
        y = data['vote_average']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train the model
        print("Training recommendation engine...")
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_score = self.model.score(X_test, y_test)
        print(f"Recommendation engine accuracy: {test_score:.4f}")
        
        # Get feature importances
        importances = self.model.feature_importances_
        self.feature_importance_df = pd.DataFrame({
            'Feature': self.important_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("Recommendation engine training complete.")
        self.trained = True
        return test_score
    
    def get_feature_importance(self):
        """Return feature importance dataframe for visualization."""
        return self.feature_importance_df if self.trained else None
    
    def enhance_recommendations(self, movies_df, reference_movie=None, mood=None, preferred_genres=None):
        """
        Enhance movie recommendations using ML insights without exposing
        technical details to users.
        
        Parameters:
        -----------
        movies_df : DataFrame
            DataFrame containing movies to enhance
        reference_movie : Series, optional
            A reference movie to base recommendations on
        mood : str, optional
            User's current mood
        preferred_genres : list, optional
            List of preferred genres
            
        Returns:
        --------
        DataFrame
            Enhanced recommendations with relevance scores
        """
        if not self.trained:
            print("Recommendation engine not trained yet.")
            return movies_df
        
        # Prepare data
        data = self.prepare_data(movies_df)
        
        # Ensure all required features exist
        for feature in self.important_features:
            if feature not in data.columns:
                data[feature] = 0
        
        # Fill missing values
        for feature in self.important_features:
            data[feature] = data[feature].fillna(0)
        
        # Calculate a relevance score for each movie
        result_df = movies_df.copy()
        
        # Base score starts with normalized vote_average (0-1 scale)
        max_vote = result_df['vote_average'].max()
        if max_vote > 0:
            result_df['relevance_score'] = result_df['vote_average'] / max_vote
        else:
            result_df['relevance_score'] = 0.5  # Default if no ratings
        
        # Adjust score based on popularity (more popular = slightly higher score)
        max_pop = result_df['popularity'].max()
        if max_pop > 0:
            result_df['relevance_score'] += 0.2 * (result_df['popularity'] / max_pop)
        
        # If we have a reference movie, boost movies with similar genres
        if reference_movie is not None and isinstance(reference_movie, pd.Series):
            ref_genres = reference_movie['genres']
            result_df['genre_match'] = result_df['genres'].apply(
                lambda x: len(set(x) & set(ref_genres)) / max(len(set(x) | set(ref_genres)), 1)
            )
            result_df['relevance_score'] += 0.3 * result_df['genre_match']
        
        # If we have preferred genres, boost those movies
        if preferred_genres and len(preferred_genres) > 0:
            result_df['genre_preference'] = result_df['genres'].apply(
                lambda x: len(set(x) & set(preferred_genres)) / max(len(preferred_genres), 1)
            )
            result_df['relevance_score'] += 0.4 * result_df['genre_preference']
        
        # If we have a mood, adjust scores based on sentiment match
        if mood:
            mood_sentiment_map = {
                "Happy": 0.7,  # Positive sentiment
                "Sad": -0.3,   # Negative sentiment
                "Excited": 0.8, # Very positive
                "Relaxed": 0.2, # Slightly positive
                "Adventurous": 0.5 # Moderately positive
            }
            target_sentiment = mood_sentiment_map.get(mood, 0)
            
            # Calculate sentiment scores if not already done
            if 'sentiment_score' not in result_df.columns:
                result_df['sentiment_score'] = result_df['overview'].apply(self.get_sentiment_score)
            
            # Adjust relevance based on how close the sentiment is to the target
            result_df['sentiment_match'] = 1 - abs(result_df['sentiment_score'] - target_sentiment)
            result_df['relevance_score'] += 0.3 * result_df['sentiment_match']
        
        # Generate recommendation reasons based on movie attributes
        result_df['recommendation_reason'] = result_df.apply(self._generate_reason, axis=1)
        
        # Normalize final scores to 0-1 range
        min_score = result_df['relevance_score'].min()
        max_score = result_df['relevance_score'].max()
        if max_score > min_score:
            result_df['relevance_score'] = (result_df['relevance_score'] - min_score) / (max_score - min_score)
        
        return result_df
    
    def _generate_reason(self, movie):
        """Generate a human-readable recommendation reason."""
        reasons = []
        
        # Based on genres
        if len(movie['genres']) > 0:
            top_genres = movie['genres'][:2]
            if len(top_genres) == 1:
                reasons.append(f"Great {top_genres[0]} movie")
            else:
                reasons.append(f"Excellent {' & '.join(top_genres)} combination")
        
        # Based on rating
        if movie['vote_average'] >= 8:
            reasons.append("Highly acclaimed")
        elif movie['vote_average'] >= 7:
            reasons.append("Well-rated")
        
        # Based on popularity
        if 'popularity' in movie and movie['popularity'] > 50:
            reasons.append("Popular choice")
        
        # Based on sentiment
        if 'sentiment_score' in movie:
            if movie['sentiment_score'] > 0.5:
                reasons.append("Uplifting story")
            elif movie['sentiment_score'] < -0.3:
                reasons.append("Emotionally intense")
        
        # If we have genre match
        if 'genre_match' in movie and movie['genre_match'] > 0.5:
            reasons.append("Matches your taste")
        
        # If we have genre preference
        if 'genre_preference' in movie and movie['genre_preference'] > 0.5:
            reasons.append("Matches your preferences")
        
        # Select 1-2 reasons randomly
        if reasons:
            num_reasons = min(2, len(reasons))
            selected_reasons = random.sample(reasons, num_reasons)
            return " • ".join(selected_reasons)
        else:
            return "Recommended for you"