import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PosterService:
    """Service to fetch movie posters from TMDB API"""
    
    def __init__(self):
        self.api_key = os.getenv("TMDB_API_KEY", "")
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/"
        self.poster_size = "w500"  # Options: w92, w154, w185, w342, w500, w780, original
        self.backdrop_size = "w1280"  # For backdrop images
        
        # Cache to avoid repeated API calls
        self.poster_cache = {}
        self.backdrop_cache = {}
        
        # Verify API key is available
        if not self.api_key:
            print("Warning: TMDB_API_KEY not found in environment variables.")
    
    def get_poster_url(self, movie_id=None, movie_title=None, fallback_url=None):
        """
        Get poster URL for a movie by ID or title.
        
        Parameters:
        -----------
        movie_id : int, optional
            TMDB movie ID
        movie_title : str, optional
            Movie title to search for
        fallback_url : str, optional
            URL to use if poster not found
            
        Returns:
        --------
        str
            URL to movie poster image
        """
        # Check cache first
        cache_key = f"id_{movie_id}" if movie_id else f"title_{movie_title}"
        if cache_key in self.poster_cache:
            return self.poster_cache[cache_key]
        
        # If no API key, return fallback
        if not self.api_key:
            return fallback_url or f"https://via.placeholder.com/500x750?text={movie_title or 'Movie+Poster'}"
        
        try:
            # If we have movie_id, get details directly
            if movie_id:
                url = f"{self.base_url}/movie/{movie_id}?api_key={self.api_key}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('poster_path'):
                        poster_url = f"{self.image_base_url}{self.poster_size}{data['poster_path']}"
                        self.poster_cache[cache_key] = poster_url
                        return poster_url
            
            # If we have title, search for the movie
            elif movie_title:
                url = f"{self.base_url}/search/movie?api_key={self.api_key}&query={movie_title}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results') and len(data['results']) > 0:
                        if data['results'][0].get('poster_path'):
                            poster_url = f"{self.image_base_url}{self.poster_size}{data['results'][0]['poster_path']}"
                            self.poster_cache[cache_key] = poster_url
                            return poster_url
        
        except Exception as e:
            print(f"Error fetching poster: {e}")
        
        # If all else fails, return fallback
        fallback = fallback_url or f"https://via.placeholder.com/500x750?text={movie_title or 'Movie+Poster'}"
        self.poster_cache[cache_key] = fallback
        return fallback
    
    def get_backdrop_url(self, movie_id=None, movie_title=None, fallback_url=None):
        """Get backdrop (background) image URL for a movie"""
        # Check cache first
        cache_key = f"id_{movie_id}" if movie_id else f"title_{movie_title}"
        if cache_key in self.backdrop_cache:
            return self.backdrop_cache[cache_key]
        
        # If no API key, return fallback
        if not self.api_key:
            return fallback_url or f"https://via.placeholder.com/1280x720?text=Movie+Backdrop"
        
        try:
            # If we have movie_id, get details directly
            if movie_id:
                url = f"{self.base_url}/movie/{movie_id}?api_key={self.api_key}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('backdrop_path'):
                        backdrop_url = f"{self.image_base_url}{self.backdrop_size}{data['backdrop_path']}"
                        self.backdrop_cache[cache_key] = backdrop_url
                        return backdrop_url
            
            # If we have title, search for the movie
            elif movie_title:
                url = f"{self.base_url}/search/movie?api_key={self.api_key}&query={movie_title}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results') and len(data['results']) > 0:
                        if data['results'][0].get('backdrop_path'):
                            backdrop_url = f"{self.image_base_url}{self.backdrop_size}{data['results'][0]['backdrop_path']}"
                            self.backdrop_cache[cache_key] = backdrop_url
                            return backdrop_url
        
        except Exception as e:
            print(f"Error fetching backdrop: {e}")
        
        # If all else fails, return fallback
        fallback = fallback_url or f"https://via.placeholder.com/1280x720?text=Movie+Backdrop"
        self.backdrop_cache[cache_key] = fallback
        return fallback