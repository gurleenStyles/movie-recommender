import pandas as pd
from pathlib import Path

def get_data_dir():
    """Returns the absolute path to the data directory."""
    # This finds the directory of this script (src), then goes up one level to the root
    project_root = Path(__file__).parent.parent
    return project_root / 'data'

def load_raw_data():
    """Loads the movies and ratings datasets."""
    data_dir = get_data_dir()
    
    # Load CSVs
    movies_df = pd.read_csv(data_dir / 'raw' / 'movies.csv')
    ratings_df = pd.read_csv(data_dir / 'raw' / 'ratings.csv')
    
    return movies_df, ratings_df

def merge_datasets(movies_df, ratings_df):
    """Merges the ratings with movie titles for our pipeline."""
    # We only need movieId and title from the movies dataframe for now
    merged_df = pd.merge(ratings_df, movies_df[['movieId', 'title']], on='movieId')
    return merged_df