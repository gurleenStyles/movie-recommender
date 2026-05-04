import numpy as np
from pathlib import Path
import os
from src.data_loader import load_raw_data, merge_datasets
from src.preprocessor import create_user_item_matrix, get_numpy_matrix
from src.similarity import calculate_similarity
from src.recommender import Recommender
def main():
    print("🎬 Starting Movie Recommender Pipeline...")
    
    # Pipeline Step 1: Load Data #data_loader
    print("Loading raw data...")
    movies, ratings = load_raw_data()
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings.")
    
    # Pipeline Step 2: Merge Data data loader
    print("Merging datasets...")
    merged_data = merge_datasets(movies, ratings)
    
    # Pipeline Step 3: Create User-Item Matrix preprocessor 
    print("Creating User-Item matrix...")
    user_item_matrix = create_user_item_matrix(merged_data)

       # Pipeline Step 4: Calculate Similarity
    numpy_matrix = get_numpy_matrix(user_item_matrix)
    similarity_matrix = calculate_similarity(numpy_matrix)
    
    print(f"Saving similarity matrix of size {similarity_matrix.shape} to disk...")
    # Dynamically find the absolute path to /data/processed
    project_root = Path(__file__).parent
    processed_dir = project_root / 'data' / 'processed'
    
    # Force create the directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save safely
    np.save(processed_dir / 'similarity_matrix.npy', similarity_matrix)

    # Extract NumPy array
    numpy_matrix = get_numpy_matrix(user_item_matrix)

    # Show the result to prove it works
    print("\nData Pipeline Output (First 5 rows):")
    print(merged_data.head())

    # Pipeline Step 5: Generate Recommendations
    print("\n--- Testing the Recommender Engine ---")
    rec_engine = Recommender(user_item_matrix, similarity_matrix)
    
    # Try getting 5 recommendations for User ID 10
    top_picks = rec_engine.recommend(target_user_id=10, top_x=5)
    
    print("\nTop 5 Movie Recommendations:")
    print(top_picks)
if __name__ == "__main__":
    main()