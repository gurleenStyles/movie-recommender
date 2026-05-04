import numpy as np
from src.data_loader import load_raw_data, merge_datasets
from src.preprocessor import create_user_item_matrix, get_numpy_matrix
from src.similarity import calculate_similarity

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

    # Pipeline Step 4: Calculate Similarity similarity
    similarity_matrix = calculate_similarity(get_numpy_matrix(user_item_matrix))
    print(f"Saving similarity matrix of size {similarity_matrix.shape} to disk...")
    np.save('data/processed/similarity_matrix.npy', similarity_matrix)  # Saving to your processed folder instead of the root!
    
    # Extract NumPy array
    numpy_matrix = get_numpy_matrix(user_item_matrix)

    # Show the result to prove it works
    print("\nData Pipeline Output (First 5 rows):")
    print(merged_data.head())

if __name__ == "__main__":
    main()