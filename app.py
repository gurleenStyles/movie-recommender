from src.data_loader import load_raw_data, merge_datasets
from src.preprocessor import create_user_item_matrix, get_numpy_matrix

def main():
    print("🎬 Starting Movie Recommender Pipeline...")
    
    # Pipeline Step 1: Load Data
    print("Loading raw data...")
    movies, ratings = load_raw_data()
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings.")
    
    # Pipeline Step 2: Merge Data
    print("Merging datasets...")
    merged_data = merge_datasets(movies, ratings)
    
    # Pipeline Step 3: Create User-Item Matrix
    print("Creating User-Item matrix...")
    user_item_matrix = create_user_item_matrix(merged_data)
    
    # Extract NumPy array
    numpy_matrix = get_numpy_matrix(user_item_matrix)

    # Show the result to prove it works
    print("\nData Pipeline Output (First 5 rows):")
    print(merged_data.head())

if __name__ == "__main__":
    main()