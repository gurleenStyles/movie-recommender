from src.data_loader import load_raw_data, merge_datasets

def main():
    print("🎬 Starting Movie Recommender Pipeline...")
    
    # Pipeline Step 1: Load Data
    print("Loading raw data...")
    movies, ratings = load_raw_data()
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings.")
    
    # Pipeline Step 2: Merge Data
    print("Merging datasets...")
    merged_data = merge_datasets(movies, ratings)
    
    # Show the result to prove it works
    print("\nData Pipeline Output (First 5 rows):")
    print(merged_data.head())

if __name__ == "__main__":
    main()