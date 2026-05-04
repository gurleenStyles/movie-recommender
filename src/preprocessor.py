import pandas as pd

def create_user_item_matrix(merged_df):
    """
    Pivots the DataFrame into a User-Item matrix.
    Rows: userId
    Columns: title
    Values: rating
    """
    print("Creating User-Item matrix...")
    
    # Create the pivot table
    matrix = pd.pivot_table(
        merged_df, 
        values='rating', 
        index='userId', 
        columns='title', 
        fill_value=0  # Fill missing ratings with 0
    )
    
    print(f"Matrix shape: {matrix.shape[0]} users x {matrix.shape[1]} movies")
    return matrix

def get_numpy_matrix(user_item_matrix):
    """Extracts the raw NumPy array from the Pandas DataFrame."""
    return user_item_matrix.values