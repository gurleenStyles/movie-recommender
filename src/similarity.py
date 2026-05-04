import numpy as np

def calculate_similarity(numpy_matrix):
    """
    Calculates the Cosine Similarity between all pairs of rows (Users).
    Returns a symmetrical matrix where the value at [i, j] is the 
    similarity score between user_i and user_j.
    """
    print("Calculating similarity engine (Vectorized)...")
    
    # 1. Add a tiny constant (epsilon) to prevent division by zero
    epsilon = 1e-9
    
    # 2. Calculate the dot product between every user and every other user simultaneously
    dot_product = np.dot(numpy_matrix, numpy_matrix.T)
    
    # 3. Calculate the magnitude (norm) of each user's row vector
    norms = np.linalg.norm(numpy_matrix, axis=1)
    
    # 4. Multiply the magnitudes together
    # We reshape norms so broadcasting correctly handles the matrix division
    norms_matrix = np.outer(norms, norms) + epsilon 
    
    # 5. Divide block 2 by block 4 to get the final similarities
    similarity_matrix = dot_product / norms_matrix
    
    print("Similarity calculations complete!")
    return similarity_matrix