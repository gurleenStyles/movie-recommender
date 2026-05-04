import numpy as np
import pandas as pd

class Recommender:
    def __init__(self, user_item_matrix, similarity_matrix):
        """
        Initializes the recommender engine.
        user_item_matrix: Pandas DataFrame (Rows are users, Columns are movies)
        similarity_matrix: NumPy array of user-to-user similarity scores
        """
        self.user_item_matrix = user_item_matrix
        self.similarity_matrix = similarity_matrix
        
        # Keep track of mapping from user ID down to matrix index numbers
        self.user_ids = list(user_item_matrix.index)
        
    def recommend(self, target_user_id, top_x=5):
        """Generates recommendations for a specific user ID."""
        
        if target_user_id not in self.user_ids:
            return f"User {target_user_id} not found."
            
        print(f"\n Finding recommendations for User {target_user_id}...")
        
        # 1. Find the target user's row index in the matrix
        user_index = self.user_ids.index(target_user_id)
        
        # 2. Get their similarity scores with all other users
        user_similarities = self.similarity_matrix[user_index]
        
        # 3. Find the index of the MOST similar user
        # (We use argsort to sort indices, and take the 2nd to last element
        # because the last element is the user themselves with a score of 1.0)
        most_similar_user_idx = np.argsort(user_similarities)[-2]
        most_similar_user_id = self.user_ids[most_similar_user_idx]
        similarity_score = user_similarities[most_similar_user_idx]
        
        print(f"Nearest neighbor is User {most_similar_user_id} (Score: {similarity_score:.4f})")
        
        # 4. Find movies the target user has NOT seen (Rating == 0)
        target_user_ratings = self.user_item_matrix.loc[target_user_id]
        unseen_movies_mask = target_user_ratings == 0
        
        # 5. Get ratings from the most similar user, strictly for those unseen movies
        similar_user_ratings = self.user_item_matrix.loc[most_similar_user_id]
        recommendations = similar_user_ratings[unseen_movies_mask]
        
        # 6. Sort and return the highest rated movies!
        top_recommendations = recommendations.sort_values(ascending=False).head(top_x)
        
        return top_recommendations