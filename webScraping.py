import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Sample dataset: Users, Movies, and Ratings
data = {
    'User': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'D'],
    'Movie': ['Inception', 'Avengers', 'Titanic', 'Inception', 'Titanic',
              'Avengers', 'Titanic', 'Inception', 'Avengers', 'Joker'],
    'Rating': [5, 4, 2, 4, 5, 5, 3, 3, 2, 4]
}

df = pd.DataFrame(data)

# Pivot the data: Users as rows, Movies as columns
movie_matrix = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(movie_matrix)

# Compute similarity between users
similarity = cosine_similarity(scaled_data)

# Create a DataFrame for user similarity
similarity_df = pd.DataFrame(similarity, index=movie_matrix.index, columns=movie_matrix.index)

# Recommend movies for a target user
def recommend_movies(target_user, num_recommendations=2):
    print(f"\nRecommendations for User {target_user}:\n")
    
    # Get similar users
    similar_users = similarity_df[target_user].sort_values(ascending=False)[1:]  # exclude self

    # Movies watched by the target user
    user_movies = set(df[df['User'] == target_user]['Movie'])

    # Weighted movie scores
    weighted_scores = {}

    for user, sim_score in similar_users.items():
        user_data = df[df['User'] == user]
        for _, row in user_data.iterrows():
            if row['Movie'] not in user_movies:
                weighted_scores.setdefault(row['Movie'], 0)
                weighted_scores[row['Movie']] += sim_score * row['Rating']

    # Sort and recommend
    recommendations = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    for movie, score in recommendations[:num_recommendations]:
        print(f"{movie} (score: {round(score, 2)})")

# Example usage
recommend_movies('A')
