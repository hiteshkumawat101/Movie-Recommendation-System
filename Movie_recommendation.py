import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie data
movie_data = pd.read_csv('C:/Users/Hitesh/Downloads/movies.csv')

# Display the first few rows of the dataset
print(movie_data.head())

# Display the shape of the dataset
print(movie_data.shape)

# Display dataset statistics
print(movie_data.describe())

# Select features for recommendation
features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replace missing values with an empty string
for feature in features:
    movie_data[feature] = movie_data[feature].fillna('')

# Combine selected features into a single feature string for each movie
combined_features = movie_data['genres'] + ' ' + movie_data['keywords'] + ' ' + movie_data['tagline'] + ' ' + movie_data['cast'] + ' ' + movie_data['director']

# Transform the combined features into TF-IDF feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute cosine similarity between feature vectors
similarity = cosine_similarity(feature_vectors)

# Get user input for favorite movie
movie_name = input('Enter your favorite movie: ')

# Create a list of all movie titles
list_of_titles = movie_data['title'].tolist()

# Find the closest match to the user's input movie title
close_matches = difflib.get_close_matches(movie_name, list_of_titles)
if not close_matches:
    print("No close matches found. Please try again with a different movie title.")
    exit()

# Get the index of the movie that matches the closest
index_of_movie = movie_data[movie_data.title == close_matches[0]]['index'].values[0]

# Get a list of similar movies in descending order of similarity score
similar_movies = list(enumerate(similarity[index_of_movie]))
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

# Print the titles of the top 30 similar movies
print("Top 30 movies similar to " + close_matches[0] + " are:\n")
count = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title = movie_data[movie_data.index == index]['title'].values[0]
    if count <= 30:
        print(count, '.', title)
        count += 1
