"""
# create a hybrid recommender system for the airlines
# Passenger travel to destinationd D, which have attributes. 
# Passengers have attributes as well.
# The recommender system will recommend the top 5 destinations for each passenger
# based on the similarity of the passenger's attributes and the destination's attributes.
Create pseudo-code for such a recommender system.

1. Import necessary libraries and modules
2. Load passenger data and destination data

# Define necessary functions
3. function calculate_similarity(passenger, destination):
    a. Initialize similarity_score to 0
    b. For each attribute in passenger and destination:
        i.   Calculate similarity for the current attribute (e.g., using cosine similarity, Pearson correlation, etc.)
        ii.  Update similarity_score with the calculated similarity
    c. Return similarity_score

4. function get_top_n_recommendations(passenger, destinations, n):
    a. Initialize an empty list called recommendations
    b. For each destination in destinations:
        i.   Calculate similarity score between passenger and destination using calculate_similarity()
        ii.  Add destination and similarity_score to recommendations list
    c. Sort recommendations list based on similarity_score in descending order
    d. Return top n destinations from the sorted recommendations list

# Main program
5. Initialize an empty dictionary called all_recommendations
6. For each passenger in passenger data:
    a. Get top 5 destinations for the current passenger using get_top_n_recommendations()
    b. Add passenger ID and top 5 destinations to all_recommendations dictionary
7. Output all_recommendations
"""

# 1. Import necessary libraries and modules
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

# 2. Load passenger data and destination data
passengers = pd.read_csv('passengers.csv')
destinations = pd.read_csv('destinations.csv')

# 3. function calculate_similarity(passenger, destination):
def calculate_similarity(passenger, destination):
    # a. Initialize similarity_score to 0
    similarity_score = 0
    # b. For each attribute in passenger and destination:
    for attribute in passenger.index:
        # i.   Calculate similarity for the current attribute (e.g., using cosine similarity, Pearson correlation, etc.)
        similarity = cosine_similarity(np.array(passenger[attribute]).reshape(1, -1), np.array(destination[attribute]).reshape(1, -1))
        # ii.  Update similarity_score with the calculated similarity
        similarity_score += similarity
    # c. Return similarity_score
    return similarity_score

# 4. function get_top_n_recommendations(passenger, destinations, n):
def get_top_n_recommendations(passenger, destinations, n):
    # a. Initialize an empty list called recommendations
    recommendations = []
    # b. For each destination in destinations:
    for index, destination in destinations.iterrows():
        # i.   Calculate similarity score between passenger and destination using calculate_similarity()
        similarity_score = calculate_similarity(passenger, destination)
        # ii.  Add destination and similarity_score to recommendations list
        recommendations.append([index, similarity_score])
    # c. Sort recommendations list based on similarity_score in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)
    # d. Return top n destinations from the sorted recommendations list
    return recommendations[:n]

# 5. Initialize an empty dictionary called all_recommendations
all_recommendations = {}
# 6. For each passenger in passenger data:
for index, passenger in passengers.iterrows():
    # a. Get top 5 destinations for the current passenger using get_top_n_recommendations()
    recommendations = get_top_n_recommendations(passenger, destinations, 5)
    # b. Add passenger ID and top 5 destinations to all_recommendations dictionary
    all_recommendations[index] = recommendations
# 7. Output all_recommendations
print(all_recommendations)


# Rewrite above algorithm without using sklearn. 
# Instead, use the following formula for cosine similarity:
# cos_sim = (A . B) / (||A|| ||B||)
# where A and B are vectors, and ||A|| and ||B|| are the norms of A and B, respectively.
# Hint: Use numpy.linalg.norm() to calculate the norms of A and B.

# 1. Import necessary libraries and modules
