# Book-recommendation-system-


Book Recommendation System Using K-Nearest Neighbors (KNN)

Overview
This project implements a book recommendation system using the K-Nearest Neighbors (KNN) algorithm.
 By leveraging book ratings, reviews, and other features, the system personalizes recommendations tailored to user preferences.
 It showcases key machine learning techniques like data preprocessing, normalization, and evaluation.

Key Features
Data Preprocessing: Handles missing values, cleans datasets, and normalizes rating distributions.
Book Recommendations: Generates personalized book suggestions using KNN with cosine similarity.
Evaluation Metrics: Implements Precision@K, Recall@K, and RMSE to assess recommendation quality.
Visualization: Creates insightful visualizations to explore data and understand recommendations.


To run this project, you need the following libraries installed in your Python environment:

numpy
pandas
matplotlib
seaborn
isbnlib
newspaper3k
lxml_html_clean
progressbar
scikit-learn


Install the required libraries using the following command:
pip install numpy pandas matplotlib seaborn isbnlib newspaper3k lxml_html_clean progressbar scikit-learn

Project Workflow
1. Data Loading and Initial Exploration
The dataset is loaded into a pandas DataFrame from a CSV file:


df = pd.read_csv('Upmerged_file.csv')
After loading the data, we inspect it by checking its shape, column names, and first few rows:

print(df.shape)
print(df.columns)
print(df.head())

2. Data Cleaning
Missing Values: Missing values in numerical columns are replaced with the median value, and categorical columns are filled with the most frequent value (mode).
Cleaning RatingDistTotal: The string "total:" is removed from the RatingDistTotal column, and the values are converted to integers.

3. Normalization
The rating columns (RatingDist1, RatingDist2) are normalized to a range of 0 to 5 using the MinMaxScaler from scikit-learn:

scaler = MinMaxScaler(feature_range=(0, 5))
df[rating_columns] = scaler.fit_transform(df[rating_columns])


4. Data Visualization
Several plots are generated to explore the dataset and understand the distribution of books, ratings, and other features:

Most Occurring Books: A bar plot is created to show the top 20 books with the highest occurrences.
Most Rated Books: A line chart is used to visualize the top 10 most rated books.
Top Authors: A bar plot displays the top authors based on high ratings and the number of books.
Top Languages: A bar chart is plotted for the top 12 most common languages in the dataset.


5. KNN-Based Recommendations
User-Item Matrix: A matrix is created where each row represents a user, and each column represents a book. The values are the ratings given by the users.
KNN Model: A KNN model is trained to find the most similar books based on cosine similarity between book ratings:

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(normalized_matrix)
Book Recommendations: The top K recommended books are retrieved by applying KNN on the user-item matrix, excluding the book itself.


6. Evaluation
Several evaluation metrics are computed to assess the quality of recommendations:

Precision at K: Measures the proportion of relevant items in the top K recommended books.
Recall at K: Measures the proportion of relevant items captured in the top K recommendations.
RMSE (Root Mean Squared Error): Measures the accuracy of the recommendation model by comparing predicted and actual ratings.



7. Recommendation Results
The final recommended books are displayed in a styled DataFrame for better visualization. The results include the book name, author, rating, and distance to the query book, sorted by similarity.

8. Saving Recommendations
The recommendations are saved into a CSV file for future reference:

final_recommendations[['Name', 'Authors', 'Rating', 'Distance']].to_csv('recommendations.csv', index=False)

Evaluation Metrics
1. Precision at K
Precision at K measures the proportion of relevant items in the top K recommendations.

def precision_at_k(recommended_items, relevant_items, k):
    top_k = recommended_items[:k]
    return len(set(top_k) & set(relevant_items)) / len(top_k)

2. Recall at K
Recall at K measures the proportion of relevant items captured in the top K recommendations.

def recall_at_k(recommended_items, relevant_items, k):
    top_k = recommended_items[:k]
    return len(set(top_k) & set(relevant_items)) / len(relevant_items)

3. RMSE (Root Mean Squared Error)
RMSE is calculated between true and predicted labels:
rmse = np.sqrt(mean_squared_error(true_labels, predicted_labels))


Future Work
Improving Data Quality: Enhance the dataset by including more user interactions.
Optimizing KNN: Experiment with different values of K and distance metrics for better recommendations.
Exploring Other Models: Implement other recommendation models such as Matrix Factorization or Collaborative Filtering.
File Outputs
recommendations.csv: A CSV file containing the recommended books.
