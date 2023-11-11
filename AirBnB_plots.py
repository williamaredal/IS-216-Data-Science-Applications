import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MinMaxScaler


def extract_number(text):
    # if the row has a NaN value, set it to 0
    if pd.isna(text):
        return 0.0
    
    # if the string contains integer(s), return these
    match = re.search(r'\d+', text)
    if match:
        extractedInteger = float(match.group())
        return extractedInteger
    
    # if there is text, but no number, return 1
    else:
        return 1.0


def CorrelationalMatrixHeatmap(dataFrame, columns):
    subsetDF = dataFrame[columns]

    # Calculate the correlation matrix
    correlationMatrix = subsetDF.corr()

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(correlationMatrix, annot=True, cmap="coolwarm")
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.title("Correlation Heatmap")
    plt.show()


filename = 'listings-clean.csv'
dataFrame = pd.read_csv(filename, encoding='latin-1')
dataFrame['number_bathrooms'] = dataFrame['bathrooms_text'].apply(extract_number)

columns = [
    'price', 
    'latitude', 'longitude', 
    'accommodates', 'bedrooms', 'beds', 'number_bathrooms',
    'minimum_nights', 'maximum_nights',
    'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
    'review_scores_cleanliness', 'review_scores_checkin',
    'review_scores_communication', 'review_scores_location',
    'review_scores_value', # 'reviews_per_month', 'number_of_reviews',
    #'calculated_host_listings_count',
    #'calculated_host_listings_count_entire_homes',
    #'calculated_host_listings_count_private_rooms',
    #'calculated_host_listings_count_shared_rooms'
]


# Assuming df is your DataFrame
df_normalized = dataFrame.copy()

scaler = MinMaxScaler()
df_normalized[columns] = scaler.fit_transform(dataFrame[columns])


# plots heatmap of correlational matrix
CorrelationalMatrixHeatmap(dataFrame, columns)