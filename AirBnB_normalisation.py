import pandas as pd
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
    'review_scores_value', # 'reviews_per_month', 
    #'calculated_host_listings_count',
    #'calculated_host_listings_count_entire_homes',
    #'calculated_host_listings_count_private_rooms',
    #'calculated_host_listings_count_shared_rooms'
]

# TODO look at the column edges to find strange data to remove
nan_counts = dataFrame[columns].isnull().sum()
print()
print(f"The number of rows containing NaN values for each column:")
print(nan_counts)

for col in columns:
    print(f"Investigation of column: {col}")
    print(f"column min value {min(dataFrame[col])}")
    print(f"column max value {max(dataFrame[col])}")
    print()


# Make the colums required for the model that are missing
# Closest to city center,  Number of Baths (is this covered by number_bathrooms?), Kitchen Availability, 
# Amenities (is this covered by amenities_count?), host_is_superhost 

# Latitude and longtitude distance to oslo 
oslo_lat_long = [59.911491, 10.757933]
closest_to_city_center = 0
# Numerical representation of true/false if the listing amenities contains "Kitchen"
kitchen_availability = 0
# Numerical representation of true/false if the listing room type is not a "shared room" (should this include private room?)
not_shared_room = 0
# Numerical representation of true/false if the listing room type is "place for yourself"
place_for_yourself = 0
# Converts superhost boolean values to numerical representation
dataFrame['host_is_superhost_numerical'] = dataFrame['host_is_superhost'].apply(lambda row: 1 if row == 't' else 0)



# Assuming df is your DataFrame
df_normalized = dataFrame.copy()


# Find a mask for the NaN value rows/cells to ignore during normalisation
scaler = MinMaxScaler()
df_normalized[columns] = scaler.fit_transform(dataFrame[columns])
print(df_normalized[columns].head())