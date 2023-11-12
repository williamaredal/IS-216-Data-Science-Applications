import pandas as pd
import numpy as np
import math
import re
from sklearn.preprocessing import MinMaxScaler

def extract_number(text):
    # if the row has a NaN value, set it to 0
    if pd.isna(text):
        return 0.0
    
    # if the string contains integer(s), return these
    match = re.search(r'\d+(\.\d+)?', text)
    if match:
        extractedInteger = float(match.group())
        return extractedInteger
    
    # if there is text, but no number, return 1
    else:
        return 1.0


def find_distance_to_city_center(latitude, longitude):
    # Latitude and longtitude distance to oslo 
    oslo_lat_long = [59.911491, 10.757933] # from this source: https://www.latlong.net/place/oslo-ostlandet-norway-14195.html
    distance_to_city_center = (abs(latitude - oslo_lat_long[0]) ** 2) + (abs(longitude - oslo_lat_long[1]) ** 2) ** 0.5

    return distance_to_city_center


def haversine_distance_to_city_center(latitude, longitude):
    # Hardcoded latitude and longitude example
    # Latitude and longtitude distance to oslo 
    oslo_lat, oslo_long = 59.911491, 10.757933 # from this source: https://www.latlong.net/place/oslo-ostlandet-norway-14195.html

    # Convert latitude and longitude from degrees to radians
    latitude, longitude, oslo_lat, oslo_long = map(math.radians, [latitude, longitude, oslo_lat, oslo_long])


    # Haversine formula
    dlat = oslo_lat - latitude
    dlon = oslo_long - longitude
    a = math.sin(dlat/2)**2 + math.cos(latitude) * math.cos(oslo_lat) * math.sin(dlon/2)**2

    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers

    return c * r  # Distance in kilometers


def has_kitchen(text):
    match = re.search(r'\bKitchen\b', text)
    if match:
        return 1
    
    else:
        return 0


def place_for_yourself(text):
    match = re.search(r'\bEntire home/apt\b', text)
    if match:
        return 1
    else:
        return 0


def is_host_superhost(text):
    if text == 't':
        return 1
    else:
        return 0


def count_amenities(text):
    amenities_count = len([amenity.strip() for amenity in text.replace('[', '').replace( ']', '').replace('"', '').split(',') if amenity != ''])

    return amenities_count



filename = 'listings-clean-william.csv'
dataFrame = pd.read_csv(filename, encoding='latin-1')

# Make the colums required for the model that are missing
# Makes column with the number of bathrooms
dataFrame['number_bathrooms'] = dataFrame['bathrooms_text'].apply(extract_number)

# Latitude and longtitude distance to oslo 
dataFrame['distance_to_city_center'] = dataFrame.apply(lambda row: haversine_distance_to_city_center(row['latitude'], row['longitude']), axis=1)

# Numerical representation of true/false if the listing amenities contains "Kitchen"
dataFrame['kitchen_availability'] = dataFrame['amenities'].apply(has_kitchen)

# Numerical representation of true/false if the listing room type is "place for yourself"
dataFrame['place_for_yourself'] = dataFrame['room_type'].apply(place_for_yourself)

# Converts superhost boolean values to numerical representation
dataFrame['host_is_superhost_numerical'] = dataFrame['host_is_superhost'].apply(is_host_superhost)

# adds new column with number of amenities count
dataFrame['amenities_count'] = dataFrame['amenities'].apply(count_amenities)



columns = [
    'price', 
    'distance_to_city_center',
    'bedrooms', 'beds', 'number_bathrooms',
    'kitchen_availability', 'place_for_yourself', 'amenities_count',  
    'review_scores_rating', 'number_of_reviews',  'host_is_superhost_numerical'

    # additional review types
    #'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',

    # additional columns than the ones we have in the google sheets document
    #'accommodates', 'reviews_per_month', 
]


# TODO look at the column edges to find strange data to remove
'''
for col in columns:
    print(f"\nInvestigation of column: {col}")
    print(f"column min value {min(dataFrame[col])}")
    print(f"column max value {max(dataFrame[col])}")
    print(dataFrame[col].value_counts())
'''


# Perform normalization 
df_normalized = dataFrame.copy()
for col in columns:
    # Extract non-NaN values and normalize them
    non_nan_values = dataFrame[col][dataFrame[col].notna()].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    # If the column should be inverted (1 - value)
    if col == 'distance_to_city_center':
        non_nan_values = 1 - scaler.fit_transform(non_nan_values)
    else:
        non_nan_values = scaler.fit_transform(non_nan_values)

    # Replace only the non-NaN values in the column with their normalized values
    df_normalized.loc[dataFrame[col].notna(), col] = np.squeeze(non_nan_values)

    # Now fill NaN values with 0
    df_normalized[col].fillna(0, inplace=True)


'''
for col in columns:
    print(f"\nInvestigation of column: {col}")
    print(f"column min value {min(df_normalized[col])}")
    print(f"column max value {max(df_normalized[col])}")
    print(df_normalized[col].value_counts())
'''


# Verification of the dataset through the filtering and normalisation process
'''
print(f"Original DataFrame rows: {len(dataFrame)}")
print(f"Dataframe normalised without NaN rows: {len(df_normalized)}")
'''

# Verifies that the final normalised dataFrame does not contain NaN values
normalised_dataFrame_nan_counts = df_normalized[columns].isnull().sum()
'''
print(f"\nNormalised dataFrame number of NaN values for each column:")
print(normalised_dataFrame_nan_counts)
'''
