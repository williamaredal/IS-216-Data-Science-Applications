import pandas as pd
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



filename = 'listings-clean.csv'
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
    'latitude', 'longitude', 'distance_to_city_center',
    'bedrooms', 'beds', 'number_bathrooms',
    'kitchen_availability', 'place_for_yourself', 'host_is_superhost_numerical', 'amenities_count',  
    'review_scores_rating', 'number_of_reviews',  'host_is_superhost_numerical'

    # additional review types
    #'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',

    # additional columns than the ones we have in the google sheets document
    #'accommodates', 'reviews_per_month', 
     
]



# Section for filtering out NaN values prior to normalisation
# Creates a mask for filtering NaN values prior to normalisation
mask = dataFrame[columns].notna().all(axis=1)

# Creates two dataframes, one containing the rows that have NaN values, the other excluding these rows
na_dataFrame = dataFrame[~mask].fillna(0)
filtered_dataFrame = dataFrame[mask]

'''
# Checks rows that contain NaN values
nan_counts = dataFrame[columns].isnull().sum()
print(f"\nThe number of rows containing NaN values for each column:")
print(nan_counts)
'''
'''
# Verifies that the dataframe does not contain NaN values
mask_filtered_nan_counts = filtered_dataFrame[columns].isnull().sum()
print(f"\nFiltered dataFrame number of NaN values for each column:")
print(mask_filtered_nan_counts)
'''

# Normalises the non-NaN value rows, then concats the NaN row dataframe with NaN replaced by zeros
scaler = MinMaxScaler()
normalised_dataFrame = pd.DataFrame(scaler.fit_transform(filtered_dataFrame[columns]), columns=columns)

# merges the na_dataFrame (with NaN values replaced with zeros)
normalised_dataFrame.reset_index(drop=True, inplace=True)
na_dataFrame.reset_index(drop=True, inplace=True)
normalised_and_na_dataFrame = pd.concat([normalised_dataFrame, na_dataFrame[columns]], axis=0, ignore_index=True)

'''
# Verifies that the final normalised dataFrame does not contain NaN values
normalised_dataFrame_nan_counts = normalised_dataFrame.isnull().sum()
print(f"\nNormalised dataFrame number of NaN values for each column:")
print(normalised_dataFrame_nan_counts)
'''


# TODO the closest_to_city_center should be inverted
# TODO look at the column edges to find strange data to remove
'''
for col in columns:
    print(f"Investigation of column: {col}")
    print(f"column min value {min(dataFrame[col])}")
    print(f"column max value {max(dataFrame[col])}")
    print(dataFrame[col].value_counts())
    print()
'''