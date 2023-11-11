import os
import re
import pandas as pd


# loads the dataset into a dataframe
filename = os.getcwd() + '/' + 'listings-clean.csv'
dataFrame = pd.read_csv(filename, encoding='latin-1')


# finds 'amenities' row that contains the most amount of amenities
most_amenities_index = dataFrame.amenities.str.len().idxmax()
most_amenities_length = dataFrame.amenities.str.len().max()


# puts every unique amenity in the dataset into this dictionary as a unique key
amenities_dictionary = {}
dataFrame['amenities'] = dataFrame['amenities'].str.replace('\\', '')
for row in dataFrame['amenities']:
        row_amenities = [amenity.strip() for amenity in row.replace('[', '').replace( ']', '').replace('"', '').split(',') if amenity != '']
        for amenity in row_amenities:
            amenities_dictionary[amenity] = 1

# adds column that has each row's number of amenities listed
dataFrame['amenities_count'] = dataFrame['amenities'].apply(lambda row: len([amenity.strip() for amenity in row.replace('[', '').replace( ']', '').replace('"', '').split(',') if amenity != '']))

# then makes a new dataset with new columns where the amenity match for that row is set as true/false for each amenity in the ameinties_dictionary key set
amenity_data = {'has_amenity_' + amenity: dataFrame['amenities'].str.contains(re.escape(amenity)) for amenity in amenities_dictionary.keys()}
ameinty_dataFrame = pd.DataFrame.from_dict(amenity_data)
dataFrame = pd.concat([dataFrame, ameinty_dataFrame], axis=1)


# prints for verification and testing 
#print(dataFrame['amenities'])
#print(f"min amenities: {min(dataFrame['amenities_count'])}")
#print(f"max amenities: {max(dataFrame['amenities_count'])}")
#print(dataFrame.columns)
#print(dataFrame.columns)
#print(dataFrame['has_amenity_Kitchen'].value_counts())
#print(dataFrame['has_amenity_Fire extinguisher'].value_counts())


# writes the new dataFrame containing the new columns of true/false for every amenity
out_file = os.getcwd() + '/' + 'listings-clean-amenities.csv'
dataFrame.to_csv(out_file)