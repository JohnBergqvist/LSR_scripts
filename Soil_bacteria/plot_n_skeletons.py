#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:09:04 2024

@author: bonnie
"""

#%% 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% 

features = pd.read_csv(
    '/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Results/features_summary_tierpsy_plate_20240904_162953.csv', 
    comment='#')

# Filter out rows where 'well_id' is between 0 and 125 (inclusive) which removes the data for imaging day 1 (20240710)
features = features[~features['file_id'].between(0, 125)]

# Generate a list of unique 'well_name' values sorted alphabetically
well_name_order = sorted(features['well_name'].unique())

plt.figure(figsize=(20, 8))  # Adjust the figure size as needed

# Create a boxplot with 'well_name' on the x-axis and 'n_skeletons' on the y-axis, ordered alphabetically
sns.boxplot(x='well_name', y='n_skeletons', data=features, order=well_name_order)

# Add a stripplot on top of the boxplot for better visualization of the data points, ordered alphabetically
sns.stripplot(x='well_name', y='n_skeletons', data=features, color='black', jitter=True, order=well_name_order)

plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability

plt.show()

#plt.savefig('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Figures/n_skeletons/n_skeletons_by_well.png', 
            #dpi=1000, bbox_inches='tight')

# %%
# Group by 'well_name' and calculate the mean of 'n_skeletons' for each group
mean_n_skeletons = features.groupby('well_name')['n_skeletons'].mean()

# Filter groups where the mean of 'n_skeletons' is less than 10,000
filtered_well_names = mean_n_skeletons[mean_n_skeletons < 10000].index

print(filtered_well_names)


# Filter the original DataFrame to include only the selected 'well_name' values
filtered_features = features[features['well_name'].isin(filtered_well_names)]

plt.figure(figsize=(20, 8))  # Adjust the figure size as needed

# Create a boxplot with 'well_name' on the x-axis and 'n_skeletons' on the y-axis for the filtered data
sns.boxplot(x='well_name', y='n_skeletons', data=filtered_features, order=sorted(filtered_well_names))

# Add a stripplot on top of the boxplot for better visualization of the data points, using the same order
sns.stripplot(x='well_name', y='n_skeletons', data=filtered_features, color='black', jitter=True, order=sorted(filtered_well_names))

plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability

plt.title('n_skeletons by well_name with Mean < 10,000')  # Add a title to the plot

plt.show()
#plt.savefig('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Figures/n_skeletons/n_skeletons_filtered_by_mean.png', 
            #dpi=1000, bbox_inches='tight')
# %%
