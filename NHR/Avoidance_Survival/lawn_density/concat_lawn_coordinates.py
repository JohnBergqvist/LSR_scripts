#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions to concatenate the x,y coordiantes of each lawn generated from 
ilastik object classification and segmentation


@author: John B 
@date: 2024/08/25

"""

#%% Imports


import os
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
import glob
import multiprocessing as mp

#%% Functions

data_path = Path('/Volumes/behavgenom$/John/data_exp_info/Candida/final/Analysis/lawn_density/Data')

# list all the csv files in the data directory
csv_files = glob.glob(str(data_path / '*.csv'))

# Initialize an empty list to store DataFrames
dfs = []

# Iterate through each CSV file
for file_path in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Extract the filename without the path and extension
    filename = Path(file_path).stem
    
    # Remove the ending '_table' from the filename
    filename = filename.replace('_table', '')

    # List of specific dates
    specific_dates = ['20240710', '20240711', '20240717', '20240719', '20240816']

    # Check if the filename contains any of the specific dates
    for date in specific_dates:
        if date in filename:
            # Prepend the date and a '/' to the filename
            filename = date + '/' + filename
            break  # Stop checking after the first match

    # Add a new column with the filename
    df['imgstore_name'] = filename

    # Create a new column 'radius' that is the sum of 'Radii of the object_0' and 'Radii of the object_1'
    df['radius'] = df['Radii of the object_0'] + df['Radii of the object_1']

     # Select only the specified columns
    df = df[['imgstore_name', 'Center of the Skeleton_0', 'Center of the Skeleton_1', 'radius']]
    
    	
    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
concatenated_df = pd.concat(dfs, ignore_index=True)

# set filepath for the lawn_metadata.csv file to be save in
meta_path = '/Users/jb3623/Desktop/lawn_training/lawn_metadata/lawn_metadata.csv'

# Save the concatenated DataFrame to 'lawn_metadata.csv'
concatenated_df.to_csv(meta_path, index=False)



# %%
