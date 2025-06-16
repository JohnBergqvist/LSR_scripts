#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determine worm density inside bacterial lawn

A script to determine the likelyhood of finding a worm inside vs. outside of a 
bacterial lawn. 


@author: John B 
@date: 2024/08/15

"""

#%% Imports
import numpy as np
import seaborn as sns
import cv2
import yaml
import sys
import h5py
import tqdm
import argparse
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
import glob
import multiprocessing as mp


from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter
from tierpsy.analysis.split_fov.helper import CAM2CH_df, serial2channel, parse_camera_serial
from tierpsy.analysis.compress.selectVideoReader import selectVideoReader


#%%

def feat2raw(featfilepath):
    """Get imgstore name that corresponds to a results video"""
    rawfilepath = Path(
        str(featfilepath.parent).replace('Results', 'RawVideos')
        ) / 'metadata.yaml'
    assert rawfilepath.exists(), f'Cannot find imgstore for {featfilepath}'
    return rawfilepath

# Function to plot coordinates on a frame
def plot_coordinates_on_frame(frame, coordinates):
    for index, row in coordinates.iterrows():
        center = (int(row['Center of the object_0']), int(row['Center of the object_1']))
        radius = int(row['radius'])
        cv2.circle(frame, center, radius, (0, 255, 0), 2)  # Green circle with thickness of 2
    return frame

def get_trajectory_data(featuresfilepath):
    """Read Tierpsy-generated featuresN file trajectories data and return
    the following info as a dataframe: ['x', 'y', 'frame_number', 'worm_id']"""
    with h5py.File(featuresfilepath, 'r') as f:
        df = pd.DataFrame({'x': f['trajectories_data']['coord_x'],
                           'y': f['trajectories_data']['coord_y'],
                           'frame_number': f['trajectories_data']['frame_number'],
                           'worm_id': f['trajectories_data']['worm_index_joined']})
    return df

def worms_in_circle(df, center, radius, frame_number):
    """Find worm IDs within a defined circle at a specific frame number."""
    frame_df = df[df['frame_number'] == frame_number]
    distances = np.sqrt((frame_df['x'] - center[0])**2 + (frame_df['y'] - center[1])**2)
    worms_in_circle = frame_df[distances <= radius]
    return worms_in_circle['worm_id'].tolist()

def worms_outside_circle(df, center, radius, frame_number):
    """Find worm IDs outside a defined circle at a specific frame number."""
    frame_df = df[df['frame_number'] == frame_number]
    distances = np.sqrt((frame_df['x'] - center[0])**2 + (frame_df['y'] - center[1])**2)
    worms_outside_circle = frame_df[distances > radius]
    return worms_outside_circle['worm_id'].tolist()

def get_frame_from_raw(rawvidname, frame_number):
    vid = cv2.VideoCapture(str(rawvidname))
    if not vid.isOpened():
        raise ValueError(f"Could not open video file {rawvidname}")
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Set the position to the desired frame_number
    status, frame = vid.read()  # Use the correct method to read the frame
    if not status:
        raise ValueError(f"Something went wrong while reading frame {frame_number} from {rawvidname}")
    vid.release()  # Don't forget to release the video capture object
    return frame


def get_total_frames(df):
    """
    Extracts the maximum value from the 'frame_number' column in the dataframe,
    which represents the total number of frames in the video.
    """
    return df['frame_number'].max()


# Function to calculate the distance
def compute_distance(chunk, center_x, center_y):
    chunk['distance_from_center'] = np.sqrt(
        (chunk['x'] - center_x)**2 + (chunk['y'] - center_y)**2
    )
    return chunk




#%% 
# Print circles around the lawn for **ALL** videos in a directory
'''
Here we focus on all the image frames specified in a specific directory and extract the worm IDs within the circle of each frame of the videos.
'''

'''TODO: 
1. match the coordinates from lawn_metadata.csv to the hdf5 files to get the worm_ids in the circle for each video.
2. create 3 different dataframes depending on the microbe_strain (in the filename?)
3. Calculate the percentage of worm_ids inside the circle for each microbe_strain.
4. Plot the density of worm_ids inside the circle for each microbe_strain.

This file does not have any worms in it so should be discarded from the analysis: 240719_run15_6h_ca_ca_20240719_142539.22956819_drawn_frame_000003
'''

# Load lawn coordinates
lawn_coordinates = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/Candida/final/Analysis/lawn_density/lawn_metadata/lawn_metadata.csv')

# Define the directory containing the videos
video_directory = Path('/Volumes/behavgenom$/John/data_exp_info/Candida/final/RawVideos')

# List all mp4 files in the directory
video_files = list(video_directory.rglob('000003.mp4'))

# Specify the frame number you want to retrieve and modify
frame_number = 500

# Iterate over each video file found
for video_file in video_files:
    # Extract imgstore_name from the video file path
    # This extraction depends on the specific format of your paths and imgstore_name
    # Adjust the slicing indices according to your path structure
    path_parts = video_file.parts
    imgstore_name = path_parts[-2]  # Adjust based on your directory structure

    # Find the matching row in lawn_coordinates
    matching_row = lawn_coordinates[lawn_coordinates['imgstore_name'] == imgstore_name]
    if matching_row.empty:
        print(f"No matching coordinates found for {imgstore_name}.")
        continue

    # Load the video
    cap = cv2.VideoCapture(str(video_file))
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}.")
        continue
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the specified frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number} from {video_file}.")
        continue
    
    # Extract circle parameters for the current video
    x_coord = int(matching_row['Center of the Skeleton_0'].values[0])
    y_coord = int(matching_row['Center of the Skeleton_1'].values[0])
    radius = int(matching_row['radius'].values[0])
    
    # Draw the circle on the frame
    center_coordinates = (x_coord, y_coord)
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2
    cv2.circle(frame, center_coordinates, radius, color, thickness)
    
    # Modify the output path to include the folder name in the image's filename
    folder_name = video_file.parent.name
    output_filename = f"{folder_name}_drawn_frame_{video_file.stem}.png"
    output_path = Path('/Volumes/behavgenom$/John/data_exp_info/Candida/final/Analysis/lawn_density/lawn_drawn') / output_filename
    
    # Save the frame as an image
    cv2.imwrite(str(output_path), frame)
    
    # Release the video capture object
    cap.release()

print("Processing completed.")


#%%

# For ALL videos in a directory

# Define the directory containing the feat files
featN_directory = Path('/Volumes/behavgenom$/John/data_exp_info/Candida/final/Results')

# List all metadata_featuresN.hdf5 files in the directory
featuresfilepath = list(featN_directory.rglob('metadata_featuresN.hdf5'))

# Define list of dates in the featN_directory 
dates = ['20240710', '20240719', '20240816']
microbe_strains = ['op50', 'jub134', 'ca_ca']

dfs = []

# Iterate over each video file found
for featN_file in featuresfilepath:
    # Extract imgstore_name from the featN_file file path
    # This extraction depends on the specific format of your paths and imgstore_name
    # Adjust the slicing indices according to your path structure
    path_parts = featN_file.parts
    imgstore_name = path_parts[-2]  # Adjust based on your directory structure

    # Find the matching row in lawn_coordinates
    matching_row = lawn_coordinates[lawn_coordinates['imgstore_name'] == imgstore_name]
    if matching_row.empty:
        print(f"No matching coordinates found for {imgstore_name}.")
    else:
        print(f"Matching coordinates found for {imgstore_name}.")
        # Since matching coordinates are found, get the trajectory data and append it to the list
        df = get_trajectory_data(featN_file)
        # add a new column to the dataframe with the imgstore_name
        df['imgstore_name'] = imgstore_name

        # Check if imgstore_name contains any of the specified dates and add it to the 'date' column
        date_found = None
        for date in dates:
            if date in imgstore_name:
                date_found = date
                break
        df['date'] = date_found

        # Check if imgstore_name contains any of the specified microbe_strains and add it to the 'microbe_strain' column
        microbe_found = None
        for microbe in microbe_strains:
            if microbe in imgstore_name:
                microbe_found = microbe
                break
        df['microbe_strain'] = microbe_found

        # Append the DataFrame to the list
        dfs.append(df)

dfs_df = pd.concat(dfs, ignore_index=True)

# Drop rows of missing values in the 'date' column (20h data)
dfs_df = dfs_df.dropna(subset=['date'])



#%%
# For ALL videos in a directory

# make a new dataframe from lawn_coordiantes with columns 'Center of the Skeleton_0', 
# 'Center of the Skeleton_1', 'radius', 'imgstore_name', 'date', 'microbe_strain', 'total_frames'

df_lawn = lawn_coordinates

df_lawn['date'] = None
df_lawn['microbe_strain'] = None

for index, row in df_lawn.iterrows():
    imgstore_name = row['imgstore_name']
    
    # Find and set the date
    date_found = None
    for date in dates:
        if date in imgstore_name:
            date_found = date
            break
    df_lawn.at[index, 'date'] = date_found
    
    # Find and set the microbe strain
    microbe_found = None
    for microbe in microbe_strains:
        if microbe in imgstore_name:
            microbe_found = microbe
            break
    df_lawn.at[index, 'microbe_strain'] = microbe_found

# Drop rows of missing values in the 'date' column (20h data)
df_lawn = df_lawn[df_lawn['date'].notnull()]

# Group by 'imgstore_name' in 'dfs' and calculate max frame number
max_frames_per_imgstore = dfs_df.groupby('imgstore_name')['frame_number'].max().reset_index()
max_frames_per_imgstore.rename(columns={'frame_number': 'total_frames'}, inplace=True)

# Merge the aggregated data with 'df_lawn' on 'imgstore_name'
df_lawn = df_lawn.merge(max_frames_per_imgstore, on='imgstore_name', how='left')


#%%

# Making the plots for the cumulative density of worms inside the circles
# For ALL videos in a directory


# Group by 'microbe_strain' & plot the cumulative frequency density of worms inside the circles
for microbe_strain, strain_df in dfs_df.groupby('microbe_strain'):
    strain_distance_df = pd.DataFrame()
    
    for imgstore_name, group_df in strain_df.groupby('imgstore_name'):
        # Retrieve center coordinates and radius for the current imgstore_name from df_lawn
        lawn_row = df_lawn[df_lawn['imgstore_name'] == imgstore_name].iloc[0]
        center_x, center_y, radius = lawn_row['Center of the Skeleton_0'], lawn_row['Center of the Skeleton_1'], lawn_row['radius']
        
        # Calculate distance from center for each worm in this group
        group_df['distance_from_center'] = np.sqrt((group_df['x'] - center_x)**2 + (group_df['y'] - center_y)**2)
        
        # Append to the strain_distance_df DataFrame
        strain_distance_df = pd.concat([strain_distance_df, group_df], ignore_index=True)

        #break  # Add this line to process only the first imgstore_name and then exit the loop

    # Calculate the average radius for the current microbe_strain
    average_radius = df_lawn[df_lawn['microbe_strain'] == microbe_strain]['radius'].mean()
    
    # Plot ECDF for the current microbe_strain using the strain_distance_df DataFrame
    plt.figure(figsize=(8, 6))
    sns.ecdfplot(data=strain_distance_df, x='distance_from_center', label=f'{microbe_strain} Cumulative Density')
    # Plot a vertical line using the average radius instead of the last retrieved radius
    plt.axvline(x=average_radius, color='k', linestyle='--', label=f'Average radius={average_radius:.2f}')
    
    
    # Calculate percentage inside for the current microbe_strain
    percentage_inside = (strain_distance_df['distance_from_center'] < radius).mean() * 100
    
    plt.text(average_radius, 0.9, f'{percentage_inside:.2f}% inside', verticalalignment='center', horizontalalignment='right', color='k', fontsize=10)
    
    plt.xlabel('Distance from Center')
    plt.ylabel('Cumulative Density')
    plt.title(f'Cumulative Density of Worms for {microbe_strain}')
    plt.legend()




    plt.show()


#%%
# Make one cumulative frequency density plot for op50 and jub134 in 'microbe_strain'
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define colors for op50 and jub134
colors = {
    'op50': '#EFE3B8',  # Light yellow
    'jub134': '#F0B64E'  # Light orange
}

op50_jub134_df = dfs_df[(dfs_df['microbe_strain'] == 'op50') | (dfs_df['microbe_strain'] == 'jub134')]

# Create a figure for the combined plot
plt.figure(figsize=(9, 6))

for microbe_strain, strain_df in op50_jub134_df.groupby('microbe_strain'):
    strain_distance_df = pd.DataFrame()
    
    for imgstore_name, group_df in strain_df.groupby('imgstore_name'):
        # Retrieve center coordinates and radius for the current imgstore_name from df_lawn
        lawn_row = df_lawn[df_lawn['imgstore_name'] == imgstore_name].iloc[0]
        center_x, center_y, radius = lawn_row['Center of the Skeleton_0'], lawn_row['Center of the Skeleton_1'], lawn_row['radius']
        
        # Calculate distance from center for each worm in this group
        group_df['distance_from_center'] = np.sqrt((group_df['x'] - center_x)**2 + (group_df['y'] - center_y)**2)
        
        # Append to the strain_distance_df DataFrame
        strain_distance_df = pd.concat([strain_distance_df, group_df], ignore_index=True)

    # Calculate the average radius for the current microbe_strain
    average_radius = df_lawn[df_lawn['microbe_strain'] == microbe_strain]['radius'].mean()
    


    # Plot ECDF for the current microbe_strain using the strain_distance_df DataFrame
    ecdf = sns.ecdfplot(data=strain_distance_df, x='distance_from_center', label=f'{microbe_strain} Cumulative Density', color=colors[microbe_strain], linewidth=2)
    
    # Plot a vertical line using the average radius instead of the last retrieved radius
    #plt.axvline(x=average_radius, color=colors[microbe_strain], linestyle='--', label=f'{microbe_strain} Average radius={average_radius:.2f}')
    


    # Calculate percentage inside for the current microbe_strain
    percentage_inside = (strain_distance_df['distance_from_center'] < radius).mean() * 100
    
    # Manually specify the y-coordinate for the text
    if microbe_strain == 'op50':
        y_value = 0.85  # Example y-coordinate for op50
        x_value = 350
    elif microbe_strain == 'jub134':
        y_value = 0.4  # Example y-coordinate for jub134
        x_value = 280
    
    plt.text(x_value, y_value, f'{percentage_inside:.2f}%\ninside ', verticalalignment='center', horizontalalignment='right', color=colors[microbe_strain], fontsize=11, weight='bold')

# Plot a vertical line at 401 for the radius (average between Jub134 and OP50)
plt.axvline(x=401, color='lightgrey', linestyle='--', label='radius=401')

plt.xlabel('Distance from Center')
plt.ylabel('Cumulative Density')
plt.title('Cumulative Density of Worms for OP50 and JUb134 (6h)')
plt.legend()

# Save the plot with high DPI
plt.savefig('/Volumes/behavgenom$/John/data_exp_info/Candida/final/Figures/cfd/cfd_op50_jub134.png', dpi=1000, bbox_inches='tight')

plt.show()



#%%
## Print circles on a **SINGLE** video frame

'''
Here we focus on a **SINGLE** image frame and extract the worm IDs within the circle of this single video.
'''

rawvidname = Path('/Volumes/behavgenom$/John/data_exp_info/Candida/final/RawVideos/20240719/240719_run15_6h_ca_ca_20240719_142539.22956814/000003.mp4')
frame_number = 500  # Specify the frame number you want to retrieve
	
center_coordinates = (1819, 1639)  # center coordinates for op50
##center_coordinates = (1350, 1650)  # center coordinates for jub134
radius = 400  
color = (0, 255, 0)  # Green color in BGR
thickness = 2

# Call get_frame_from_raw with the video file path and frame number
frame = get_frame_from_raw(rawvidname, frame_number)

if frame is not None:  # Check if a frame was successfully retrieved
    plt.imshow(frame)
    plt.title(f"Frame at {frame_number}")
    plt.axis('off')  # Hide axis for better visualization
    #plt.show()
else:
    print("Frame could not be read or does not exist.")

# draw a circle on the image frame

cv2.circle(frame, center_coordinates, radius, color, thickness)

# Display the modified frame
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
plt.title(f"Frame {frame_number} with circle around op50 lawn ")
plt.axis('off')  # Hide axis for better visualization
plt.show()




#%%
# Get the worm IDs within the circle at in a SINGLE video

# featuresfilepath for jub134: featuresfilepath = '/Volumes/behavgenom$/John/data_exp_info/Sm_avoidance_screen/240222/Results/20240222/240222_n2_op50jub134_3fps_run1_20240222_214700.22956805/metadata_featuresN.hdf5'
featuresfilepath = '/Volumes/behavgenom$/John/data_exp_info/Sm_avoidance_screen/240222/Results/20240222/240222_n2_op50jub134_3fps_run1_20240222_214700.22956818/metadata_featuresN.hdf5'
df = get_trajectory_data(featuresfilepath)



# print the worm_ids in the circle
lawn_worms = worms_in_circle(df, center_coordinates, radius, frame_number)
print(f"Worm IDs within the circle at frame {frame_number}: {lawn_worms}")




#%% Attempt to extract a list of the number of worm IDs in the circle for each frame in a SINGlE video

total_frames = get_total_frames(df)

# Initialize a set to keep track of unique worm_ids across all frames
unique_worm_ids_inside = set()

# Loop through all frames
for frame_number in range(1, total_frames + 1):  # Assuming frame_number starts from 1
    # Get the worm_ids in the circle for the current frame
    lawn_worms = worms_in_circle(df, center_coordinates, radius, frame_number)
    # Update the set of unique worm_ids with the new ids found in the current frame
    unique_worm_ids_inside.update(lawn_worms)

# Count the number of unique worm_ids
count_of_unique_worm_ids_inside = len(unique_worm_ids_inside)
print(f"Count of all unique worm IDs on the lawn: {count_of_unique_worm_ids_inside}")


# Initialize a set to keep track of unique worm_ids outside the circle
unique_worm_ids_outside = set()

# Loop through all frames
for frame_number in range(1, total_frames + 1):
    # Assuming a modified function or a new one that returns worm_ids outside the circle
    outside_worms = worms_outside_circle(df, center_coordinates, radius, frame_number)
    # Update the set of unique worm_ids outside the circle with the new ids found in the current frame
    unique_worm_ids_outside.update(outside_worms)

# Count the number of unique worm_ids outside the circle
count_of_unique_worm_ids_outside = len(unique_worm_ids_outside)
print(f"Count of all unique worm IDs outside the lawn: {count_of_unique_worm_ids_outside}")

# Calculate the percentage of unique worm_ids inside the circle
percentage_inside = (count_of_unique_worm_ids_inside / (count_of_unique_worm_ids_inside + count_of_unique_worm_ids_outside)) * 100
print(f"Percentage of worm IDs inside the circle: {percentage_inside:.2f}%")



# %% density plot of worm locations

# Step 1: Shift Coordinates
# Subtract center_coordinates from each worm's coordinates to re-center
df['x_shifted'] = df['x'] - center_coordinates[0]
df['y_shifted'] = df['y'] - center_coordinates[1]

# Step 2: Calculate Density
# Use gaussian_kde for density estimation on the shifted coordinates
xy = np.vstack([df['x_shifted'], df['y_shifted']])
density = gaussian_kde(xy)(xy)

#%%
# Step 3: Plot Density
# Create a scatter plot of the shifted coordinates colored by density
plt.scatter(df['x_shifted'], df['y_shifted'], c=density, s=50)
plt.colorbar(label='Density')
plt.xlabel('X (shifted)')
plt.ylabel('Y (shifted)')
plt.title('Density Plot of Worms Relative to Circle Center')
plt.axvline(x=0, color='r', linestyle='--')  # Mark the X=0 line
plt.axhline(y=0, color='r', linestyle='--')  # Mark the Y=0 line if needed
plt.show()




# %%
# Step 1: Calculate Distance from Center
# Calculate the Euclidean distance of each worm's coordinates from the center (combines x and y relative to the center)


df['distance_from_center'] = np.sqrt((df['x'] - center_coordinates[0])**2 + (df['y'] - center_coordinates[1])**2)

# Step 2: Calculate Density
# Sort distances and calculate density
sorted_distances = np.sort(df['distance_from_center'])
density = gaussian_kde(sorted_distances)(sorted_distances)
#%%

# Step 3: Plot
plt.figure(figsize=(8, 6))
plt.plot(sorted_distances, density, label='Density')
plt.axvline(x=400, color='k', linestyle='--', label='radius=400')  # Add dashed vertical line at x=400
plt.text(400 - 100, max(density) * 0.9, f'{percentage_inside:.2f}% inside', verticalalignment='center', horizontalalignment='right', color='k', fontsize=10)
plt.xlabel('Distance from Center')
plt.ylabel('Density')
plt.title('Density of Worms as a Function of Distance from Center')
plt.legend()
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.ecdfplot(data=df, x='distance_from_center', label='Cumulative Density')
plt.axvline(x=400, color='k', linestyle='--', label='radius=400')  # Add dashed vertical line at x=400

# Assuming percentage_inside is calculated elsewhere in your code
plt.text(400 - 100, 0.9, f'{percentage_inside:.2f}% inside', verticalalignment='center', horizontalalignment='right', color='k', fontsize=10)

plt.xlabel('Distance from Center')
plt.ylabel('Cumulative Density')
plt.title('Cumulative Density of Worms as a Function of Distance from Center')
plt.legend()
plt.show()
# %%
