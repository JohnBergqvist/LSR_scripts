# -*- coding: utf-8 -*-
'''

This script processes video files to calculate pixel variance over time for each well in a mobility assay experiment.
It extracts frames from the videos, applies Gaussian filtering, and computes the pixel differences between frames. The results are saved as images and numpy arrays for further analysis.


author: John Bergqvist
date: 2025-06-16

'''


#%%

#import libraries
import os
import numpy as np
import cv2
import pandas as pd
import seaborn as sns
from pathlib import Path
from skimage import filters, measure
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.draw import disk
from natsort import natsorted
from collections import defaultdict
from IPython.display import display



import statsmodels.api as sm
import statsmodels.formula.api as smf

#%%
# Find the number of frames in the video
test_video = '/Users/jb3623/Documents/250511_Analysis/pixel_var_FS/videos/D7_airtable_run_1558_DateTaken_20250508_TimeTaken_110630_CameraSerial_22956813.mp4'
v = cv2.VideoCapture(test_video)
total_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total number of frames in the video: {total_frames}")


#%%
# Parameters:
# Change these to suit the videos being processed
imout = 1 #figure output 1=yes, 0=no
tl = 0.8  # Binarisation threshold level factor starting level (experiment between 0.5 and 1.0 depending on brightness & contrast)
well_r = 457  # Well radius in pixels (confirmed with imageJ)
inc = 2  # Frame increment between frames to be subtracted
framestep=3 # frame interval between frame samples for variance 
sigma = 0.6  # Gaussian filter strength, the higher the stronger the filter. Decrease to preserve more detail. If video is noisy, increase the value.
framenums = 25  # Number of frames to be processed

#%%
# (1)indicate data location

# initialise dictionary holding data for various movies in given folder
parentF = r'/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/C-WellSegmenter_files'
os.chdir(parentF)
dayfolders = os.listdir(parentF)
dayfolders.sort()

# Directory to save difpics
difpics_dir = '/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/G-PixVar/difpics'
# Directory for savinf npy files
npy_dir = '/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/G-PixVar/npy_files'

# Ensure the difpics directory exists
if not os.path.exists(difpics_dir):
    os.makedirs(difpics_dir)

#try to load partial data back if existing:
#load data back

#%%

# USE THIS TO RESUME THE PROCESSING IF IT WAS INTERUPTED

# Initialize MovieVarData dictionary
MovieVarData = {}

# Check if npy_dir exists and contains any .npy files
if os.path.exists(npy_dir):
    npy_files = []
    for root, _, files in os.walk(npy_dir):
        for file in files:
            if file.endswith('.npy') and file != 'MovieVarData.npy':
                npy_files.append(os.path.join(root, file))
    
    if npy_files:
        print(f"Found {len(npy_files)} .npy files in {npy_dir} (including subdirectories). Loading data...")
        for npy_file in npy_files:
            try:
                # Load the .npy file
                data = np.load(npy_file, allow_pickle=True)
                
                # Check if the data is a dictionary (MovieVarData.npy) or an array (individual .npy files)
                if isinstance(data, dict):
                    MovieVarData.update(data)
                elif isinstance(data, np.ndarray):
                    # Extract the video name from the file path
                    video_name = os.path.splitext(os.path.basename(npy_file))[0]
                    MovieVarData[video_name] = data
                else:
                    print(f"Skipping file {npy_file}: Unexpected data format.")
            except Exception as e:
                print(f"Error loading file {npy_file}: {e}")
        print("Loaded existing MovieVarData from .npy files.")
        
        # Save the loaded MovieVarData as MovieVarData.npy
        movie_var_data_path = os.path.join(npy_dir, 'MovieVarData.npy')
        if os.path.exists(movie_var_data_path):
            os.remove(movie_var_data_path)
        np.save(movie_var_data_path, MovieVarData)
        print(f"Saved MovieVarData to {movie_var_data_path}")
    else:
        print("No .npy files found in the npy_dir or its subdirectories. Starting fresh.")
else:
    print("npy_dir does not exist. Starting fresh.")

#%%
# Perform the pixel variance analysis on the videos
# For each day folder, process the videos

#---main code:----
# (1) goes through folders of recording days and  loads movies 
# (2) finds X and Y offset from theoretical well centroids as given in paranmeter section
# (3) creates a mask for each well with corrected well centroid of given radius from paramter section 
# (4) subtracts pixel values from each well from next movie frame with interval defined in parameters
# ----------


# Change into the directory containing all videos
os.chdir(parentF)

# Get a list of all video files in the directory and its subdirectories
video_files = []
for root, dirs, files in os.walk(parentF):
    for file in files:
        if file.endswith('.mp4'):
            video_files.append(os.path.join(root, file))
video_files.sort()
print(f"Found {len(video_files)} video files.")

# Ensure the difpics and npy directories exist
if not os.path.exists(difpics_dir):
    os.makedirs(difpics_dir)
if not os.path.exists(npy_dir):
    os.makedirs(npy_dir)

# Load existing MovieVarData if available
MovieVarData = {}
movie_var_data_path = os.path.join(npy_dir, 'MovieVarData.npy')
if os.path.exists(movie_var_data_path):
    print(f"Loading existing MovieVarData from {movie_var_data_path}...")
    MovieVarData = np.load(movie_var_data_path, allow_pickle=True).item()
    print(f"Loaded {len(MovieVarData)} processed videos.")

# Get a list of already processed video identifiers
processed_videos = set()
for video_path in MovieVarData.keys():
    # Extract the common identifier from the .npy file path
    video_identifier = os.path.basename(video_path).split('_well_diffs')[0]
    processed_videos.add(video_identifier)

# Process each video file
for mc, video in enumerate(video_files, start=1):
    # Extract the common identifier from the .mp4 file path
    video_identifier = os.path.basename(video).split('.mp4')[0]

    # Skip already processed videos
    if video_identifier in processed_videos:
        print(f"Skipping already processed video: {video}")
        continue

    # Extract the relative subdirectory path of the video
    relative_path = os.path.relpath(os.path.dirname(video), parentF)
    
    # Create corresponding subdirectories in difpics_dir and npy_dir
    difpics_subdir = os.path.join(difpics_dir, relative_path)
    npy_subdir = os.path.join(npy_dir, relative_path)
    if not os.path.exists(difpics_subdir):
        os.makedirs(difpics_subdir)
    if not os.path.exists(npy_subdir):
        os.makedirs(npy_subdir)

    video_path = os.path.join(parentF, video)
    print(f"Processing video {mc}/{len(video_files)}: {video}")

    # Open the video file
    v = cv2.VideoCapture(video_path)
    total_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize well_diffs
    well_diffs = np.zeros(round(framenums / framestep) + 1)
    FC = 1

    for fr in range(1, framenums, framestep):  # pixel variance every "framestep" frames
        if FC == 1:
            # Set the video position to the desired frame
            v.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ret, frame = v.read()
            if not ret:
                print('Could not read frame from video')
                break

            # Create mask
            fn = frame[::2, ::2, 0]
            ps = fn.shape
            rr, cc = disk((round(ps[0] / 2), round(ps[1] / 2)), round(ps[0] / 2.5), shape=ps)
            mask = np.zeros(fn.shape, dtype=bool)
            mask[rr, cc] = 1

        # Calculate pixel variance over time
        diff_pix = [None] * 3

        # Set the video position to the desired frame
        v.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ret, frame_o = v.read()
        if not ret:
            print('Could not read frame from video')
            break

        pic_o = frame_o[:, :, 0].astype(float)
        pic_of = gaussian_filter(pic_o, sigma)

        if fr + inc >= total_frames:
            print('Frame index exceeds total number of frames in the video')
            break

        v.set(cv2.CAP_PROP_POS_FRAMES, fr + inc)
        ret, frame_1 = v.read()
        if not ret:
            print('Could not read frame from video')
            break

        pic1 = frame_1[:, :, 0].astype(float)
        pic1f = gaussian_filter(pic1, sigma)

        pic1 = pic1f[::2, ::2]
        pic_o = pic_of[::2, ::2]
        difpic = np.abs(pic1 - pic_o)

        mi = np.nonzero(mask)
        diff_pix = difpic[mi]
        dpn = len(diff_pix[diff_pix > 4])  # number of pixels with diff > 4
        difp_norm = dpn / sum(sum(mask))  # normalized to number of pixels in well

        if FC == 1 and imout == 1:
            plt.imshow(difpic)
            plt.colorbar()
            # Save the difpic in the corresponding subdirectory
            imname = os.path.join(difpics_subdir, f'pix_var_{os.path.basename(video)}.png')
            plt.title(video)
            plt.savefig(imname, dpi=300, format='png', bbox_inches='tight')
            plt.close()

        well_diffs[FC] = difp_norm
        FC += 1

        if FC % 4 == 0:
            print(f"Frames analyzed: {FC}")

    # Save well_diffs for the current video in the corresponding subdirectory
    npy_file_path = os.path.join(npy_subdir, f'{os.path.splitext(os.path.basename(video))[0]}_well_diffs.npy')
    np.save(npy_file_path, well_diffs)

    # Add well_diffs to MovieVarData
    MovieVarData[video] = well_diffs

# Save the updated MovieVarData.npy in the npy_dir
movie_var_data_path = os.path.join(npy_dir, 'MovieVarData.npy')
if os.path.exists(movie_var_data_path):
    os.remove(movie_var_data_path)
np.save(movie_var_data_path, MovieVarData)
print(f"Updated MovieVarData saved to {movie_var_data_path}")


# Close the video capture object
v.release()

# %%
# Convert the numpy arrays to dataframes and average the pixel variance values for each well across all timepoints (not averaging the replicate wells)


# Path to the NumPy file
movie_var_data_path = '/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/G-PixVar/npy_files/MovieVarData.npy'
def process_numpy_array(movie_var_data):
    # Initialize a list to store DataFrames for each video
    all_data = []

    # Iterate over the dictionary
    for video_name, values in movie_var_data.items():
        # Create a DataFrame for each video
        data = []
        for timepoint, value in enumerate(values):
            data.append({'video_name': video_name, 'timepoint': timepoint, 'value': value})

        df = pd.DataFrame(data)

        # Remove rows with timepoint 0
        df = df[df['timepoint'] != 0]

        # Append to the list of all data
        all_data.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(all_data, ignore_index=True)

    # Group by video_name and calculate the mean value for each video
    averaged_df = combined_df.groupby('video_name', as_index=False)['value'].mean()

    return averaged_df

# Load the NumPy array from the file
movie_var_data = np.load(movie_var_data_path, allow_pickle=True).item()

# Process the NumPy array
#combined_df, mean_values_by_well_run = process_numpy_array(movie_var_data)
averaged_df = process_numpy_array(movie_var_data)

def standardize_video_name(video_name):
    # If the name ends with '_well_diffs', strip it
    if video_name.endswith('_well_diffs'):
        return video_name.split('_well_diffs')[0]
    # If the name contains an absolute path, extract the base name and remove '.mp4'
    elif video_name.endswith('.mp4'):
        return os.path.basename(video_name).split('.mp4')[0]
    # Return the name as-is if it doesn't match the above patterns
    return video_name

# Apply the function to the 'video_name' column
averaged_df['video_name'] = averaged_df['video_name'].apply(standardize_video_name)

averaged_df['well_name'] = averaged_df['video_name'].str.extract(r'^([^_]+)')
averaged_df['airtable_run'] = averaged_df['video_name'].str.extract(r'(run_\d+)')
averaged_df['airtable_run'] = averaged_df['airtable_run'].astype(str)
# Display the DataFrame and average value
print(averaged_df)



#%%
# Combine the 'averaged_df' with the metadata 
metadata = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/A-Well_information/AllRunMetadata.csv')

# Ensure the keys are of the same type and clean
averaged_df['well_name'] = averaged_df['well_name'].astype(str).str.strip()
metadata['well_name'] = metadata['well_name'].astype(str).str.strip()
averaged_df['airtable_run'] = averaged_df['airtable_run'].astype(str).str.strip()

# Add 'run_' prefix to all values in 'airtable_run' in metadata
metadata['airtable_run'] = 'run_' + metadata['airtable_run'].astype(str).str.strip()

# Specify the columns to include from metadata
columns_to_include = ['well_name', 'airtable_run', 'source_plate', 'culture_plateid', 'date', 'day', 'run', 'label_for_plotting']
metadata_subset = metadata[columns_to_include]

# Merge the DataFrames
combined_df = pd.merge(
    averaged_df,
    metadata_subset,
    left_on=['well_name', 'airtable_run'],
    right_on=['well_name', 'airtable_run'],
    how='left'
)

# Check for NaN values in the merged DataFrame
print("Columns with NaN values after merge:")
print(combined_df.isna().sum())

# Check the resulting DataFrame
print(combined_df.head())
print(combined_df.columns)


combined_df.to_csv('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/data/final_merged_pixel_difference_results.csv', index=False)
