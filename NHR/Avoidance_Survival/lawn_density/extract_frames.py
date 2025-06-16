#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export frames from a video file to a designated folder


@author: John B 
@date: 2024/08/15

"""

#%% Imports

import cv2

from pathlib import Path


#%%

# using the third video ('000003.mp4') of every video as that is when the Ca lawn is mostly visible. 
# the first frame of the third video is frame 54,000 (or 54,001)
video_path = Path('/Volumes/behavgenom$/John/data_exp_info/Candida/final/RawVideos/20240710/240710_run3_6h_ca_ca_20240710_162203.22956840/000003.mp4')
lawn = 'Ca'
output_dir = Path('/Users/jb3623/Desktop/lawn_training/training_images_b1/')

# Ensure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

#%%
# Open the video file
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    for i in range(50, 60):  # Loop from 1 to 100 (1, 26), (26, 51), (51, 76), (76, 101) (1, 51) (51, 101) 
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            print(f"Error: Could not read frame {i}.")
            break
        output_filename = output_dir / f'training_{lawn}_{i}.png'  # Define the output filename
        cv2.imwrite(str(output_filename), frame)  # Save the frame as a PNG file
    cap.release()  # Release the video capture object
    print("Extraction complete.")


# %% extract frames from all videos in a folder
# Define the base path where the folders are located
base_path = Path('/Volumes/behavgenom$/John/data_exp_info/Candida/final/RawVideos/20240711/')
output_dir = Path('/Users/jb3623/Desktop/lawn_training/20h/')
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

# Find all '000003.mp4' files within the specified folders
for video_path in base_path.glob('**/000000.mp4'):
    folder_name = video_path.parent.name
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        continue
    
    ret, frame = cap.read()  # Read the first frame (or any specific frame you want)
    if not ret:
        print(f"Error: Could not read frame from {video_path}.")
    else:
        output_filename = output_dir / f'{folder_name}.png'  # Save with the same name as the folder
        cv2.imwrite(str(output_filename), frame)
    
    cap.release()

print("Extraction complete.")

# %%
