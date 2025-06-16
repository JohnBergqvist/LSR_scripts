'''
(C) John Bergqvist 2025 - Adapted from James Marshall
'''

#%%
import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import re

# Imports from tierpsy
from tierpsy.helper.params.tracker_param import SplitFOVParams
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter
from tierpsy.analysis.compress.selectVideoReader import selectVideoReader

#%%
def extract_and_save_well_videos(video_path, wells_df, output_dir, airtable_run, airtable_plate, source_plate, culture_plateid, day, date):
    """
    Extract wells from the original video and save each well as a separate video.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Prepare video writers for each well
    video_writers = {}
    for idx, row in wells_df.iterrows():
        well_name = row['well_name'].decode("utf-8") if isinstance(row['well_name'], bytes) else str(row['well_name'])
        x_min, x_max = row['x_min'], row['x_max']
        y_min, y_max = row['y_min'], row['y_max']

        # Construct the output file name
        output_file = os.path.join(output_dir, f"fs5_{well_name}_{airtable_run}_{airtable_plate}_{source_plate}_{culture_plateid}_{day}_{date}.mp4")
        writer = cv2.VideoWriter(
            output_file,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (x_max - x_min, y_max - y_min),
            isColor=False
        )
        video_writers[well_name] = writer

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        for idx, row in wells_df.iterrows():
            well_name = row['well_name'].decode("utf-8") if isinstance(row['well_name'], bytes) else str(row['well_name'])
            x_min, x_max = row['x_min'], row['x_max']
            y_min, y_max = row['y_min'], row['y_max']
            well_frame = gray_frame[y_min:y_max, x_min:x_max]
            video_writers[well_name].write(well_frame)

    cap.release()
    for writer in video_writers.values():
        writer.release()


def _return_masked_image(Video_input, well_number=96):
    """
    Returns the FOVMultiWellsSplitter object for the given video.
    """
    if well_number == 96:
        json_fname = Path('HYDRA_96WP_UPRIGHT.json')
    elif well_number == 24:
        json_fname = Path('HYDRA_24WP_UPRIGHT.json')
    else:
        raise ValueError("Invalid well number. The code can only handle 24 and 96 wells.")

    splitfov_params = SplitFOVParams(json_file=json_fname)
    shape, edge_frac, sz_mm = splitfov_params.get_common_params()

    # Extract parameters from the video filename
    uid, rig, ch, mwp_map = splitfov_params.get_params_from_filename(Video_input)
    px2um = 12.4

    vid = selectVideoReader(str(Video_input))
    success, img = vid.read()
    if not success or img is None:
        raise RuntimeError(f"Error reading frame from video: {Video_input}")

    fovsplitter = FOVMultiWellsSplitter(
        img,
        microns_per_pixel=px2um,
        well_shape=shape,
        well_size_mm=sz_mm,
        well_masked_edge=edge_frac,
        camera_serial=uid,
        rig=rig,
        channel=ch,
        wells_map=mwp_map,
    )
    return fovsplitter


def extract_wells_from_video(video_path, well_number=96):
    """
    Extract well data using the _return_masked_image function.
    """
    print(f"Loading wells data from video: {video_path}")
    try:
        fovsplitter = _return_masked_image(video_path, well_number)
        wells_df = fovsplitter.get_wells_data()
        return wells_df
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None    

'''Function not called in James's script..
def process_videos_from_dataframe(df, base_output_dir):
    """
    Process videos listed in the dataframe and save segmented well videos based on 'ExpDay' and 'AssayStart'.
    """
    for _, row in df.iterrows():
        video_path = f"../{'/'.join(row['VideoPath'].split('/')[-5:])}"
        exp_day = row['ExpDay']
        assay_start = row['AssayStart']
        repeat = row['Repeat']
        plate = row['Plate']

        output_dir = os.path.join(base_output_dir, str(assay_start), f"Day{str(exp_day)}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing video: {video_path}")
        print(f"Saving to directory: {output_dir}")

        try:
            wells_df = extract_wells_from_video(video_path)
            if wells_df is not None:
                extract_and_save_well_videos(video_path, wells_df, output_dir, repeat, plate)
            else:
                print(f"No wells found for video: {video_path}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    print("Processing complete. All segmented videos have been saved.")
    '''


def categorize_video_directories(directory_path, output_dir, run_metadata_dir):
    """
    Categorize video directories into All, ForProcessing, and include a 'corrupt' column.

    Args:
        directory_path (str): Path to the root directory containing video directories.
        output_dir (str): Directory to save the resulting CSV files.
        run_metadata_dir (str): Path to the directory containing run_metadata CSV files.

    Returns:
        None
    """
    # Initialize lists for metadata
    metadata = []

    # Step 1: Collect all non-corrupt airtable_run values from run_metadata files
    non_corrupt_runs = set()
    for root, _, files in os.walk(run_metadata_dir):
        for file in files:
            if file.startswith("run_metadata_") and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    run_metadata_df = pd.read_csv(file_path)
                    if "airtable_run" in run_metadata_df.columns:
                        non_corrupt_runs.update(run_metadata_df["airtable_run"].dropna().astype(int).tolist())
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Step 2: Traverse directories and extract metadata
    for root, dirs, files in os.walk(directory_path):
        for dir_name in dirs:
            # Check if the directory contains 000000.mp4 files
            dir_path = os.path.join(root, dir_name)
            if not any(file == "000000.mp4" for file in os.listdir(dir_path)):
                continue

            # Match non-`_redo_` directories
            metadata_match = re.match(
                r"run_(\d+)_(\d{8})_(\d{6})\.(\d{8})",
                dir_name
            )

            if metadata_match:
                # Extract metadata from the directory name
                airtable_run = int(metadata_match.group(1))
                date_taken = metadata_match.group(2)       # Date (YYYYMMDD)
                time_taken = metadata_match.group(3)       # Time (HHMMSS)
                final_sequence = metadata_match.group(4)   # Final numeric sequence

                # Determine if the run is corrupt
                is_corrupt = airtable_run not in non_corrupt_runs

                metadata.append({
                    "DirPath": dir_path,
                    "airtable_run": airtable_run,
                    "DateTaken": date_taken,
                    "TimeTaken": time_taken,
                    "CameraSerial": final_sequence,
                    "corrupt": is_corrupt
                })

    # Step 3: Combine metadata into a single DataFrame
    all_videos_df = pd.DataFrame(metadata)

    # Sort for logical processing
    all_videos_df = all_videos_df.sort_values(
        by=["airtable_run", "DateTaken"],
        ascending=[True, True]
    )

    # Step 4: Save the DataFrame
    os.makedirs(output_dir, exist_ok=True)
    all_videos_df.to_csv(os.path.join(output_dir, "AllVideoFileDirectories.csv"), index=False)


    print("Categorization complete. DataFrame saved to:", output_dir)

    # Check which videos have already been processed by the 'WellSegmenter_JB.py' script by determining 
    # if, for the row in the column 'DirPath' has the boolean 'TRUE' in the column 'Processed' in the 
    # csv 'Processed.csv' in '3.WellSegmenter_files'.
    # Make a new csv file called 'ForProcessing.csv' with the same columns as 'AllVideoFileDirectories.csv'
    # but with an additional column 'Processed' that is True if the video has been processed and False otherwise.
    
    # Path to the WellSegmenter processed CSV
    well_seg_processed_csv_path = "/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/C-WellSegmenter_files/Processed.csv"

    # Check if the processed CSV exists
    if os.path.exists(well_seg_processed_csv_path):
        processed_df = pd.read_csv(well_seg_processed_csv_path)
        all_videos_df["WellSeg_Processed"] = all_videos_df["DirPath"].isin(processed_df["DirPath"])
    else:
        all_videos_df["WellSeg_Processed"] = False

    # Filter rows where 'Processed' is False
    for_processing_df = all_videos_df[all_videos_df["WellSeg_Processed"] == False]

    # Remove the 'WellSeg_Processed' column
    for_processing_df = for_processing_df.drop(columns=["WellSeg_Processed"])

    # Save the filtered DataFrame to 'ForProcessing.csv'
    for_processing_csv_path = os.path.join(output_csv_dir, "ForProcessing.csv")
    for_processing_df.to_csv(for_processing_csv_path, index=False)

    print(f"'ForProcessing.csv' created with {len(for_processing_df)} unprocessed files at {for_processing_csv_path}.")

#%%


if __name__ == "__main__":

    raw_videos_directory = "/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/RawVideos"
    output_csv_dir = "/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/B-CategorizedDirectories"
    run_metadata_dir = "/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/A-Well_information"
    # Path to save the resulting CSV
    categorize_video_directories(raw_videos_directory, output_csv_dir, run_metadata_dir)


# %%
