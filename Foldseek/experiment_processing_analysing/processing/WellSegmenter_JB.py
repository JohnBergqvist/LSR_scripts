'''
(C) John Bergqvist 2025, adapted from James Marshall
'''

#%%
import os
from pathlib import Path
import pandas as pd
import cv2
from tierpsy.helper.params.tracker_param import SplitFOVParams
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter
from tierpsy.analysis.compress.selectVideoReader import selectVideoReader
import gc

#%%
def extract_and_save_well_videos(video_path, wells_df, output_dir, metadata):
    """
    Extract wells from the original video, downsample, and save each well as a separate video.
    Keeps only frames from 20s to 25s and takes every 5th frame.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = 20 * fps  # Frame index at 20 seconds
    end_frame = 25 * fps    # Frame index at 25 seconds

    # Ensure the video has enough frames
    if total_frames < end_frame:
        print(f"Warning: Video has fewer frames ({total_frames}) than expected ({end_frame}). Adjusting range.")
        end_frame = total_frames
        start_frame = max(0, end_frame - (5 * fps))  # Set start_frame to 5 seconds before end_frame, but not less than 0

    # Prepare video writers for each well
    video_writers = {}
    for idx, row in wells_df.iterrows():
        well_name = row['well_name'].decode("utf-8") if isinstance(row['well_name'], bytes) else str(row['well_name'])
        x_min, x_max = row['x_min'], row['x_max']
        y_min, y_max = row['y_min'], row['y_max']

        # Construct the output file name using metadata
        metadata_str = "_".join([f"{key}_{value}" for key, value in metadata.items()])
        output_file = os.path.join(output_dir, f"{well_name}_{metadata_str}.mp4")

        writer = cv2.VideoWriter(
            output_file,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps // 5,  # Adjust FPS to reflect frame skipping
            (x_max - x_min, y_max - y_min),
            isColor=False
        )
        video_writers[well_name] = writer

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Start at 20 seconds

    frame_idx = start_frame
    saved_frames = 0

    while cap.isOpened() and frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % 5 == 0:  # Take every 5th frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            for idx, row in wells_df.iterrows():
                well_name = row['well_name'].decode("utf-8") if isinstance(row['well_name'], bytes) else str(row['well_name'])
                x_min, x_max = row['x_min'], row['x_max']
                y_min, y_max = row['y_min'], row['y_max']
                well_frame = gray_frame[y_min:y_max, x_min:x_max]
                video_writers[well_name].write(well_frame)

            saved_frames += 1
            if saved_frames >= 25:  # Stop after saving 25 frames
                break

        frame_idx += 1

    cap.release()
    gc.collect()  # Force garbage collection to free memory
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


def process_videos_from_dataframe(df, processed_df, base_output_dir, processed_csv_path):
    """
    Process videos listed in the dataframe and save segmented well videos based on metadata.
    """
    # Merge ForProcessing with Processed.csv on DirPath
    df = df.merge(processed_df, on="DirPath", how="left").fillna({"Processed": False})
    df.rename(columns=lambda x: x.strip(), inplace=True)  # Remove leading/trailing spaces
    df['Processed'] = df['Processed'].astype(bool)  # Convert to boolean if necessary

    # Filter rows where 'Processed' is False and 'corrupt' is False
    df_to_process = df[(df["Processed"] == False) & (df["corrupt"] == False)]

    for _, row in df_to_process.iterrows():
        video_path = f"{row['DirPath']}/000000.mp4"  # Use the full path from the dataframe
        metadata = row.drop(labels=['DirPath', 'Processed', 'corrupt']).to_dict()  # Exclude 'DirPath' and 'Processed'

        # Construct the output directory dynamically using metadata
        metadata_str = "_".join([f"{key}_{value}" for key, value in metadata.items()])
        output_dir = os.path.join(base_output_dir, str(row['DateTaken']))
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing video: {video_path}")
        print(f"Saving to directory: {output_dir}")

        # Check if the video file is valid
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Skipping corrupt video file: {video_path}")
            cap.release()
            gc.collect()  # Force garbage collection to free memory
            continue
        cap.release()
        gc.collect() # Force garbage collection to free memory

        try:
            wells_df = extract_wells_from_video(video_path)
            if wells_df is not None:
                extract_and_save_well_videos(video_path, wells_df, output_dir, metadata)
                # Mark as processed and save immediately
                new_entry = pd.DataFrame([{"DirPath": row["DirPath"], "Processed": True}])
                processed_df = pd.concat([processed_df, new_entry], ignore_index=True)
                processed_df.to_csv(processed_csv_path, index=False)
                print(f"Updated Processed.csv for {row['DirPath']}.")
            else:
                print(f"No wells found for video: {video_path}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    print("Processing complete. All remaining files logged to Processed.csv.")


if __name__ == '__main__':
    # Hardcoded paths for running in VS Code
    for_processing_csv = "/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/B-CategorizedDirectories/ForProcessing.csv"  # Path to the ForProcessing CSV from the previous step
    processed_csv_path = "/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/C-WellSegmenter_files/Processed.csv"      # Path to the Processed CSV with info on which wells have already been processed
    base_output_dir = "/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/C-WellSegmenter_files"      # Output directory path

    # Load or initialize the processed CSV
    if os.path.exists(processed_csv_path):
        processed_df = pd.read_csv(processed_csv_path)
    else:
        processed_df = pd.DataFrame(columns=["DirPath", "Processed"])

    # Load the ForProcessing CSV
    df = pd.read_csv(for_processing_csv)

    # Process the videos
    process_videos_from_dataframe(df, processed_df, base_output_dir, processed_csv_path)

# %%
