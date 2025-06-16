

#%%
import os
import cv2
import glob
import sys

def create_movie_from_tiffs(directory):
    # Get the list of subdirectories
    subdirs = [subdir for subdir, _, _ in os.walk(directory) if subdir != directory]
    print(f"Number of subdirectories to process: {len(subdirs)}")
    
    # Traverse the given directory and its subdirectories
    for subdir in subdirs:
        print(f"Processing subdirectory: {subdir}")
        
        # Find all TIFF files in the current subdirectory
        tiff_files = sorted(glob.glob(os.path.join(subdir, '*.tiff')))
        
        if not tiff_files:
            continue
        
        # Extract date and time from the first TIFF file name
        first_tiff_name = os.path.basename(tiff_files[0])
        date_time_str = first_tiff_name.split('__')[2]
        date_str = date_time_str[:8]
        time_str = date_time_str[9:15]
        
        # Read the first image to get the frame size
        first_frame = cv2.imread(tiff_files[0])
        if first_frame is None:
            print(f"Error reading the first frame from {tiff_files[0]}")
            continue
        
        height, width, layers = first_frame.shape
        size = (width, height)
        
        # Define the codec and create VideoWriter object
        subdir_name = os.path.basename(subdir)
        output_filename = f"{subdir_name}_{date_str}_{time_str}.avi"
        output_path = os.path.join(subdir, output_filename)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, size) # change the number for the frame rate
        
        for tiff_file in tiff_files:
            frame = cv2.imread(tiff_file)
            if frame is None:
                print(f"Error reading frame from {tiff_file}")
                continue
            out.write(frame)
        
        out.release()
        print(f"Video created for {subdir} at {output_path}")
    
    print("All videos have been made")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tiff_to_avi.py <directory_path>")
        sys.exit(1)
    
    main_directory = sys.argv[1]
    create_movie_from_tiffs(main_directory)
# %%