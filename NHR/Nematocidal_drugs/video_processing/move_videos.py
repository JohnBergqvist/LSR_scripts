import os
import shutil
import sys

def move_videos_to_folders(base_directory):
    for subdir, _, files in os.walk(base_directory):
        if 'plate_' in subdir:
            plate_directory = subdir.split('plate_')[0] + 'plate_' + subdir.split('plate_')[1][0]
            single_wells_directory = os.path.join(plate_directory, 'single_wells')
            multi_wells_directory = os.path.join(plate_directory, 'multi_wells')
            
            if not os.path.exists(single_wells_directory):
                os.makedirs(single_wells_directory)
            if not os.path.exists(multi_wells_directory):
                os.makedirs(multi_wells_directory)
            
            for file in files:
                if file.endswith('.avi'):
                    file_path = os.path.join(subdir, file)
                    parts = file.split('_')
                    well_location = parts[-1].split('.')[0]
                    
                    if len(well_location) >= 2 and well_location[0] in 'ABCDEFGH' and well_location[1:].isdigit():
                        # Move to single_wells directory
                        new_folder_path = os.path.join(single_wells_directory, well_location)
                        if not os.path.exists(new_folder_path):
                            os.makedirs(new_folder_path)
                        new_file_path = os.path.join(new_folder_path, file)
                        shutil.move(file_path, new_file_path)
                        print(f"Moved: {file_path} to {new_file_path}")
                    elif len(parts[-1]) == 10 and parts[-1].endswith('.avi') and parts[-1][-10:-4].isdigit():
                        # Move to multi_wells directory
                        well_location = parts[0] + '_' + parts[1]
                        new_folder_path = os.path.join(multi_wells_directory, well_location)
                        if not os.path.exists(new_folder_path):
                            os.makedirs(new_folder_path)
                        new_file_path = os.path.join(new_folder_path, file)
                        shutil.move(file_path, new_file_path)
                        print(f"Moved: {file_path} to {new_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python move_videos.py <base_directory>")
        # Example of base directory: /Users/jb3623/Desktop/basler/test/250225_48h
        # Inside this base directory, there should be subdirectories with the name 'plate_1', 'plate_2', etc.
        sys.exit(1)
    
    base_directory = sys.argv[1]
    move_videos_to_folders(base_directory)