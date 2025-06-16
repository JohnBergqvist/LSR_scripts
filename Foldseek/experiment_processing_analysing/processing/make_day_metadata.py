'''
(C) John Bergqvist 2025

Make metadata files out of the AirTable excel forms to them all into one day metadata file and extract information 
from the pathnames and filenames:
    'run' - which repeat of the experiment - from the 'runX' folder
    'day' - which day of the experiment - from the 'YYYYMMDD_DX' folder
    'date' - the date of the experiment - from the 'YYYYMMDD_DX' folder
    'airtable_run' - AirTable assigned a run for each recording corresponding to one plate (sourced from excel filename)
    'airtable_plate' - AirTable assigned a plate for each recording corresponding to one plate (sourced from excel filename)
    'culture_plateid' - Which plate was used for the experiment (sourced from excel filename) 
        '1_1_A_2': Culture plate 1, colony 1, colony repeat A, repeat of colony plate 2
    'source_plate' - Which culture plate was used. Only 1 and 2, combined consists of all unique strains tested (sourced from excel filename).
    'strain_repeat' - repeat of the molecule being tested (sourced from treatment_1 column in the excel file).
'''


#%%
import numpy as np
import pandas as pd
import os 
import sys
from pathlib import Path
import re

#%%
# Directory to run, plate, and well information

PLATE_DIR = Path('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/A-Well_information')

# Dictionary to store DataFrames grouped by day
day_metadata = {}

# Dictionary to track mol_repeat counts for each treatment
treatment_repeat_counter = {}

# Recursively iterate through all Excel files in subdirectories
for excel_file in PLATE_DIR.rglob("*.xlsx"):
    # Skip files that do not match the expected filename pattern
    if not re.match(r'.*run_\d+_plate_\d+_FS\d+_cult_\d+.*\.xlsx$', excel_file.name):
        print(f"Skipping file {excel_file} as it does not match the expected pattern.")
        continue

    # Extract the run folder name (e.g., 'run1') and day folder name (e.g., '20250418_D0')
    run_folder = excel_file.parts[-3]  # Third-to-last part of the path
    day_folder = excel_file.parts[-2]  # Second-to-last part of the path
    day_name = day_folder.split('_')[-1]  # Extract 'D0' from '20250418_D0'
    date_yyyymmdd = day_folder.split('_')[0]  # Extract '20250418' from '20250418_D0'

    if day_folder not in day_metadata:
        day_metadata[day_folder] = {"dataframes": [], "parent_dir": excel_file.parent}  # Store DataFrames and parent directory

    # Read the Excel file, skipping the first 7 rows and using the 8th row as the header
    try:
        df = pd.read_excel(excel_file, skiprows=7, engine='openpyxl')

        # Extract information from the filename
        filename = excel_file.name
        airtable_run = int(re.search(r'run_(\d+)', filename).group(1))  # Extract '1167' from the airtable run 'run_1167'
        airtable_plate = re.search(r'(plate_\d+)', filename).group(1)  # Extract airtable 'plate_843'
        culture_plateid = re.search(r'cult_(\d+_\d+_[A-Z]_\d+)\.xlsx', filename).group(1)  # Extract '1_1_A_2'
        source_plate = re.search(r'(cult_\d+)', filename).group(1)  # Extract 'cult_1'
        

        # Add columns to the DataFrame
        df['run'] = run_folder
        df['day'] = day_name
        df['date'] = date_yyyymmdd
        df['airtable_run'] = airtable_run
        df['airtable_plate'] = airtable_plate
        df['culture_plateid'] = culture_plateid
        df['source_plate'] = source_plate

        # Initialize treatment_repeat_counter for each day
        treatment_repeat_counter = {}

        # Assign mol_repeat based on treatment_1
        strain_repeats = []
        for treatment in df['treatment_1']:
            if treatment not in treatment_repeat_counter:
                treatment_repeat_counter[treatment] = 1
            else:
                treatment_repeat_counter[treatment] += 1
            strain_repeats.append(treatment_repeat_counter[treatment])
        df['strain_repeat'] = strain_repeats

        # Append the DataFrame to the corresponding day's list
        day_metadata[day_folder]["dataframes"].append(df)
    except Exception as e:
        print(f"Error reading file {excel_file}: {e}")


# Combine all DataFrames for each day and save or process as needed
for day, metadata in day_metadata.items():
    combined_df = pd.concat(metadata["dataframes"], ignore_index=True)
    print(f"Processed data for {day}:")
    print(combined_df.head())

# Create individual metadata files for each day
for day, metadata in day_metadata.items():
    # Concatenate all DataFrames for the day
    day_metadata_combined = pd.concat(metadata["dataframes"], ignore_index=True)
    # Construct the output file path in the same directory as the source Excel files
    output_file = metadata["parent_dir"] / f"run_metadata_{day}.csv"
    # Save the combined DataFrame to a CSV file
    day_metadata_combined.to_csv(output_file, index=False)
    print(f"Collated metadata for {day} saved to {output_file}")


#%%
# Make one 'run' metadata file containing all the metadata files for each day in each individual 'run' folder

# Directory path
WELL_INFO_DIR = Path('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/A-Well_information')

# Iterate through all 'runX' subdirectories
for run_dir in WELL_INFO_DIR.rglob("run*"):
    # Check if the directory contains metadata files
    metadata_files = list(run_dir.rglob("run_metadata_*.csv"))
    if not metadata_files:
        print(f"No metadata files found in {run_dir}.")
        continue

    # List to store DataFrames for the current run
    run_metadata = []
    # Read and concatenate all metadata files for the current run
    for metadata_file in metadata_files:
        try:
            df = pd.read_csv(metadata_file)
            run_metadata.append(df)
        except Exception as e:
            print(f"Error reading file {metadata_file}: {e}")
    
    # Combine all DataFrames for the current run into one
    if run_metadata:  # Ensure there is data to combine
        run_metadata_df = pd.concat(run_metadata, ignore_index=True)
        # Save the combined DataFrame to a CSV file in the current run directory
        output_file = run_dir / f"AllRunMetadata_{run_dir.name}.csv"
        run_metadata_df.to_csv(output_file, index=False)
        print(f"All run metadata for {run_dir.name} saved to {output_file}")

# %%
# Make one metadata file containing all the run_* metadata files

# Directory path
WELL_INFO_DIR = Path('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/A-Well_information')

# List to store DataFrames for all runs
all_run_metadata = []

# Iterate through all immediate 'run*' subdirectories
for run_dir in WELL_INFO_DIR.iterdir():
    if run_dir.is_dir() and run_dir.name.startswith("run"):
        # Check if the directory contains metadata files
        metadata_files = list(run_dir.glob("AllRunMetadata_*.csv"))  # Use glob to avoid recursive search
        if not metadata_files:
            print(f"No metadata files found in {run_dir}.")
            continue

        # Read and concatenate all metadata files for the current run
        for metadata_file in metadata_files:
            try:
                df = pd.read_csv(metadata_file)
                all_run_metadata.append(df)
            except Exception as e:
                print(f"Error reading file {metadata_file}: {e}")

# Combine all DataFrames for all runs into one
if all_run_metadata:  # Ensure there is data to combine
    all_run_metadata_df = pd.concat(all_run_metadata, ignore_index=True)
    # Save the combined DataFrame to a CSV file in the parent directory of the first run directory
    output_file = WELL_INFO_DIR / "AllRunMetadata.csv"
    all_run_metadata_df.to_csv(output_file, index=False)
    print(f"All run metadata saved to {output_file}")
else:
    print("No metadata files found in any 'run*' subdirectory. No combined metadata file was created.")
# %%
