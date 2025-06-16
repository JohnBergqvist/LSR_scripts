'''
(C) John Bergqvist 2025

This script is for plotting the OD of the FoldSeek experiment

The script requires the ODs to be extracted first using the script 'OD_extraction.py'
'''

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



#%%

df = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/OD/combined_OD600_data.csv')

# %%

# Add a column called 'cult' to the dataframe which takes information from the 'plate_name' column
# The plate_name structure is either this 'FS5_Cult_1_1_A_1' or 'FS5_1_1_A_1'. 
# If the first number is 1, then the cult is 'Cult_1', if it is 2, then the cult is 'Cult_2'

def extract_cult_from_plate_name(df):
    if 'plate_name' in df.columns:
        df['plate_name'] = df['plate_name'].astype(str)  # Ensure column values are strings
        df['cult'] = df['plate_name'].str.extract(r'(\d)_\d_\w_\d')[0]  # Extract cult number
        df['cult'] = df['cult'].replace({'1': 'cult_1', '2': 'cult_2'})  # Map numbers to cult names
    else:
        print("Column 'plate_name' not found in the DataFrame.")
    return df
df = extract_cult_from_plate_name(df)


# Add a acolumn called 'culture_plateid' that consists of only '1_1_A_1' of the 'plate_name' column
def extract_culture_plateid(df):
    if 'plate_name' in df.columns:
        df['plate_name'] = df['plate_name'].astype(str)  # Ensure column values are strings
        df['culture_plateid'] = df['plate_name'].str.extract(r'(\d_\d_\w_\d)')[0]  # Extract culture plate ID
    else:
        print("Column 'plate_name' not found in the DataFrame.")
    return df
df = extract_culture_plateid(df)

# Change the 'Run' column data to only lowercase strings
def lowercase_run_column(df):
    if 'Run' in df.columns:
        df['Run'] = df['Run'].str.lower()  # Convert to lowercase
    else:
        print("Column 'Run' not found in the DataFrame.")
    return df
df = lowercase_run_column(df)


#%%
# Add the 'label_for_plotting' column from the metadata file 
metadata = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/AuxiliaryFiles/A-Well_information/AllRunMetadata.csv')

print(len(metadata))

# Change the column 'run' to 'Run' in the metadata file
metadata.rename(columns={'run': 'Run'}, inplace=True)
# Change the column 'well_name' to 'Well' in the metadata file
metadata.rename(columns={'well_name': 'Well'}, inplace=True)

# Extract the unique dates from 'date_yyyymmdd' column in the df file to a 'unique_dates' list
unique_dates = df['date_yyyymmdd'].unique().tolist()

#Filter out the rows in the metadata file that do not have a date in the 'date_yyyymmdd' column that is in the 'unique_dates' list
metadata = metadata[metadata['date'].isin(unique_dates)]
print(len(metadata))

#%%


# Merge on the 'Run', 'Well', and 'culture_plateid' columns
def merge_metadata(df, metadata):
    if 'Run' in df.columns and 'Well' in df.columns and 'culture_plateid' in df.columns:
        merged_df = pd.merge(df, metadata, on=['Run', 'Well', 'culture_plateid'], how='left')
        
        # Debugging: Check for NaN values in the merged column
        if 'label_for_plotting' in merged_df.columns:
            nan_count = merged_df['label_for_plotting'].isna().sum()
            print(f"'label_for_plotting' column contains {nan_count} NaN values after merge.")
        else:
            print("'label_for_plotting' column not found in merged DataFrame.")
        
        return merged_df
    else:
        print("Required columns for merging not found in the DataFrame.")
        return df

metadata_subset = metadata[['Run', 'Well', 'culture_plateid', 'label_for_plotting']]

# Perform the merge
df_merged = merge_metadata(df, metadata_subset)


# %%

# Make one plot per 'cult' displaying the mean OD600 value and the data variation from all the different runs
def plot_mean_od_by_cult(df):
    cults = df['cult'].unique()
    
    for cult in cults:
        cult_df = df[df['cult'] == cult]

        # Define the custom order for plotting
        custom_order = ['cry6A', 'no_bacteria_control', 'mScarlet_IPTG_control', 'empty_vector_pet24_control']
        # Add the rest of the labels specific to the current cult
        remaining_labels = [label for label in cult_df['label_for_plotting'].unique() if label not in custom_order]
        full_order = custom_order + remaining_labels

        plt.figure(figsize=(14, 8))
        
        # Create boxplot for OD600 values grouped by 'label_for_plotting'
        sns.boxplot(
            x='label_for_plotting', 
            y='OD600', 
            data=cult_df, 
            order=full_order,  # Use the custom order
            color='gray'
        )
        
        # Overlay stripplot with colored dots based on 'date'
        sns.stripplot(
            x='label_for_plotting', 
            y='OD600', 
            data=cult_df, 
            hue='date_yyyymmdd', 
            order=full_order,  # Use the custom order
            dodge=False, 
            palette='Set1', 
            size=5, 
            jitter=True
        )
        
        plt.title(f'OD600 - {cult}')
        plt.xticks(rotation=90)
        plt.xlabel('Strains')
        plt.ylabel('OD600 Values')
        plt.legend(title='Date', loc='lower right')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/Analysis/figures/OD/{cult}_OD600_plot.png', dpi=600)
        plt.show()
        plt.close()

plot_mean_od_by_cult(df_merged)
# %%
