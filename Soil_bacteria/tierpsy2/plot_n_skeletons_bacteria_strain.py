#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:16:16 2024

@author: John

Plotting the n_skeletons by bacteria_strain and imaging day (date_yyyymmdd)
"""

#%% 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tierpsytools.read_data.hydra_metadata import (read_hydra_metadata, align_bluelight_conditions)
import os
from matplotlib.ticker import ScalarFormatter


#%% Define file locations, save directory 

# ROOT_DIR = Path('/Users/bonnie/Documents/Bode_compounds/Analysis')

FEAT_FILE =  Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Results_NN/features_summary_tierpsy_plate_20241011_160311.csv') 
FNAME_FILE = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Results_NN/filenames_summary_tierpsy_plate_20241011_160311.csv')
METADATA_FILE = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/AuxiliaryFiles/wells_updated_metadata.csv')

output_dir = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/Correct/Figures/tierpsy2')

#%%
feat, meta = read_hydra_metadata(
    FEAT_FILE,
    FNAME_FILE,
    METADATA_FILE)
meta['worm_gene']=meta['bacteria_strain']

print(feat)
print(meta)

#%%
# Check how many wells per bacteria_strain are found in the metadata

# Identify unique dates and remove the first imaging day
#unique_dates = meta['date_yyyymmdd'].unique()
#meta = meta[meta['date_yyyymmdd'] != 20240710]

# Define the order of bacteria_strain explicitly
bacteria_strain_order = ['OP50', 'JUb134'] + [f'B{i}' for i in range(1, 97)]

meta_copy = meta.copy()
meta_copy = meta_copy[meta_copy['date_yyyymmdd'] != 20240710]
# Convert the bacteria_strain column to a categorical type with the specified order
meta_copy['bacteria_strain'] = pd.Categorical(meta_copy['bacteria_strain'], categories=bacteria_strain_order, ordered=True)


plt.figure(figsize=(20, 6))
sns.histplot(x='bacteria_strain', data=meta_copy, bins=meta_copy['bacteria_strain'].nunique(), discrete=True)
plt.title('Number of Wells per Bacteria Strain')
plt.xticks(rotation=90)


#%%
# drops wells that weren't tracked in all 3 conditions
feat, meta = align_bluelight_conditions(feat, meta, how='inner')

#%%

# Define the order of bacteria_strain explicitly
# With JUb134
#bacteria_strain_order = ['OP50', 'JUb134'] + [f'B{i}' for i in range(1, 97)]

## Without JUb134
#bacteria_strain_order = ['OP50'] + [f'B{i}' for i in range(1, 97)]
#
### Identify unique dates
unique_dates = meta['date_yyyymmdd'].unique()
##
### Remove the first imaging day
#meta = meta[meta['date_yyyymmdd'] != 20240710]


#%% Check the number of wells per bacteria_strain that are labeled as 'True' in the 'is_bad_well' column
# Create a new dataframe for all the wells that are labeled as 'True' in the 'is_bad_well' column
bad_wells = meta[meta['is_bad_well'] == True]

# Plot the number of wells per 'bacteria_strain' that are labeled as 'True' in the 'is_bad_well' column
plt.figure(figsize=(10, 6))
sns.histplot(x='bacteria_strain', data=bad_wells, bins=len(bacteria_strain_order), discrete=True)
plt.title('Number of Wells per Bacteria Strain Labeled as Bad Wells')
plt.xticks(rotation=90)
plt.show()

#%%
#Drop bad wells
mask = meta['is_bad_well'] == True 
meta = meta[~mask]    
feat = feat[~mask]


#%% # Check the number of wells per bacteria strain where the minimum n_skeletons threshold was not met in any condition

meta_cop_skel = meta.copy()
skeletons = meta_cop_skel[(meta_cop_skel['n_skeletons_prestim'] <2000) | (meta_cop_skel['n_skeletons_bluelight'] <2000) 
                 | (meta_cop_skel['n_skeletons_poststim'] <2000)]#.index

# Plot the number of wells per 'bacteria_strain' with 'n_skeletons' below 2000 for any condition
plt.figure(figsize=(10, 6))
sns.histplot(x='bacteria_strain', data=skeletons, bins=len(bacteria_strain_order), discrete=True)
plt.title('Number of Wells per Bacteria Strain with n_skeletons < 2000')
plt.xticks(rotation=90)
plt.show()


#%%
# Filter by skeletons
skeletons_drop = meta[(meta['n_skeletons_prestim'] <2000) | (meta['n_skeletons_bluelight'] <2000) 
                 | (meta['n_skeletons_poststim'] <2000)].index
meta.drop(skeletons_drop, inplace = True)
feat.drop(skeletons_drop, inplace = True)



#%% Plot n_skeletons per bacteria_strain for each date
for date in unique_dates:
    # Filter data for the current date
    date_data = meta[meta['date_yyyymmdd'] == date]
    
    plt.figure(figsize=(15, 6))  # Adjust figure size as needed
    box = sns.boxplot(x='bacteria_strain', y='n_skeletons', data=date_data, order=bacteria_strain_order)
    strip = sns.stripplot(x='bacteria_strain', y='n_skeletons', data=date_data, color='black', jitter=True, order=bacteria_strain_order, alpha=0.5)
    
    # Set alpha for boxplot patches
    for patch in box.artists:
        patch.set_alpha(0.5)
    
    # Calculate the mean value of n_skeletons for each bacteria_strain for the current date
    means = date_data.groupby('bacteria_strain')['n_skeletons'].mean().reindex(bacteria_strain_order)
    
    # Plot the means as a line, with zorder set to a higher value to draw it in front
    plt.plot(bacteria_strain_order, means, color='r', marker='o', linestyle='-', label='Mean per strain', zorder=3)
    
    plt.title(f'n_skeletons per bacteria_strain for {date}')
    plt.xticks(rotation=90)  # Rotate labels for better readability
    #plt.gca().set_yscale('log')  # Set the y-axis to a logarithmic scale
    plt.legend()  # Display the legend
    plt.show()



#%%

# exclude the first imaging day
meta = meta[meta['date_yyyymmdd'] != 20240710]
#meta = meta[meta['bacteria_strain'] != 'OP50']
#meta = meta[meta['bacteria_strain'] != 'JUb134']

# create a new column to store the sum of n_skeletons across all conditions
meta['n_skeletons'] = meta['n_skeletons_prestim'] + meta['n_skeletons_bluelight'] + meta['n_skeletons_poststim']

plt.figure(figsize=(20, 6))
box = sns.boxplot(x='bacteria_strain', y='n_skeletons', data=meta, order=bacteria_strain_order)
strip = sns.stripplot(x='bacteria_strain', y='n_skeletons', data=meta, color='black', jitter=True, order=bacteria_strain_order, alpha=0.6, hue='date_yyyymmdd', palette='Set1')

# Calculate mean n_skeletons for each date and plot as horizontal lines
unique_dates = meta['date_yyyymmdd'].unique()
colors = ['red', 'blue', 'green']  # Example colors for the lines
for i, date in enumerate(unique_dates):
    mean_value = meta[meta['date_yyyymmdd'] == date]['n_skeletons'].mean()
    plt.axhline(y=mean_value, color=colors[i], linestyle='--', label=f'Mean for {date}')

plt.title('n_skeletons per bacteria_strain - tierpsy2 (wells not meeting 2000 n_skeletons in any condition removed)')
plt.xticks(rotation=90)
#plt.gca().set_yscale('log')  # Set the y-axis to a logarithmic scale
plt.legend()
plt.show()


#%%

# exclude the first imaging day
meta = meta[meta['date_yyyymmdd'] != 20240710]

# Define the lists of bacteria strains
top_t257_bluelight_cluster = ['B35', 'B28', 'B44']
middle_top_t257_bluelight_cluster = ['JUb134', 'B92','B69', 'B90', 'B45']
middle_bottom_t257_bluelight_cluster = ['B64', 'B21', 'B30']
bottom_t257_bluelight_cluster = ['B88', 'B82', 'B95','B91', 'B87', 'B96']


# Define color palettes for each cluster
op50_color = 'purple'  # Distinct color for 'OP50'
color_palettes = {
    'top': sns.color_palette("Blues", len(top_t257_bluelight_cluster)),
    'middle_top': sns.color_palette("Greens", len(middle_top_t257_bluelight_cluster)),
    'middle_bottom': sns.color_palette("PuRd", len(middle_bottom_t257_bluelight_cluster)),
    'bottom': sns.color_palette("Reds", len(bottom_t257_bluelight_cluster))
}

all_strains = top_t257_bluelight_cluster + middle_top_t257_bluelight_cluster + middle_bottom_t257_bluelight_cluster + bottom_t257_bluelight_cluster + ['OP50']


# Create a combined color palette for all strains
combined_palette = {}
combined_palette.update({strain: color for strain, color in zip(top_t257_bluelight_cluster, color_palettes['top'])})
combined_palette.update({strain: color for strain, color in zip(middle_top_t257_bluelight_cluster, color_palettes['middle_top'])})
combined_palette.update({strain: color for strain, color in zip(middle_bottom_t257_bluelight_cluster, color_palettes['middle_bottom'])})
combined_palette.update({strain: color for strain, color in zip(bottom_t257_bluelight_cluster, color_palettes['bottom'])})
combined_palette['OP50'] = op50_color

# Set the order of the strains
strain_order = all_strains + ['OP50']


# Exclude the first imaging day
#meta = meta[meta['date_yyyymmdd'] != 20240710]

# Filter the meta DataFrame to include only the strains in the clusters
meta_filtered = meta[meta['bacteria_strain'].isin(all_strains + ['OP50'])]



# Iterate over each feature and create a plot

plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

ax = sns.boxplot(y='n_skeletons_bluelight', 
                    x='bacteria_strain', 
                    data=meta_filtered, 
                    #showfliers=False, 
                    palette=combined_palette, 
                    order=strain_order)
plt.title(f"n_skeletons during bluelight per strain")

# Create the stripplot

sns.stripplot(y='n_skeletons_bluelight', 
                x='bacteria_strain', 
                data=meta_filtered,
                color='black', 
                hue='date_yyyymmdd', 
                dodge=False, 
                jitter=True,
                palette='Set1', 
                alpha=0.6, 
                ax=ax, 
                order=all_strains)

# Modify x-tick labels to include value counts
#xtick_labels = [f'{strain}\n(n={value_counts[strain]})' for strain in all_strains]
plt.xticks(ticks=range(len(all_strains)), rotation=0, ha='center', fontsize=16)
plt.yticks(fontsize=14)
plt.ylabel('n_skeletons during bluelight', fontsize=18)

plt.title('Number of Worms per Well Estimation', fontsize=20)
plt.xlabel('')
plt.legend(fontsize=14, loc='upper right')

# Set y-axis to logarithmic scale
ax.set_yscale('log')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the figure
plt.savefig('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/Figures/tierpsy2/n_skeletons/excluding_240710/excluding_min_n_skeletons/with_jub134/n_skeletons_bluelight_clusters.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close()


# %%
