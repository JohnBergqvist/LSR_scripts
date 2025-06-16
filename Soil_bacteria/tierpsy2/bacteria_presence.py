#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:41:47 2025

Script for conducting in-depth stats analysis, calculating timeseries and
plotting window summaries of all positive hit compounds vs N2

@author: John
"""
#%%
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats
import colorcet as cc
import glob 
import csv
from itertools import chain
import plotly.graph_objects as go
from plotly.offline import plot
import os
import math
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from itertools import product
import matplotlib.lines as mlines

from tierpsytools.analysis.significant_features import k_significant_feat
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.read_data.hydra_metadata import (read_hydra_metadata, 
                                                    _does_it_need_6WP_patch,
                                                    align_bluelight_conditions)
from tierpsytools.hydra.platechecker import fix_dtypes
import time
from tierpsytools.drug_screenings.filter_compounds import (
    compounds_with_low_effect_univariate)
from tierpsytools.analysis.statistical_tests import (univariate_tests,
                                                     _multitest_correct)

sys.path.insert(0, '/Users/jb3623/Desktop/Behavgenom_repo/tierpsy-tools-python/tierpsytools/helper_scripts')
from  tierpsytools.helper_scripts.helper import (read_disease_data,
                    select_strains,
                    filter_features,
                    make_colormaps,
                    find_window,
                    strain_gene_dict,
                    BLUELIGHT_WINDOW_DICT,
                    STIMULI_ORDER)
from tierpsytools.helper_scripts.plotting_helper import  (plot_colormap,
                              plot_cmap_text,
                              make_clustermaps,
                              clustered_barcodes,
                              feature_box_plots,
                              average_feature_box_plots,
                              clipped_feature_box_plots,
                              window_errorbar_plots,
                              CUSTOM_STYLE)
from tierpsytools.helper_scripts.ts_helper import (align_bluelight_meta,
                       # load_bluelight_timeseries_from_results,
                        make_feats_abs,
                        plot_strains_ts,
                        get_motion_modes,
                        get_frac_motion_modes_with_ci,
                        plot_frac_all_modes_coloured_by_motion_mode,
                        plot_frac_by_mode,
                        plot_frac_all_modes,
                        MODECOLNAMES,
                        short_plot_frac_by_mode)

from tierpsytools.helper_scripts.luigi_helper import load_bluelight_timeseries_from_results

from tierpsytools.analysis.statistical_tests import univariate_tests

def four_sig_digits(x, pos):
    """Format numbers to four significant digits."""
    return f'{x:.4g}'

# Function to format p-values
def format_p_value(p):
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return "p < 0.01"
    elif p < 0.05:
        return "p < 0.05"
    else:
        return f"p = {p:.3f}"
#%% Decide what analysis to run

plot_syngenta = True

N2_analysis = True
# creates clustermap and feature order for (DMSO and water) controls

do_clustermaps = True
# creates clustermap and feature order for all compounds

ANALYSIS_TYPE = [
                'all_stim'
                  # 'timeseries',
                # 'bluelight'
                 ] 
motion_modes=False

exploratory=True
# plots k sig feats

do_stats=True
# TODO: change file directory for do_stats=False depending if you want t-test vs. control, t-test vs. strain control or permutation p-values


which_stat_test = 't-test' # permutation_ttest' or 'LMM' or 't-test'

feats_to_plot = [
                'all' # plot all significant features
                # 'select' # plot select significant features with stars on clustermaps 
                # 'tierpsy_16'
                ] 

is_reload_timeseries_from_results = False
is_recalculate_frac_motion_modes = False

#%% Load feature sets

# get tierpsy16 features
tierpsy_16 = []
with open(
        '/Users/jb3623/Desktop/Behavgenom_repo/tierpsy-tools-python/tierpsytools/extras/feat_sets/tierpsy_16.csv','r') as fid:
   for l in fid.readlines():
       if 'path' not in l:
         tierpsy_16.append(l.rstrip().strip(',') + '_bluelight')
         
# Replace space in feature
tierpsy_16[0] = tierpsy_16[0].replace('﻿length_90th_bluelight', 'length_90th_bluelight')

#drop feateures in tierpsy_16 that are not in the feature set after filtering
# Features to remove
features_to_remove = ['width_midbody_norm_10th_bluelight', 'width_head_base_norm_10th_bluelight']

# Remove specified features from tierpsy_16
tierpsy_16 = [feature for feature in tierpsy_16 if feature not in features_to_remove]

# get tierpsy256 features
tierpsy_256 = []
with open('/Users/jb3623/Documents/tierpsy256.csv', 'r', encoding='utf-8', errors='ignore') as fid:
    for l in fid.readlines():
        if 'path' not in l:
            tierpsy_256.append(l.rstrip().strip(','))

# Create a new list with the required suffixes
tierpsy_256_extended = []
for feature in tierpsy_256:
    tierpsy_256_extended.append(feature + '_bluelight')
    tierpsy_256_extended.append(feature + '_prestim')
    tierpsy_256_extended.append(feature + '_poststim')
print("Number of features after extending:", len(tierpsy_256_extended))
print("First few extended features:", tierpsy_256_extended[:10])

# Print the number of features read and the first few features for debugging
print("Number of features read:", len(tierpsy_256_extended))
print("First few features:", tierpsy_256_extended[:10])

# Check for any potential encoding issues
## tierpsy_256 = [feature.encode('utf-8').decode('utf-8') for feature in tierpsy_256]

# Print the number of features after encoding fix
print("Number of features after encoding fix:", len(tierpsy_256_extended))

# Fix any specific issues with feature names
tierpsy_256_extended[0] = tierpsy_256_extended[0].replace('﻿motion_mode_paused_frequency_bluelight', 'motion_mode_paused_frequency_bluelight')
tierpsy_256_extended[1] = tierpsy_256_extended[1].replace('﻿motion_mode_paused_frequency_prestim', 'motion_mode_paused_frequency_prestim')
tierpsy_256_extended[2] = tierpsy_256_extended[2].replace('﻿motion_mode_paused_frequency_poststim', 'motion_mode_paused_frequency_poststim')
# List of features not in tierpsy 2.0
feat_not_in_2_0 = [
    'width_tail_base_w_forward_50th', 'width_head_base_w_forward_10th', 'd_area_w_backward_10th', 
    'width_midbody_10th', 'd_width_head_base_IQR', 'width_midbody_w_forward_10th', 
    'd_width_tail_base_IQR', 'width_midbody_90th', 'width_tail_base_w_backward_IQR', 
    'd_width_midbody_w_forward_10th', 'width_midbody_w_backward_50th', 'width_tail_base_w_backward_10th', 
    'width_midbody_w_backward_10th', 'd_width_tail_base_50th', 'width_tail_base_10th', 
    'd_width_head_base_w_backward_50th', 'd_width_midbody_50th', 'width_tail_base_w_backward_50th', 
    'd_width_head_base_w_forward_IQR', 'd_width_head_base_w_forward_50th', 'width_head_base_10th', 
    'width_head_base_w_forward_50th', 'width_tail_base_90th', 'width_midbody_w_forward_50th', 
    'd_width_head_base_50th', 'width_midbody_50th', 'width_midbody_w_backward_90th', 
    'width_tail_base_w_backward_90th', 'width_midbody_w_forward_90th', 'width_tail_base_50th', 
    'width_tail_base_w_forward_10th'
]
# Create a new list with the required suffixes
feat_not_in_2_0_extended = []
for feature in feat_not_in_2_0:
    feat_not_in_2_0_extended.append(feature + '_bluelight')
    feat_not_in_2_0_extended.append(feature + '_prestim')
    feat_not_in_2_0_extended.append(feature + '_poststim')

# Filter out features not in tierpsy 2.0
tierpsy_256_filtered = [feature for feature in tierpsy_256_extended if feature not in feat_not_in_2_0_extended]

# Print the number of features after filtering
print("Number of features after filtering:", len(tierpsy_256_filtered))
print("First few filtered features:", tierpsy_256_filtered[:10])

#%% Define file locations, save directory 

# ROOT_DIR = Path('/Users/bonnie/Documents/Bode_compounds/Analysis')

FEAT_FILE =  Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Results_NN/features_summary_tierpsy_plate_20241011_160311.csv') 
FNAME_FILE = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Results_NN/filenames_summary_tierpsy_plate_20241011_160311.csv')
METADATA_FILE = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/AuxiliaryFiles/wells_updated_metadata.csv')
BAC_PRES_FILE = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/bacteria_presence/bacteria_presence.csv')
# WINDOW_FILES = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Results/window_summaries')

# ANALYSIS_DIR = Path('/Users/bonnie/OneDrive - Imperial College London/Bode_compounds/Analysis')

# RAW_DATA_DIR = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Results')

# feats_plot = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Analysis/Scripts')

figures_dir = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/bacteria_presence')

#%%Setting plotting styles and filtering data
if __name__ == '__main__':
    ['']
    
    # CUSTOM_STYLE= mplt style card ensuring figures are consistent for papers
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    
    # Read in data and align by bluelight with tierpsy tools functions
    # make matching metadata and feature summaries, drops wells that were not tracked
    feat, meta = read_hydra_metadata(
        FEAT_FILE,
        FNAME_FILE,
        METADATA_FILE)

    # Drop wells that were recorded on a specific day
    meta = meta[meta['date_yyyymmdd'] != 20240710]

    # Drop wells that contained JUb134
    meta = meta[meta['bacteria_strain'] != 'JUb134']

    # drops wells that weren't tracked in all 3 conditions
    feat, meta = align_bluelight_conditions(feat, meta, how='inner')
    
    # Add columns to reuse existing function
#    meta['drug_type'] = meta['compound']
#    meta['imaging_plate_drug_concentration'] = meta['imaging_plate_concentration_uM']
#    
#    # Drop empty wells
#    mask = meta['drug_type']=='empty'
#    meta = meta[~mask]
#    feat = feat[~mask]
#    
    #Drop bad wells
    mask = meta['is_bad_well'] == True 
    meta = meta[~mask]    
    feat = feat[~mask]

    # Filter by skeletons
    skeletons = meta[(meta['n_skeletons_prestim'] <2000) | (meta['n_skeletons_bluelight'] <2000) 
                     | (meta['n_skeletons_poststim'] <2000)].index
    meta.drop(skeletons, inplace = True)
    feat.drop(skeletons, inplace = True)

    # Filter nans with tierpsy tools functions                           
    # drop wells that have 50% or more NaN values for features, expect 0-5
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]


    # drop features that have 5% NaN values, expect ~300
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())


    # changes formatting of date so easier to plot etc
    imaging_date_yyyymmdd = pd.DataFrame(meta['date_yyyymmdd'])
    meta['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd      

    print(meta)
    print(feat)


 #%%
 
    meta['worm_gene'] = meta['bacteria_strain']
#    meta.loc[meta['imaging_plate_drug_concentration'].notna(
#        ), 'worm_gene'] =  meta['drug_type'] + '_' + meta['imaging_plate_drug_concentration'].astype(str)
#    meta.replace({'DMSO_0.0':'DMSO'},inplace=True)
    
    # Find number of replicates for each drug dose per day
    grouped = pd.DataFrame(meta.groupby(['worm_gene','date_yyyymmdd']
                                        ).size())
    
    grouped.reset_index(level=1,inplace=True)
    grouped = grouped.pivot(columns='date_yyyymmdd')
    
    # Count the number of NaN values in each row
    num_nan_per_row = grouped.isnull().sum(axis=1)

    # Use boolean indexing to filter out rows with less than 3 replicates
    filtered_grouped = grouped[num_nan_per_row == 0]
    
    meta = meta[meta['worm_gene'].isin(filtered_grouped.index)]
    feat = feat.loc[meta.index]
    
    genes = list(meta['worm_gene'].unique())
    
    # remove control
#    genes.remove('DMSO')
    
    strain_list = list(meta['bacteria_strain'].unique())
    cmap = list(sns.color_palette(cc.glasbey, n_colors=len(strain_list)))
    STRAIN_cmap = (dict(zip(strain_list,cmap)))
    
    colours = (dict(zip(meta.worm_gene,meta.bacteria_strain)))
    
    # # Select subset of features
    # if select_compounds:
    #     genes = compounds
    
    # else:
    #     # create a dictionary of "genes" (drugs) and their solvents 
    #     genes = dict(zip(meta.worm_gene, meta.solvent))   
    #     # remove controls
    #     for key in controls:
    #         del genes[key]

#%%

# Filter the meta file for the bacteria of interest
top_t256_cluster = ['B28', 'B35', 'B44', 'B34', 'B14', 'B57']
middle_top_t256_cluster = ['B45', 'B90', 'B69', 'B92']
middle_bottom_t256_cluster = ['B21', 'B64', 'B30']
bottom_t256_cluster = ['B13', 'B87', 'B91', 'B95', 'B82', 'B88']

# Create a list of the bacteria strains to keep
strains_of_interest = (top_t256_cluster + middle_top_t256_cluster + middle_bottom_t256_cluster + bottom_t256_cluster)
print(f"Strains of interest: {strains_of_interest}")

# Filter the meta file for the bacteria of interest
meta_copy = meta.copy()
print(f"Number of wells before filtering for strains of interest: {meta_copy.shape[0]}")
meta_copy = meta_copy[meta_copy['bacteria_strain'].isin(strains_of_interest)]
print(f"Number of wells after filtering for strains of interest: {meta_copy.shape[0]}")

# Ensure 'bacteria_strain' column is a categorical type with the specified order
meta_copy['bacteria_strain'] = pd.Categorical(meta_copy['bacteria_strain'], categories=strains_of_interest, ordered=True)

# Read in the bacteria_presence file
bac_pres_df = pd.read_csv(BAC_PRES_FILE)


#%%
# Add a column called 'bluelight_imgstore' to the bac_pres_df dataframe that contains the imgstore values with the 'date_yyyymmdd' added to the
# beginning of the imgstore value followed by a '/' character
bac_pres_df['imgstore_name_bluelight'] = bac_pres_df.apply(lambda row: str(row['date_yyyymmdd']) + '/' + row['imgstore'], axis=1)

print(f"Name of the first row in the 'imgstore_name_bluelight' column: {bac_pres_df['imgstore_name_bluelight'].iloc[6]}")

#%%

# Merge OD data with metadata on 'bacteria_strain' and 'imgstore'
meta_copy = pd.merge(meta_copy, bac_pres_df, on=['bacteria_strain', 'imgstore_name_bluelight'], how='inner')
print(f"Number of wells after merging with bacteria_presence: {meta_copy.shape[0]}")

# Drop wells with OP50
meta_copy = meta_copy[meta_copy['bacteria_strain'] != 'OP50']
print(f"Number of wells after filtering out OP50: {meta_copy.shape[0]}")

# Convert the 'keep' column to a string
meta_copy['keep'] = meta_copy['keep'].astype(str).str.strip().str.lower()


# Count the number of rows that have 'false' in the column 'keep' in the 'meta_copy' dataframe
num_false = meta_copy[meta_copy['keep'] == 'false'].shape[0]

# Drop wells that are 'false' in column 'keep' - should not be kept in the analysis
meta_copy = meta_copy[meta_copy['keep'] != 'false']
print(f"Number of wells after filtering out wells with 'keep' = 'false': {meta_copy.shape[0]}")

meta_copy['bacteria_visibly_present'] = meta_copy['bacteria_visibly_present'].astype(str).str.strip().str.lower()
print(f"Number of wells listed as 'true' in the 'bacteria_visibly_present' column: {meta_copy[meta_copy['bacteria_visibly_present'] == 'true'].shape[0]}")

#%%
# Convert the 'bacteria_visibly_present' column to a boolean
meta_copy['bacteria_visibly_present'] = meta_copy['bacteria_visibly_present'].map({'true': True, 'false': False})
# Check that the conversion worked
print(f"Number of wells listed as 'true' in the 'bacteria_visibly_present' column: {meta_copy[meta_copy['bacteria_visibly_present'] == True].shape[0]}")

#%%

# Plot the number of wells listed as true vs false in the 'bacteria_visibly_present' column in the 'meta_copy' dataframe
fig, ax = plt.subplots()
meta_copy['bacteria_visibly_present'].value_counts().plot(kind='bar', ax=ax)
ax.set_title('Number of wells listed as true vs false in the "bacteria_visibly_present" column')
ax.set_xlabel('bacteria_visibly_present')
ax.set_ylabel('Number of wells')
plt.show()


#%%

# print the number of true vs false in the 'bacteria_visibly_present' column per 'bacteria_strain'
print(meta_copy.groupby('bacteria_strain')['bacteria_visibly_present'].value_counts())

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))
meta_copy.groupby('bacteria_strain')['bacteria_visibly_present'].value_counts().unstack().reindex(strains_of_interest).plot(kind='bar', ax=ax)
ax.set_title('')
ax.set_xlabel('')
ax.set_ylabel('Number of wells')
plt.legend(title='Bacteria in Well')
plt.show()

#%%
combined_data = pd.concat([feat, meta_copy], axis=1)

data = combined_data.copy()

data = data.dropna(subset=['bacteria_visibly_present'])
#%% Plot the true vs false data for each strain

for strain in strains_of_interest:
    sns.boxplot(x='bacteria_visibly_present', y='length_90th_poststim', data=data[data['bacteria_strain'] == strain])
    plt.title(f"Strain: {strain}")
    plt.show()


#%%
# Kolmogorov-Smirnov Test
from scipy.stats import ks_2samp

clusters = {
    'Top Cluster': ['B28', 'B35', 'B44', 'B34', 'B14', 'B57'],
    'Middle Top Cluster': ['B45', 'B90', 'B69', 'B92'],
    'Middle Bottom Cluster': ['B21', 'B64', 'B30'],
    'Bottom Cluster': ['B13', 'B87', 'B91', 'B95', 'B82', 'B88']
}

color_palettes = {
    'Top Cluster': sns.color_palette("Blues", 2),
    'Middle Top Cluster': sns.color_palette("Greens", 2),
    'Middle Bottom Cluster': sns.color_palette("PuRd", 2),
    'Bottom Cluster': sns.color_palette("Reds", 2)
}

# Significance threshold
alpha = 0.05

# Loop through each cluster
for cluster_name, cluster_strains in clusters.items():
    significant_features = []
    p_values = {}
    
    # Loop through each feature
    for feature in tierpsy_16:
        # Perform K-S test
        group_true = data[(data['bacteria_strain'].isin(cluster_strains)) & (data['bacteria_visibly_present'] == True)][feature]
        group_false = data[(data['bacteria_strain'].isin(cluster_strains)) & (data['bacteria_visibly_present'] == False)][feature]
        
        if len(group_true) > 0 and len(group_false) > 0:
            ks_stat, p_value = ks_2samp(group_true, group_false)
            
            # Check if the p-value is below the significance threshold
            if p_value < alpha:
                significant_features.append(feature)
                p_values[feature] = p_value
    
    # Plot the significant features
    for feature in significant_features:
        plt.figure(figsize=(10, 6))
        palette = {'True': color_palettes[cluster_name][0], 'False': color_palettes[cluster_name][1]}
        sns.violinplot(x='bacteria_visibly_present', y=feature, data=data[data['bacteria_strain'].isin(cluster_strains)], inner=None, palette=palette)
        sns.stripplot(x='bacteria_visibly_present', y=feature, data=data[data['bacteria_strain'].isin(cluster_strains)], color='k', alpha=0.5)
        plt.xlabel('Bacteria Visibly Present')
        plt.ylabel(feature, fontsize=14)
        plt.yticks(fontsize=12)
        plt.title(f"{cluster_name}: {feature}", size=16)
        plt.text(0.5, 0.95, f"Two-Sample K-S test p-value: {p_values[feature]:.4f}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.show()


#%%
# Mann-Whitney U Test with Bonferroni Correction
from scipy.stats import mannwhitneyu

clusters = {
    'Top Cluster': ['B28', 'B35', 'B44', 'B34', 'B14', 'B57'],
    'Middle Top Cluster': ['B45', 'B90', 'B69', 'B92'],
    'Middle Bottom Cluster': ['B21', 'B64', 'B30'],
    'Bottom Cluster': ['B13', 'B87', 'B91', 'B95', 'B82', 'B88']
}

color_palettes = {
    'Top Cluster': sns.color_palette("Blues", 2),
    'Middle Top Cluster': sns.color_palette("Greens", 2),
    'Middle Bottom Cluster': sns.color_palette("PuRd", 2),
    'Bottom Cluster': sns.color_palette("Reds", 2)
}

# Significance threshold
alpha = 0.05

# Number of features for Bonferroni correction
num_features = len(tierpsy_256_filtered)
bonferroni_alpha = alpha / num_features

# Loop through each cluster
for cluster_name, cluster_strains in clusters.items():
    significant_features = []
    p_values = {}
    
    # Loop through each feature
    for feature in tierpsy_256_filtered:
        # Perform Mann-Whitney U test
        group_true = data[(data['bacteria_strain'].isin(cluster_strains)) & (data['bacteria_visibly_present'] == True)][feature]
        group_false = data[(data['bacteria_strain'].isin(cluster_strains)) & (data['bacteria_visibly_present'] == False)][feature]
        
        if len(group_true) > 0 and len(group_false) > 0:
            u_stat, p_value = mannwhitneyu(group_true, group_false, alternative='two-sided')
            
            # Check if the Bonferroni-corrected p-value is below the significance threshold
            if p_value < bonferroni_alpha:
                significant_features.append(feature)
                p_values[feature] = p_value
    
    # Check if there are no significant features
    if not significant_features:
        print(f"No features were found to be statistically different between bacteria present and absent for {cluster_name}.")
    else:
        # Plot the significant features
        for feature in significant_features:
            plt.figure(figsize=(10, 6))
            palette = {'True': color_palettes[cluster_name][0], 'False': color_palettes[cluster_name][1]}
            sns.violinplot(x='bacteria_visibly_present', y=feature, data=data[data['bacteria_strain'].isin(cluster_strains)], inner=None, palette=palette)
            sns.stripplot(x='bacteria_visibly_present', y=feature, data=data[data['bacteria_strain'].isin(cluster_strains)], color='k', alpha=0.5)
            plt.xlabel('Bacteria Visibly Present')
            plt.ylabel(feature, fontsize=14)
            plt.yticks(fontsize=12)
            plt.title(f"{cluster_name}: {feature}", size=16)
            plt.text(0.5, 0.95, f"Mann-Whitney U test p-value: {p_values[feature]:.4f}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            plt.show()

# %%
# Make an interactive PCA plot for each strain of interest and color by 'bacteria_visibly_present'
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px


for strain in strains_of_interest:
    # Filter the data for the current strain
    strain_data = data[data['bacteria_strain'] == strain]

    # Remove rows with NaNs in the 'angular_velocity_abs_10th_prestim' column
    strain_data = strain_data.dropna(subset=['angular_velocity_abs_10th_prestim'])


    # Select features for PCA (excluding non-numeric columns and the target column)
    features = strain_data[tierpsy_16]

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(scaled_features)

    # Get the explained variance ratios
    explained_variance = pca.explained_variance_ratio_

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    pca_df['bacteria_visibly_present'] = strain_data['bacteria_visibly_present'].values

    # Plot the PCA results in 3D using plotly
    fig = px.scatter_3d(
        pca_df, 
        x='PC1', 
        y='PC2', 
        z='PC3', 
        color='bacteria_visibly_present',
        color_discrete_map={True: 'orange', False: 'blue'},
        title=f'PCA of Bacteria {strain} Colored by Bacteria Visibly Present',
        labels={
            'PC1': f'Principal Component 1 ({explained_variance[0]*100:.2f}% variance)',
            'PC2': f'Principal Component 2 ({explained_variance[1]*100:.2f}% variance)',
            'PC3': f'Principal Component 3 ({explained_variance[2]*100:.2f}% variance)'
        }
    )

    fig.show()
# %%
# Make an interactive PCA plot for each strain of interest and color by 'bacteria_visibly_present'
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px


for cluster_name, strains in clusters.items():
    # Filter the data for the current cluster
    cluster_data = data[data['bacteria_strain'].isin(strains)]

    # Remove rows with NaNs in the 'angular_velocity_abs_10th_prestim' column
    cluster_data = cluster_data.dropna(subset=['angular_velocity_abs_10th_prestim'])

    # Select features for PCA (excluding non-numeric columns and the target column)
    features = cluster_data[tierpsy_16]

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(scaled_features)

    # Get the explained variance ratios
    explained_variance = pca.explained_variance_ratio_

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    pca_df['bacteria_visibly_present'] = cluster_data['bacteria_visibly_present'].values

    # Plot the PCA results in 3D using plotly
    fig = px.scatter_3d(
        pca_df, 
        x='PC1', 
        y='PC2', 
        z='PC3', 
        color='bacteria_visibly_present',
        color_discrete_map={True: 'orange', False: 'blue'},
        title=f'PCA of {cluster_name} Colored by Bacteria Visibly Present',
        labels={
            'PC1': f'Principal Component 1 ({explained_variance[0]*100:.2f}% variance)',
            'PC2': f'Principal Component 2 ({explained_variance[1]*100:.2f}% variance)',
            'PC3': f'Principal Component 3 ({explained_variance[2]*100:.2f}% variance)'
        }
    )

    # Update the marker size
    fig.update_traces(marker=dict(size=3)) 

    # Update the layout to make the axes titles smaller
    fig.update_layout(
        scene=dict(
            xaxis_title_font=dict(size=10),  # Adjust the size value as needed
            yaxis_title_font=dict(size=10),  # Adjust the size value as needed
            zaxis_title_font=dict(size=10)   # Adjust the size value as needed
        ),
            width=1000,  # Adjust the width value as needed
            height=800   # Adjust the height value as needed
    )

    fig.show()
# %%
# Make an interactive PCA plot for all strains of interest and color by cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px

clusters = {
    'Top Cluster': ['B28', 'B35', 'B44', 'B34', 'B14', 'B57'],
    'Middle Top Cluster': ['B45', 'B90', 'B69', 'B92'],
    'Middle Bottom Cluster': ['B21', 'B64', 'B30'],
    'Bottom Cluster': ['B13', 'B87', 'B91', 'B95', 'B82', 'B88']
}

# Combine all strains into a single list
all_strains = [strain for strains in clusters.values() for strain in strains]

# Filter the data for all strains
combined_data = data[data['bacteria_strain'].isin(all_strains)]

# Remove rows with NaNs in the 'angular_velocity_abs_10th_prestim' column
combined_data = combined_data.dropna(subset=['angular_velocity_abs_10th_prestim'])

# Select features for PCA (excluding non-numeric columns and the target column)
features = combined_data[tierpsy_16]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(scaled_features)

# Get the explained variance ratios
explained_variance = pca.explained_variance_ratio_

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
pca_df['cluster'] = combined_data['bacteria_strain'].map(
    {strain: cluster_name for cluster_name, strains in clusters.items() for strain in strains}
)

# Plot the PCA results in 3D using plotly
fig = px.scatter_3d(
    pca_df, 
    x='PC1', 
    y='PC2', 
    z='PC3', 
    color='cluster',
    title='PCA of All Clusters Colored by Cluster',
    labels={
        'PC1': f'Principal Component 1 ({explained_variance[0]*100:.2f}% variance)',
        'PC2': f'Principal Component 2 ({explained_variance[1]*100:.2f}% variance)',
        'PC3': f'Principal Component 3 ({explained_variance[2]*100:.2f}% variance)'
    }
)

# Update the marker size
fig.update_traces(marker=dict(size=3)) 

# Update the layout to make the axes titles smaller
fig.update_layout(
    scene=dict(
        xaxis_title_font=dict(size=10),  # Adjust the size value as needed
        yaxis_title_font=dict(size=10),  # Adjust the size value as needed
        zaxis_title_font=dict(size=10)   # Adjust the size value as needed
    ),
    width=1000,  # Adjust the width value as needed
    height=800   # Adjust the height value as needed
)

fig.show()
# %%
