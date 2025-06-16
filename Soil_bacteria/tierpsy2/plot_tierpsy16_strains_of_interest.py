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

#%% Decide if looking at subset of compounds or features

select_compounds = False

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

# Create three different lists with appropriate suffixes
tierpsy_16_bluelight = tierpsy_16
tierpsy_16_prestim = [feature.replace('_bluelight', '_prestim') for feature in tierpsy_16]
tierpsy_16_poststim = [feature.replace('_bluelight', '_poststim') for feature in tierpsy_16]


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

# WINDOW_FILES = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Results/window_summaries')

# ANALYSIS_DIR = Path('/Users/bonnie/OneDrive - Imperial College London/Bode_compounds/Analysis')

# RAW_DATA_DIR = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Results')

# feats_plot = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Analysis/Scripts')

figures_dir = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/Correct/Figures/tierpsy2')

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

    

#%% Plotting tierpsy16 features for strains of interest - One graph per cluster from the tierpsy 257 clustermap for bluelight stim
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

# Combine data
combined_data = pd.concat([feat, meta], axis=1)
combined_data = combined_data.loc[combined_data['date_yyyymmdd'].isin([20240720, 20240816, 20240830])]

# Define the directory to save the figures
#save_dir = "/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/Figures/tierpsy2/tierpsy_16/excluding_240710/strains_of_interest/without_jub134/imaging_day_coloured"
#if not os.path.exists(save_dir):
#    os.makedirs(save_dir)

#%%

# One plot per cluster - dots not coloured by imaging day

# Define the directory to save the figures
save_dir = "/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/Figures/tierpsy2/tierpsy_16/excluding_240710/strains_of_interest/without_jub134/LSR/bluelight"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Iterate over each feature
for feature in tierpsy_16_bluelight: # change to 'tierpsy_16_bluelight', 'tierpsy_16_prestim', tierpsy_16_poststim' accordingly
    # Iterate over each list of bacteria strains
    for cluster_name, cluster in zip(['top', 'middle_top', 'middle_bottom', 'bottom'],
                                     [top_t257_bluelight_cluster, middle_top_t257_bluelight_cluster, middle_bottom_t257_bluelight_cluster, bottom_t257_bluelight_cluster]):
        # Filter data for the strains in the current list
        filtered_data = combined_data[combined_data['bacteria_strain'].isin(cluster + ['OP50'])]

        # Create a color palette with 'OP50' explicitly set to purple
        palette = {strain: color for strain, color in zip(cluster, color_palettes[cluster_name])}
        palette['OP50'] = op50_color

        # Set the order of the strains
        strain_order = cluster + ['OP50']

        # Plot
        plt.figure(figsize=(8, 10))
        plt.style.use(CUSTOM_STYLE)
        sns.set_style('ticks')

        ax = sns.boxplot(y=feature, 
                        x='worm_gene', 
                        data=filtered_data, 
                        hue='bacteria_strain', 
                        showfliers=False, 
                        palette=palette, 
                        order=strain_order)
        plt.title(f"{cluster_name} cluster of strains")

        # Adjust legend and labels
        handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles=handles, labels=labels, title='Bacteria Strain')
        ax.set_ylabel(f'{feature}', fontsize=18)
        ax.set_xlabel('')

        sns.stripplot(y=feature, 
                    x='worm_gene', 
                    data=filtered_data, 
                    hue='bacteria_strain', 
                    dodge=False, 
                    jitter=True,
                    color='black', 
                    alpha=0.5, 
                    ax=ax, 
                    order=strain_order)

        handles, labels = ax.get_legend_handles_labels()
        #l = plt.legend(handles, labels, title='Bacteria Strain')
        ax.set_xlabel('')

        plt.tight_layout()

        # Construct the file path
        file_path = os.path.join(save_dir, f"{cluster_name}_cluster_{feature}.png")

        # Save the plot
        plt.savefig(file_path, dpi=600)
        plt.show()

# %% 
# Plotting tierpsy16 features for strains of interest and coloured by imaging day
# One graph per feature, including all clusters of interest from the tierpsy 257 clustermap for bluelight stim

# Define the directory to save the figures. Change the final subdirectory depending on the stimulus
save_dir = "/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/Figures/tierpsy2/tierpsy_16/excluding_240710/strains_of_interest/with_jub134/LSR/bluelight"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Convert the 'date_yyyymmdd' column to strings without decimals
combined_data['date_yyyymmdd'] = combined_data['date_yyyymmdd'].apply(lambda x: str(int(float(x))))

# Define the colors for each imaging day
imaging_day_palette = {
    '20240720': 'red',
    '20240816': 'blue',
    '20240830': 'green'
}

# Combine all bacteria strains into a single list
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

# Iterate over each feature
for feature in tierpsy_16_bluelight: # change to 'tierpsy_16_prestim', 'tierpsy_16_bluelight', tierpsy_16_poststim' accordingly
    # Filter data for all strains
    filtered_data = combined_data[combined_data['bacteria_strain'].isin(all_strains)]

    # Plot
    plt.figure(figsize=(12, 10))
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')

    ax = sns.violinplot(y=feature, 
                     x='worm_gene', 
                     data=filtered_data, 
                     #hue='date_yyyymmdd', 
                     #showfliers=False, 
                     palette=combined_palette, 
                     order=strain_order)
    plt.title(f"All clusters of strains ")

    # Adjust legend and labels
    handles, labels = ax.get_legend_handles_labels()
    ax.set_ylabel(f'{feature}', fontsize=18)
    ax.set_xlabel('')

    sns.stripplot(y=feature, 
                  x='worm_gene', 
                  data=filtered_data, 
                  hue='date_yyyymmdd', 
                  dodge=False, 
                  jitter=True,
                  palette=imaging_day_palette, 
                  alpha=0.35, 
                  ax=ax, 
                  order=strain_order)

    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('')

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

    plt.tight_layout()

    # Construct the file path
    file_path = os.path.join(save_dir, f"all_clusters_{feature}.png")

    # Save the plot
    plt.savefig(file_path, dpi=300)
    plt.show()

# %%
# Plotting speed_90th features for all three conditions for all strains of interest - One graph per feature, including all clusters of interest from the tierpsy 257 clustermap for bluelight stim

# Convert the 'date_yyyymmdd' column to strings without decimals
combined_data['date_yyyymmdd'] = combined_data['date_yyyymmdd'].apply(lambda x: str(int(float(x))))

# Define the colors for each imaging day
imaging_day_palette = {
    '20240720': 'red',
    '20240816': 'blue',
    '20240830': 'green'
}

# Speed features
speed_features = ['speed_90th_bluelight', 'speed_90th_prestim', 'speed_90th_poststim']

# Combine all bacteria strains into a single list
all_strains = top_t257_bluelight_cluster + middle_top_t257_bluelight_cluster + middle_bottom_t257_bluelight_cluster + bottom_t257_bluelight_cluster + ['OP50']

# Create a combined color palette for all strains
combined_palette = {}
combined_palette.update({strain: color for strain, color in zip(top_t257_bluelight_cluster, color_palettes['top'])})
combined_palette.update({strain: color for strain, color in zip(middle_top_t257_bluelight_cluster, color_palettes['middle_top'])})
combined_palette.update({strain: color for strain, color in zip(middle_bottom_t257_bluelight_cluster, color_palettes['middle_bottom'])})
combined_palette.update({strain: color for strain, color in zip(bottom_t257_bluelight_cluster, color_palettes['bottom'])})
combined_palette['OP50'] = op50_color

# Set the order of the strains
#strain_order = ['OP50'] + top_t257_bluelight_cluster + middle_top_t257_bluelight_cluster + middle_bottom_t257_bluelight_cluster + bottom_t257_bluelight_cluster

strain_order = all_strains + ['OP50']

# Filter data for all strains
filtered_data = combined_data[combined_data['bacteria_strain'].isin(all_strains)]

# Iterate over each feature and create a plot
for feature in speed_features:
    plt.figure(figsize=(12, 10))
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')

    ax = sns.violinplot(y=feature, 
                     x='worm_gene', 
                     data=filtered_data, 
                     #showfliers=False, 
                     palette=combined_palette, 
                     order=strain_order)
    plt.title(f"All clusters of strains - {feature}")

    # Adjust legend and labels
    handles, labels = ax.get_legend_handles_labels()
    ax.set_ylabel(f'{feature}', fontsize=18)
    ax.set_xlabel('')

    sns.stripplot(y=feature, 
                  x='worm_gene', 
                  data=filtered_data, 
                  hue='date_yyyymmdd', 
                  dodge=False, 
                  jitter=True,
                  palette=imaging_day_palette, 
                  alpha=0.35, 
                  ax=ax, 
                  order=strain_order)

    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('')

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

    # Set y-axis limit
    plt.ylim(0, 800)

    plt.tight_layout()

    # Construct the file path
    file_path = os.path.join(save_dir, f"all_clusters_{feature}.png")

    # Save the plot
    plt.savefig(file_path, dpi=300)
    plt.show()

    plt.close()
# %%
