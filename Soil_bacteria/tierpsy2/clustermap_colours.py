#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 7 17:20:54 2025

Make clustermaps to explore behaviour of N2 on different soil bacteria

@author: jbergqvist
"""
#%%
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
import colorsys

from scipy.cluster.hierarchy import linkage, dendrogram

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

SEQ_DATA = Path('/Users/jb3623/Desktop/soil_bacteria/MOYb1_strains_plate_map_BCoding.csv')
# WINDOW_FILES = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Results/window_summaries')

# ANALYSIS_DIR = Path('/Users/bonnie/OneDrive - Imperial College London/Bode_compounds/Analysis')

# RAW_DATA_DIR = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Results')

# feats_plot = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Analysis/Scripts')

#figures_dir = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/Correct/Figures/tierpsy2')
figures_dir = Path('/Users/jb3623/Desktop/soil_bacteria/clustermaps/tierpsy256/excluding_240710')

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

    
    # Read in the sequencing data
    seq_df = pd.read_csv(SEQ_DATA)
    # Change column name from 'B_code' to 'bacteria_strain'
    seq_df.rename(columns={'B_code': 'bacteria_strain'}, inplace=True)

    # Create mappings from 'bacteria_strain' to 'genus', 'phylum', 'class', and 'family', 'isolation_type', 'Ce_strain', 'CeMBIO
    genus_mapping = seq_df.set_index('bacteria_strain')['genus'].to_dict()
    phylum_mapping = seq_df.set_index('bacteria_strain')['phylum'].to_dict()
    class_mapping = seq_df.set_index('bacteria_strain')['class'].to_dict()
    family_mapping = seq_df.set_index('bacteria_strain')['family'].to_dict()
    isolation_type_mapping = seq_df.set_index('bacteria_strain')['isolation_type'].to_dict()
    Ce_strain_mapping = seq_df.set_index('bacteria_strain')['Ce_strain'].to_dict()
    CEMBIO_mapping = seq_df.set_index('bacteria_strain')['CeMBIO'].to_dict()

    # Map the values from seq_df to meta based on 'bacteria_strain'
    meta['genus'] = meta['bacteria_strain'].map(genus_mapping)
    meta['phylum'] = meta['bacteria_strain'].map(phylum_mapping)
    meta['class'] = meta['bacteria_strain'].map(class_mapping)
    meta['family'] = meta['bacteria_strain'].map(family_mapping)
    meta['isolation_type'] = meta['bacteria_strain'].map(isolation_type_mapping)
    meta['Ce_strain'] = meta['bacteria_strain'].map(Ce_strain_mapping)
    meta['CeMBIO'] = meta['bacteria_strain'].map(CEMBIO_mapping)

    # Add 'Escherichia' to 'genus' column for rows with 'OP50' in 'bacteria_strain'
    meta.loc[meta['bacteria_strain'].str.contains('OP50'), 'genus'] = 'Escherichia'
    meta.loc[meta['bacteria_strain'].str.contains('OP50'), 'family'] = 'Enterobacteriaceae'
    meta.loc[meta['bacteria_strain'].str.contains('OP50'), 'class'] = 'Gammaproteobacteria'
    meta.loc[meta['bacteria_strain'].str.contains('OP50'), 'phylum'] = 'Pseudomonadota'
    meta.loc[meta['bacteria_strain'].str.contains('OP50'), 'CeMBIO'] = 'FALSE'
    # Add 'Sphingomonas' to 'genus' column for rows with 'JUb134' in 'bacteria_strain'
    meta.loc[meta['bacteria_strain'].str.contains('JUb134'), 'genus'] = 'Sphingomonas'
    meta.loc[meta['bacteria_strain'].str.contains('JUb134'), 'family'] = 'Sphingomonadaceae'
    meta.loc[meta['bacteria_strain'].str.contains('JUb134'), 'class'] = 'Alphaproteobacteria'
    meta.loc[meta['bacteria_strain'].str.contains('JUb134'), 'phylum'] = 'Pseudomonadota'
    meta.loc[meta['bacteria_strain'].str.contains('JUb134'), 'CeMBIO'] = 'TRUE'
			

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
    
#    strain_list = list(meta['bacteria_strain'].unique())
#    cmap = list(sns.color_palette(cc.glasbey, n_colors=len(strain_list)))
#    STRAIN_cmap = (dict(zip(strain_list,cmap)))
#    
#    colours = (dict(zip(meta.worm_gene,meta.bacteria_strain)))
    
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
# Defining the colour palette for the row_colours in a taxonomically hierarchal way

## *** NEED TO KEEP WORKING ON THIS. THE COLOURS ASSIGNED ARE NOT DIFFERENT ENOUGH FOR THE THOSE UNDER 'PSEUDOMONADOTA' ***

import pandas as pd
import matplotlib.colors as mcolors
import colorsys


# Function to generate distinguishable colors within a parent color
def generate_color_variations(base_color, num_variations, hue_variation=0.15):
    """Generates color variations by adjusting the hue and saturation slightly."""
    base_rgb = mcolors.to_rgb(base_color)
    variations = []
    for i in range(num_variations):
        # Convert to HSV
        h, s, v = colorsys.rgb_to_hsv(*base_rgb)
        
        # Slight hue shift per sibling and slight brightness adjustment
        hue_shift = (i / max(1, num_variations)) * hue_variation
        brightness_shift = 0.9 if i % 2 == 0 else 1.1  # Alternate slightly darker/lighter
        
        # Adjust hue and brightness
        new_h = (h + hue_shift) % 1.0
        new_s = min(1.0, s * brightness_shift)
        
        # Convert back to RGB
        new_rgb = colorsys.hsv_to_rgb(new_h, new_s, v)
        variations.append(mcolors.rgb2hex(new_rgb))
    return variations

# Step 1: Assign base colors to each phylum
phylum_list = meta["phylum"].unique()
num_phyla = len(phylum_list)
base_colors = mcolors.TABLEAU_COLORS  # Using Tableau colors for distinctiveness
base_color_mapping = {phylum_list[i]: list(base_colors.values())[i % len(base_colors)] for i in range(num_phyla)}

# Step 2: Generate colors for each taxonomic level
meta["phylum_color"] = meta["phylum"].map(base_color_mapping)

# Step 3: Iterate through each phylum to generate distinguishable child colors
for phylum in meta["phylum"].unique():
    # Filter the rows for the current phylum
    phylum_meta = meta[meta["phylum"] == phylum]
    
    # Check if all strains belong to the same class and family
    unique_classes = phylum_meta["class"].unique()
    unique_families = phylum_meta["family"].unique()
    
    if len(unique_classes) == 1 and len(unique_families) == 1:
        # Assign the same color as the phylum color to class and family and genus
        meta.loc[meta["phylum"] == phylum, "class_color"] = base_color_mapping[phylum]
        meta.loc[meta["phylum"] == phylum, "family_color"] = base_color_mapping[phylum]
        meta.loc[meta["phylum"] == phylum, "genus_color"] = base_color_mapping[phylum]
    else:
        # Determine hue variation based on phylum
        if phylum == 'Bacillota':
            hue_variation = 0.3
        elif phylum == 'Pseudomonadota':
            hue_variation = 0.12
        else:
            hue_variation = 0.15
        
        # Generate unique colors for classes under this phylum
        classes = phylum_meta["class"].unique()
        class_colors = generate_color_variations(base_color_mapping[phylum], len(classes), hue_variation)
        class_color_mapping = {classes[i]: class_colors[i] for i in range(len(classes))}
        meta.loc[meta["phylum"] == phylum, "class_color"] = meta["class"].map(class_color_mapping)

        # Repeat this process for family and genus levels
        for class_ in phylum_meta["class"].unique():
            class_meta = phylum_meta[phylum_meta["class"] == class_]
            
            # Generate family colors
            families = class_meta["family"].unique()
            family_colors = generate_color_variations(class_color_mapping[class_], len(families), hue_variation)
            family_color_mapping = {families[i]: family_colors[i] for i in range(len(families))}
            meta.loc[(meta["phylum"] == phylum) & (meta["class"] == class_), "family_color"] = meta["family"].map(family_color_mapping)

            # Generate genus colors
            for family in class_meta["family"].unique():
                family_meta = class_meta[class_meta["family"] == family]
                genera = family_meta["genus"].unique()
                
                if len(genera) == 1:
                    # Assign the same color as the family color to genus
                    meta.loc[(meta["phylum"] == phylum) & (meta["class"] == class_) & (meta["family"] == family), "genus_color"] = family_color_mapping[family]
                else:
                    genus_colors = generate_color_variations(family_color_mapping[family], len(genera), hue_variation)
                    genus_color_mapping = {genera[i]: genus_colors[i] for i in range(len(genera))}
                    meta.loc[(meta["phylum"] == phylum) & (meta["class"] == class_) & (meta["family"] == family), "genus_color"] = meta["genus"].map(genus_color_mapping)


# Step 4: Display the resulting dataframe with assigned colors
print(meta)

# Step 5: Assign 'gray' to all taxonomic levels if all are NaN
def assign_default_color_for_missing_data(df, levels, default_color="gray"):
    """Assigns a default color to all taxonomic levels if all levels are NaN."""
    # Identify rows where all taxonomic levels are NaN
    missing_rows = df[levels].isna().all(axis=1)
    
    # Assign 'gray' color to each level's color column for missing rows
    for level in levels:
        color_column = f"{level}_color"
        df.loc[missing_rows, color_column] = default_color

# Taxonomic levels to check for missing data
taxonomic_levels = ["phylum", "class", "family", "genus"]

# Assign default color for missing rows
assign_default_color_for_missing_data(meta, taxonomic_levels)

# Colour map of each phylum group

from IPython.core.display import display, HTML

def display_hierarchical_color_map_by_phylum(df, taxonomic_levels, color_columns):
    """Display hierarchical color maps grouped by phylum, including phylum color and showing unique classes, families, and genera."""
    
    # Get unique phyla
    unique_phyla = df['phylum'].dropna().unique()
    
    html_output = "<h2>Hierarchical Color Map Grouped by Phylum</h2>"
    
    for phylum in unique_phyla:
        phylum_color = df.loc[df['phylum'] == phylum, 'phylum_color'].iloc[0]  # Get the color for the phylum
        
        html_output += f"<h3>Phylum: {phylum}</h3>"
        html_output += f"""
        <div style="display: inline-block; background-color: {phylum_color}; color: #000; padding: 5px 10px; border-radius: 5px; margin-bottom: 10px;">
            {phylum}
        </div><br>
        """
        
        html_output += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        
        # Add headers for taxonomic levels
        html_output += "<tr>"
        for level in taxonomic_levels:
            html_output += f"<th>{level.capitalize()}</th>"
        html_output += "</tr>"
        
        # Filter dataframe for the current phylum
        phylum_df = df[df['phylum'] == phylum]
        
        # Collect unique combinations of all taxonomic levels
        unique_entries = phylum_df[taxonomic_levels + color_columns].drop_duplicates()
        
        # Add rows for each unique entry
        for _, row in unique_entries.iterrows():
            html_output += "<tr>"
            for level, color_column in zip(taxonomic_levels, color_columns):
                color = row[color_column]
                name = row[level] if pd.notna(row[level]) else "N/A"
                html_output += f"""
                <td style="background-color: {color}; padding: 5px; text-align: center; border: 1px solid #ccc;">
                    {name}
                </td>
                """
            html_output += "</tr>"
        
        html_output += "</table><br>"

    display(HTML(html_output))

# Taxonomic levels and corresponding color columns
taxonomic_levels = ["phylum", "class", "family", "genus"]
color_columns = [f"{level}_color" for level in taxonomic_levels]

# Display the hierarchical color map grouped by phylum
display_hierarchical_color_map_by_phylum(meta, taxonomic_levels, color_columns)

# %%

# Colour map of each phylum group

from IPython.core.display import display, HTML

def display_hierarchical_color_map_by_phylum(df, taxonomic_levels, color_columns):
    """Display hierarchical color maps grouped by phylum, including phylum color and showing unique classes, families, and genera."""
    
    # Get unique phyla
    unique_phyla = df['phylum'].dropna().unique()
    
    html_output = "<h2>Hierarchical Color Map Grouped by Phylum</h2>"
    
    for phylum in unique_phyla:
        phylum_color = df.loc[df['phylum'] == phylum, 'phylum_color'].iloc[0]  # Get the color for the phylum
        
        html_output += f"<h3>Phylum: {phylum}</h3>"
        html_output += f"""
        <div style="display: inline-block; background-color: {phylum_color}; color: #000; padding: 5px 10px; border-radius: 5px; margin-bottom: 10px;">
            {phylum}
        </div><br>
        """
        
        html_output += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        
        # Add headers for taxonomic levels
        html_output += "<tr>"
        for level in taxonomic_levels:
            html_output += f"<th>{level.capitalize()}</th>"
        html_output += "</tr>"
        
        # Filter dataframe for the current phylum
        phylum_df = df[df['phylum'] == phylum]
        
        # Collect unique combinations of all taxonomic levels
        unique_entries = phylum_df[taxonomic_levels + color_columns].drop_duplicates()
        
        # Add rows for each unique entry
        for _, row in unique_entries.iterrows():
            html_output += "<tr>"
            for level, color_column in zip(taxonomic_levels, color_columns):
                color = row[color_column]
                name = row[level] if pd.notna(row[level]) else "N/A"
                html_output += f"""
                <td style="background-color: {color}; padding: 5px; text-align: center; border: 1px solid #ccc;">
                    {name}
                </td>
                """
            html_output += "</tr>"
        
        html_output += "</table><br>"

    display(HTML(html_output))

# Taxonomic levels and corresponding color columns
taxonomic_levels = ["phylum", "class", "family", "genus"]
color_columns = [f"{level}_color" for level in taxonomic_levels]

# Display the hierarchical color map grouped by phylum
display_hierarchical_color_map_by_phylum(meta, taxonomic_levels, color_columns)

# %%

# Colour map of each taxonomic level

from IPython.core.display import display, HTML

def display_color_map(df, level, color_column):
    """Display a color map for a given taxonomic level with corresponding background colors."""
    unique_entries = df[[level, color_column]].drop_duplicates().sort_values(by=level)
    
    html_output = f"<h3>Color Map for {level.capitalize()}:</h3><ul>"
    for _, row in unique_entries.iterrows():
        name = row[level]
        color = row[color_column]
        html_output += f"""
        <li style="margin-bottom: 5px;">
            <span style="display: inline-block; background-color: {color}; color: #000; padding: 3px 10px; border-radius: 5px;">
                {name}
            </span>
        </li>
        """
    html_output += "</ul>"
    display(HTML(html_output))

# Display color maps for each taxonomic level
display_color_map(meta, "phylum", "phylum_color")
display_color_map(meta, "class", "class_color")
display_color_map(meta, "family", "family_color")
display_color_map(meta, "genus", "genus_color")



# %%

# Colour map of the taxonomy of all 'bacteria_strains'

from IPython.core.display import display, HTML

def display_bacteria_color_map(df, taxonomic_levels, color_columns):
    """Display each taxonomic level for each 'bacteria_strain' with the corresponding background colors."""
    html_output = "<h3>Taxonomic Color Map for Each Bacteria Strain:</h3><table border='1' style='border-collapse: collapse; width: 100%;'>"
    html_output += "<tr><th>Bacteria Strain</th>"
    
    # Add table headers for each taxonomic level
    for level in taxonomic_levels:
        html_output += f"<th>{level.capitalize()}</th>"
    html_output += "</tr>"
    
    # Iterate through each row to display bacteria strain and its taxonomic levels with colors
    for _, row in df.iterrows():
        html_output += f"<tr><td>{row['bacteria_strain']}</td>"
        for level, color_column in zip(taxonomic_levels, color_columns):
            color = row[color_column]
            name = row[level] if pd.notna(row[level]) else "N/A"
            html_output += f"""
            <td style="background-color: {color}; padding: 5px; text-align: center; border: 1px solid #ccc;">
                {name}
            </td>
            """
        html_output += "</tr>"
    
    html_output += "</table>"
    display(HTML(html_output))

# Taxonomic levels and corresponding color columns
taxonomic_levels = ["phylum", "class", "family", "genus"]
color_columns = [f"{level}_color" for level in taxonomic_levels]

# Display the color-coded taxonomy for each bacteria strain
display_bacteria_color_map(meta, taxonomic_levels, color_columns)





# %%
