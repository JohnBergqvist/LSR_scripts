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

SEQ_DATA = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/from_odonnell/MOYb1_strains_plate_map_BCoding.csv')
# WINDOW_FILES = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Results/window_summaries')

# ANALYSIS_DIR = Path('/Users/bonnie/OneDrive - Imperial College London/Bode_compounds/Analysis')

# RAW_DATA_DIR = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Results')

# feats_plot = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Analysis/Scripts')

#figures_dir = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/Figures/tierpsy2/Clustermaps/excluding_240710/tierpsy_256')
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
    #meta = meta[meta['date_yyyymmdd'] != 20240720]

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


    '''If I am not using row_colours, these are the colours I would use'''
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
# Defining the colour palette for the row_colours in a taxonomically hierarchal way to plot 
        # the 'phylum', 'class', 'family' and 'genus' of the bacteria strains


import pandas as pd
import matplotlib.colors as mcolors
import colorsys


# Function to generate distinguishable colors within a parent color
def generate_color_variations(base_color, num_variations, hue_variation=0.15, alpha_variation=0.1):
    """Generates color variations by adjusting the hue, brightness, and transparency slightly."""
    base_rgb = mcolors.to_rgb(base_color)
    variations = []
    for i in range(num_variations):
        # Convert to HSV
        h, s, v = colorsys.rgb_to_hsv(*base_rgb)
        
        # Calculate equal hue shift per sibling
        hue_shift = (i / num_variations) * hue_variation
        
        # Alternate slightly darker/lighter brightness
        brightness_shift = 0.8 if i % 2 == 0 else 1.2
        
        # Adjust hue and brightness
        new_h = (h + hue_shift) % 1.0
        new_s = min(1.0, s * brightness_shift)
        
        # Convert back to RGB
        new_rgb = colorsys.hsv_to_rgb(new_h, new_s, v)
        
        # Calculate equal alpha shift per sibling
        alpha = 1.0 - (i / num_variations) * alpha_variation
        
        # Convert to RGBA hex
        new_rgba = mcolors.to_rgba(new_rgb, alpha=alpha)
        hex_color = mcolors.to_hex(new_rgba, keep_alpha=True)
        variations.append(hex_color)
        print(f"Generated color: {hex_color} with alpha: {alpha}")  # Debug print
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
    
    # Check if there is only 1 unique class in the phylum
    unique_classes = phylum_meta["class"].unique()
    print(f"Unique classes in phylum {phylum}: {unique_classes}")  # Debug statement
    class_color_mapping = {}  # Initialize class_color_mapping
    if len(unique_classes) == 1:
        # Assign the same color as the phylum color to the class
        class_color_mapping[unique_classes[0]] = base_color_mapping[phylum]
        meta.loc[meta["phylum"] == phylum, "class_color"] = base_color_mapping[phylum]
        unique_families = phylum_meta["family"].unique()
        family_color_mapping = {}  # Initialize family_color_mapping
        # Check if there is only 1 unique family in the phylum
        if len(unique_families) == 1:
            # Assign the same color as the class color to the family
            family_color_mapping[unique_families[0]] = base_color_mapping[phylum]
            meta.loc[meta["phylum"] == phylum, "family_color"] = base_color_mapping[phylum]
            unique_genuses = phylum_meta["genus"].unique()
            genus_color_mapping = {}
            # Check if there is only 1 unique genus in the phylum
            if len(unique_genuses) == 1:
                # Assign the same color as the family color to the genus
                genus_color_mapping[unique_genuses[0]] = base_color_mapping[phylum]
                meta.loc[meta["phylum"] == phylum, "genus_color"] = base_color_mapping[phylum]
        else:
            # Generate color variations for families where there is only 1 class, but >1 families in the phylum
            family_colors = generate_color_variations(base_color_mapping[phylum], len(unique_families), hue_variation=0.3, alpha_variation=0.3)
            family_color_mapping = {unique_families[i]: family_colors[i] for i in range(len(unique_families))}
            meta.loc[meta["phylum"] == phylum, "family_color"] = meta["family"].map(family_color_mapping)

            # Debug print to check family color mapping
            print(f"Family color mapping for phylum {phylum}: {family_color_mapping}")

            # Iterate through each family within the class to make a color for each genus
            for family, color in family_color_mapping.items():
                meta.loc[(meta["class"] == class_) & (meta["family"] == family), "family_color"] = color
                # Debug print to check family color assignment
                print(f"Assigned family color for {family}: {color}")

                # Generate color variations for genuses within each family
                unique_genuses = phylum_meta[phylum_meta["family"] == family]["genus"].unique()
                genus_color_mapping = {}
                if len(unique_genuses) == 1:
                    # Assign the same color as the family color to the genus
                    genus_color_mapping[unique_genuses[0]] = color
                    meta.loc[(meta["family"] == family) & (meta["genus"] == unique_genuses[0]), "genus_color"] = color
                    # Debug print to check genus color assignment
                    print(f"Assigned genus color for {unique_genuses[0]}: {color}")
                else:
                    genus_colors = generate_color_variations(color, len(unique_genuses), hue_variation=0.3, alpha_variation=0.3)
                    genus_color_mapping = {unique_genuses[i]: genus_colors[i] for i in range(len(unique_genuses))}
                    for genus, genus_color in genus_color_mapping.items():
                        meta.loc[(meta["family"] == family) & (meta["genus"] == genus), "genus_color"] = genus_color
                        # Debug print to check genus color assignment
                        print(f"Assigned genus color for {genus}: {genus_color}")

            # Debug print to check final meta DataFrame
            print("Final meta DataFrame for class and phylum:")
            print(meta[(meta["class"] == class_) & (meta["phylum"] == phylum)][["family", "genus", "family_color", "genus_color"]])

    else:
        # Generate color variations for classes where there are >1 classes in the phylum
        class_colors = generate_color_variations(base_color_mapping[phylum], len(unique_classes), hue_variation=0.1, alpha_variation=0.3)
        class_color_mapping = {unique_classes[i]: class_colors[i] for i in range(len(unique_classes))}
        meta.loc[meta["phylum"] == phylum, "class_color"] = meta["class"].map(class_color_mapping)
    
    # Iterate through each class within the phylum to make a color for each family
        for class_ in unique_classes:
            print(f"Processing class: {class_}")  # Debug statement
            class_meta = phylum_meta[phylum_meta["class"] == class_]

            # Check if there is only 1 unique family in the class
            unique_families = class_meta["family"].unique()
            print(f"Unique families in class {class_}: {unique_families}")   # Debug statement
            family_color_mapping = {}
            if len(unique_families) == 1:
                family_color_mapping[unique_families[0]] = class_color_mapping[class_]
                # Assign the same color as the class color to the family where there are >1 class in the phylum, but only 1 family in the class
                meta.loc[(meta["phylum"] == phylum) & (meta["class"] == class_), "family_color"] = class_color_mapping[class_]
            else:
                ## Initialize family_color_mapping for each class
                #family_color_mapping = {}
#
                ## Adjust hue_variation and alpha_variation for 'Bacilli' class
                #if class_ == 'Bacilli':
                #    family_color_mapping = {
                #        'Bacillaceae': '#4bce4b',  # Military green color
                #        'Paenibacillaceae': '#217821'  # Another green color
                #    }
                #    print("Assigned colors to families in Bacilli:", meta.loc[meta["class"] == "Bacilli", "family_color"].unique())
                
                # Generate color variations for families where there are >1 class in the phylum and >1 families in the class
                family_colors = generate_color_variations(class_color_mapping[class_], len(unique_families), hue_variation=0.2, alpha_variation=0.3)
                family_color_mapping = {unique_families[i]: family_colors[i] for i in range(len(unique_families))}
                for family, color in family_color_mapping.items():
                    meta.loc[(meta["phylum"] == phylum) & (meta["class"] == class_) & (meta["family"] == family), "family_color"] = color

            # Debug print to check family color mapping
            print(f"Family color mapping for class {class_}: {family_color_mapping}")

            # Iterate through each family within the class to make a color for each genus
            for family in unique_families:
                print(f"Processing family: {family} in class: {class_}")  # Debug statement
                family_meta = class_meta[class_meta["family"] == family]

                # Check if there is only 1 unique genus in the family
                unique_genuses = family_meta["genus"].unique()
                print(f"Unique genuses in family {family}: {unique_genuses}")  # Debug statement
                if len(unique_genuses) == 1:
                    # Assign the same color as the family color to the genus where 
                    if family in family_color_mapping:
                        meta.loc[(meta["phylum"] == phylum) & (meta["class"] == class_) & (meta["family"] == family), "genus_color"] = family_color_mapping[family]
                        genus_color_mapping = {unique_genuses[0]: family_color_mapping[family]}  # Store genus color
                    else:
                        print(f"Warning: Family {family} not found in family_color_mapping")
                else:
                    # Generate color variations for genuses
                    if family in family_color_mapping:
                        genus_colors = generate_color_variations(family_color_mapping[family], len(unique_genuses), hue_variation=0.2, alpha_variation=0.3)
                        genus_color_mapping = {unique_genuses[i]: genus_colors[i] for i in range(len(unique_genuses))}
                        meta.loc[(meta["phylum"] == phylum) & (meta["class"] == class_) & (meta["family"] == family), "genus_color"] = meta["genus"].map(genus_color_mapping)
                    else:
                        print(f"Warning: Family {family} not found in family_color_mapping")

phylum_colors = meta['phylum_color']
class_colors = meta['class_color']
family_colors = meta['family_color']
genus_colors = meta['genus_color']

# Step 5: Assign gray (#808080FF) to all taxonomic levels if all are NaN
def assign_default_color_for_missing_data(df, levels, default_color="#808080FF"):
    """Assigns a default color to all taxonomic levels if all levels are NaN."""
    # Identify rows where all taxonomic levels are NaN
    missing_rows = df[levels].isna().all(axis=1)
    
    # Assign gray (#808080FF) color to each level's color column for missing rows
    for level in levels:
        color_column = f"{level}_color"
        df.loc[missing_rows, color_column] = default_color

# Taxonomic levels to check for missing data
taxonomic_levels = ["phylum", "class", "family", "genus"]

# Assign default color for missing rows
assign_default_color_for_missing_data(meta, taxonomic_levels)


# Colour map of each phylum group

from IPython.core.display import display, HTML
import pandas as pd

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

'''
tierpsy_256 clustermap with row_color taxonomically hierarchal colouring according to the 
'phylum', 'class', 'family' and 'genus' of the bacteria
'''

# Filter feat_df to only include features in tierpsy_256
feat = feat[tierpsy_256_filtered]

# Set save path for figures
#saveto = figures_dir / 'Clustermaps' / 'excluding_240710' / 'tierpsy_256' / 'excluding_jub134' / 'seq_info'
#saveto = figures_dir / 'excluding_240720' / 'row_colours' / 'taxonomy'
figures_dir = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/Figures/tierpsy2/Clustermaps/excluding_240710/tierpsy_256/row_colours/genus_family_class_phylum/LSR')
saveto = figures_dir
#saveto.mkdir(exist_ok=True)

# Removes nan's, bad wells, bad days and selected tierpsy features
feat_df, meta_df, featsets = filter_features(feat, meta)

# Exclude a specific date from meta_df
meta_df = meta_df[meta_df['date_yyyymmdd'] != 20240710]
#meta_df = meta_df[meta_df['date_yyyymmdd'] != 20240720]


# Exclude JUb134
#meta_df = meta_df[meta_df['bacteria_strain'] != 'JUb134']


# Make a stimuli colour map/ look up table with sns
stim_cmap = sns.color_palette('Pastel1', 3)
stim_lut = dict(zip(STIMULI_ORDER.keys(), stim_cmap))

# Save colour maps as legends/figure keys for use in paper
plot_colormap(stim_lut)
plt.savefig(saveto / 'stim_cmap.png')

plot_cmap_text(stim_lut)
plt.savefig(saveto / 'stim_cmap_text.png')

feat_lut = {f: v for f in tierpsy_256_filtered for k, v in stim_lut.items() if k in f}

# Impute nans from feature dataframe
feat_nonan = impute_nan_inf(feat_df)

# Calculate Z score of features
featZ = pd.DataFrame(data=stats.zscore(feat_nonan[tierpsy_256_filtered], axis=0),
                    columns=tierpsy_256_filtered,
                    index=feat_nonan.index)

# Assert no nans
assert featZ.isna().sum().sum() == 0

featZ_grouped = pd.concat([featZ, meta_df], axis=1).groupby(['bacteria_strain']).mean()

# Make clustermap
plt.style.use(CUSTOM_STYLE)

N2clustered_features = {}
sns.set(font_scale=1.2)

# Filter meta to include only the unique bacteria_strain values present in featZ_grouped
unique_strains = featZ_grouped.index.unique()
filtered_meta = meta_df[meta_df['bacteria_strain'].isin(unique_strains)].drop_duplicates(subset='bacteria_strain')

# Set the index of filtered_meta to bacteria_strain
filtered_meta = filtered_meta.set_index('bacteria_strain').loc[unique_strains]


# Extract the color series
phylum_colors = filtered_meta['phylum_color']
class_colors = filtered_meta['class_color']
family_colors = filtered_meta['family_color']
genus_colors = filtered_meta['genus_color']

# Create a DataFrame for row colors
row_colors = pd.DataFrame({
    'Phylum': phylum_colors,
    'Class': class_colors,
    'Family': family_colors,
    'Genus': genus_colors
})

# Ensure the DataFrame is correctly indexed
row_colors.index = featZ_grouped.index

# Print row_colors DataFrame for debugging
print("row_colors DataFrame:")
print(row_colors)

# Function to convert hex color codes with alpha to RGBA format
def hex_to_rgba(hex_color):
    if len(hex_color) == 9:  # If the hex color has an alpha component
        rgba = mcolors.to_rgba(hex_color)
        return rgba
    else:
        return mcolors.to_rgba(hex_color + 'FF')  # Add full opacity if no alpha component

# Apply the function to the row_colors DataFrame
row_colors = row_colors.applymap(hex_to_rgba)

# Print row_colors DataFrame for debugging
print("row_colors DataFrame after converting to RGBA:")
print(row_colors)

def update_tick_labels(cg, strain_to_level, cmap_dict):
    for tick_label in cg.ax_heatmap.axes.get_yticklabels():
        tick_label.set_color('black')  # Set the color to black for all tick labels
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=2)
    cg.ax_heatmap.tick_params(axis='y', width=0.5)


for stim, fset in featsets.items():
    col_colors = featZ_grouped[fset].columns.map(feat_lut)
    # Map the bacteria_strain to their corresponding taxonomy colors
    #row_colors = featZ_grouped.index.map(strain_to_genus).map(GENUS_cmap) # Change this to 'genus', 'class', 'family' or 'phylum' depending on what to colour the bacteria_strain by
    row_colors = row_colors.applymap(lambda x: x if pd.notnull(x) else 'white')

    print("row_colors DataFrame:")
    print(row_colors)

    cg = sns.clustermap(featZ_grouped[fset],
                        col_colors=col_colors,
                        row_colors=row_colors,
                        vmin=-2,
                        vmax=2,
                        yticklabels=1,
                        cbar_pos=None,
                        figsize=(10,20))

    # remove feature labels from x axis
    cg.ax_heatmap.axes.set_xticklabels([])

    # customise y axis
    cg.ax_heatmap.axes.set_ylabel('')

    # Update tick labels with colors and reduce fontsize
    for tick_label in cg.ax_heatmap.axes.get_yticklabels():
        tick_label.set_fontsize(8)  # Set the fontsize

    cg.savefig(saveto / '{}_clustermap.png'.format(stim), dpi=1000)

    # get order of features
    N2clustered_features[stim] = np.array(fset)[cg.dendrogram_col.reordered_ind]

    # Write order of clustered features into .txt file
    for k, v in N2clustered_features.items():
        with open(saveto / 'clustered_features_{}.txt'.format(k), 'w+') as fid:
            for line in v:
                fid.write(line + '\n')



# Function to save color legend
def save_colour_legend(cmap, title, filename):
    # Create a figure for the legend
    fig, ax = plt.subplots(figsize=(6, 4))
    patches = [mpatches.Patch(color=color, label=label) for label, color in cmap.items()]
    ax.legend(handles=patches, loc='center', fontsize='small', title=title)
    ax.axis('off')  # Hide the axes
    plt.savefig(saveto / filename, bbox_inches='tight')
    plt.close(fig)

# Extract unique colors and their corresponding labels from meta_df
genus_cmap = meta_df[['genus', 'genus_color']].dropna().drop_duplicates().set_index('genus')['genus_color'].to_dict()
family_cmap = meta_df[['family', 'family_color']].dropna().drop_duplicates().set_index('family')['family_color'].to_dict()
class_cmap = meta_df[['class', 'class_color']].dropna().drop_duplicates().set_index('class')['class_color'].to_dict()
phylum_cmap = meta_df[['phylum', 'phylum_color']].dropna().drop_duplicates().set_index('phylum')['phylum_color'].to_dict()

# Save colour maps as legends/figure keys for use in paper
save_colour_legend(genus_cmap, 'Genus Colors', 'genus_colour_legend.png')
save_colour_legend(family_cmap, 'Family Colors', 'family_colour_legend.png')
save_colour_legend(class_cmap, 'Class Colors', 'class_colour_legend.png')
save_colour_legend(phylum_cmap, 'Phylum Colors', 'phylum_colour_legend.png')

# Check if the colorbar axis exists before setting the label
if cg.ax_cbar is not None:
    cg.ax_cbar.set_ylabel('Z score', rotation=90)
else:
    print("Warning: Colorbar axis (cg.ax_cbar) is None.")


# Plot colorbar only
if cg.ax_heatmap is not None:
    cg.ax_heatmap.axes.remove()
if cg.ax_row_dendrogram is not None:
    cg.ax_row_dendrogram.axes.remove()
if cg.ax_col_dendrogram is not None:
    cg.ax_col_dendrogram.axes.remove()
if cg.ax_col_colors is not None:
    cg.ax_col_colors.axes.remove()

cg.savefig(saveto / 'taxonomy_colourmap.png', dpi=1000)

# save colour bar for figures
col_colors = featZ_grouped[featsets['all']].columns.map(feat_lut)
    
# plt.figure(figsize=[7.5,5])
cg = sns.clustermap(featZ_grouped[fset],
                col_colors=col_colors,
                vmin=-2,
                vmax=2,
                yticklabels=1)

cg.ax_cbar.set_ylabel('Z score', rotation=90)

if cg.ax_heatmap is not None:
    cg.ax_heatmap.axes.remove()
if cg.ax_row_dendrogram is not None:
    cg.ax_row_dendrogram.axes.remove()
if cg.ax_col_dendrogram is not None:
    cg.ax_col_dendrogram.axes.remove()
if cg.ax_col_colors is not None:
    cg.ax_col_colors.axes.remove()

cg.savefig(saveto / 'colourmap_zscore.png', dpi=1000)

plt.close('all')


plt.close('all')



# %%

'''
tierpsy_256 clustermap split up into top, mid-top, mid-bottom, and bottom clusters 
with row_color taxonomically hierarchal colouring according to the 
'phylum', 'class', 'family' and 'genus' of the bacteria
'''

tierpsy_256_filtered_clusters = tierpsy_256_filtered.copy()



# Filter feat_df to only include features in tierpsy_256
feat = feat[tierpsy_256_filtered_clusters]

# Set save path for figures
#saveto = figures_dir / 'Clustermaps' / 'excluding_240710' / 'tierpsy_256' / 'excluding_jub134' / 'seq_info'
#saveto = figures_dir / 'excluding_240720' / 'row_colours' / 'taxonomy'
figures_dir = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/Figures/tierpsy2/Clustermaps/excluding_240710/tierpsy_256/row_colours/genus_family_class_phylum/LSR/all_clusters')
saveto = figures_dir
#saveto.mkdir(exist_ok=True)

# Removes nan's, bad wells, bad days and selected tierpsy features
feat_df, meta_df, featsets = filter_features(feat, meta)

# Exclude a specific date from meta_df
meta_df = meta_df[meta_df['date_yyyymmdd'] != 20240710]
#meta_df = meta_df[meta_df['date_yyyymmdd'] != 20240720]

# Define the clusters
top_cluster = [
    'B28',
    'B35',
    'B44']
mid_top_cluster = [
    'B45',
    'B90',
    'B69',
    'B92',
    'JUb134'
]
mid_bottom_cluster = [
    'B21',
    'B64',
    'B30'
]
bottom_cluster = [
    'B87',
    'B91',
    'B95',
    'B82',
    'B88'
]
op50 = [
    'OP50',
    'B96'
]

# Combine all cluster lists into a single list
all_clusters = top_cluster + mid_top_cluster + mid_bottom_cluster + bottom_cluster + op50

# Filter the meta_df to only include rows where 'bacteria_strain' is in the combined cluster list
meta_df = meta_df = meta_df[meta_df['bacteria_strain'].isin(all_clusters)]

# Exclude JUb134
#meta_df = meta_df[meta_df['bacteria_strain'] != 'JUb134']


# Make a stimuli colour map/ look up table with sns
stim_cmap = sns.color_palette('Pastel1', 3)
stim_lut = dict(zip(STIMULI_ORDER.keys(), stim_cmap))

# Save colour maps as legends/figure keys for use in paper
plot_colormap(stim_lut)
plt.savefig(saveto / 'stim_cmap.png')

plot_cmap_text(stim_lut)
plt.savefig(saveto / 'stim_cmap_text.png')

feat_lut = {f: v for f in tierpsy_256_filtered_clusters for k, v in stim_lut.items() if k in f}

# Impute nans from feature dataframe
feat_nonan = impute_nan_inf(feat_df)

# Calculate Z score of features
featZ = pd.DataFrame(data=stats.zscore(feat_nonan[tierpsy_256_filtered_clusters], axis=0),
                    columns=tierpsy_256_filtered_clusters,
                    index=feat_nonan.index)

# Assert no nans
assert featZ.isna().sum().sum() == 0

featZ_grouped = pd.concat([featZ, meta_df], axis=1).groupby(['bacteria_strain']).mean()

# Make clustermap
plt.style.use(CUSTOM_STYLE)

N2clustered_features = {}
sns.set(font_scale=1.2)

# Filter meta to include only the unique bacteria_strain values present in featZ_grouped
unique_strains = featZ_grouped.index.unique()
filtered_meta = meta_df[meta_df['bacteria_strain'].isin(unique_strains)].drop_duplicates(subset='bacteria_strain')

# Set the index of filtered_meta to bacteria_strain
filtered_meta = filtered_meta.set_index('bacteria_strain').loc[unique_strains]


# Extract the color series
phylum_colors = filtered_meta['phylum_color']
class_colors = filtered_meta['class_color']
family_colors = filtered_meta['family_color']
genus_colors = filtered_meta['genus_color']

# Create a DataFrame for row colors
row_colors = pd.DataFrame({
    'Phylum': phylum_colors,
    'Class': class_colors,
    'Family': family_colors,
    'Genus': genus_colors
})

# Ensure the DataFrame is correctly indexed
row_colors.index = featZ_grouped.index

# Print row_colors DataFrame for debugging
print("row_colors DataFrame:")
print(row_colors)

# Function to convert hex color codes with alpha to RGBA format
def hex_to_rgba(hex_color):
    if len(hex_color) == 9:  # If the hex color has an alpha component
        rgba = mcolors.to_rgba(hex_color)
        return rgba
    else:
        return mcolors.to_rgba(hex_color + 'FF')  # Add full opacity if no alpha component

# Apply the function to the row_colors DataFrame
row_colors = row_colors.applymap(hex_to_rgba)

# Print row_colors DataFrame for debugging
print("row_colors DataFrame after converting to RGBA:")
print(row_colors)

def update_tick_labels(cg, strain_to_level, cmap_dict):
    for tick_label in cg.ax_heatmap.axes.get_yticklabels():
        tick_label.set_color('black')  # Set the color to black for all tick labels
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=2)
    cg.ax_heatmap.tick_params(axis='y', width=0.5)

# Reverse the order of the featZ_grouped DataFrame
#featZ_grouped_reversed = featZ_grouped.iloc[::-1]

# Reverse the order of the row_colors DataFrame to match the new order
#row_colors_reversed = row_colors.iloc[::-1]

for stim, fset in featsets.items():
    col_colors = featZ_grouped[fset].columns.map(feat_lut)
    # Map the bacteria_strain to their corresponding taxonomy colors
    #row_colors = featZ_grouped.index.map(strain_to_genus).map(GENUS_cmap) # Change this to 'genus', 'class', 'family' or 'phylum' depending on what to colour the bacteria_strain by
    row_colors = row_colors.applymap(lambda x: x if pd.notnull(x) else 'white')

    print("row_colors DataFrame:")
    print(row_colors)

    cg = sns.clustermap(featZ_grouped[fset],
                        col_colors=col_colors,
                        row_colors=row_colors,
                        vmin=-2,
                        vmax=2,
                        yticklabels=1,
                        cbar_pos=None,
                        figsize=(10,20))

    # remove feature labels from x axis
    cg.ax_heatmap.axes.set_xticklabels([])

    # customise y axis
    cg.ax_heatmap.axes.set_ylabel('')

    # Update tick labels with colors and reduce fontsize
    for tick_label in cg.ax_heatmap.axes.get_yticklabels():
        tick_label.set_fontsize(12)  # Set the fontsize
        tick_label.set_rotation(0)

    cg.savefig(saveto / '{}_clustermap.svg'.format(stim), dpi=1000)

    # get order of features
    N2clustered_features[stim] = np.array(fset)[cg.dendrogram_col.reordered_ind]

    # Write order of clustered features into .txt file
    for k, v in N2clustered_features.items():
        with open(saveto / 'clustered_features_{}.txt'.format(k), 'w+') as fid:
            for line in v:
                fid.write(line + '\n')



# Function to save color legend
def save_colour_legend(cmap, title, filename):
    # Create a figure for the legend
    fig, ax = plt.subplots(figsize=(6, 4))
    patches = [mpatches.Patch(color=color, label=label) for label, color in cmap.items()]
    ax.legend(handles=patches, loc='center', fontsize='small', title=title)
    ax.axis('off')  # Hide the axes
    plt.savefig(saveto / filename, bbox_inches='tight', dpi=1000)
    plt.close(fig)

# Extract unique colors and their corresponding labels from meta_df
genus_cmap = meta_df[['genus', 'genus_color']].dropna().drop_duplicates().set_index('genus')['genus_color'].to_dict()
family_cmap = meta_df[['family', 'family_color']].dropna().drop_duplicates().set_index('family')['family_color'].to_dict()
class_cmap = meta_df[['class', 'class_color']].dropna().drop_duplicates().set_index('class')['class_color'].to_dict()
phylum_cmap = meta_df[['phylum', 'phylum_color']].dropna().drop_duplicates().set_index('phylum')['phylum_color'].to_dict()

# Save colour maps as legends/figure keys for use in paper
save_colour_legend(genus_cmap, 'Genus Colors', 'genus_colour_legend.png')
save_colour_legend(family_cmap, 'Family Colors', 'family_colour_legend.png')
save_colour_legend(class_cmap, 'Class Colors', 'class_colour_legend.png')
save_colour_legend(phylum_cmap, 'Phylum Colors', 'phylum_colour_legend.png')

# Check if the colorbar axis exists before setting the label
if cg.ax_cbar is not None:
    cg.ax_cbar.set_ylabel('Z score', rotation=90)
else:
    print("Warning: Colorbar axis (cg.ax_cbar) is None.")


# Plot colorbar only
if cg.ax_heatmap is not None:
    cg.ax_heatmap.axes.remove()
if cg.ax_row_dendrogram is not None:
    cg.ax_row_dendrogram.axes.remove()
if cg.ax_col_dendrogram is not None:
    cg.ax_col_dendrogram.axes.remove()
if cg.ax_col_colors is not None:
    cg.ax_col_colors.axes.remove()

cg.savefig(saveto / 'taxonomy_colourmap.png', dpi=1000)

# save colour bar for figures
col_colors = featZ_grouped[featsets['all']].columns.map(feat_lut)
    
# plt.figure(figsize=[7.5,5])
cg = sns.clustermap(featZ_grouped[fset],
                col_colors=col_colors,
                vmin=-2,
                vmax=2,
                yticklabels=1)

cg.ax_cbar.set_ylabel('Z score', rotation=90)

if cg.ax_heatmap is not None:
    cg.ax_heatmap.axes.remove()
if cg.ax_row_dendrogram is not None:
    cg.ax_row_dendrogram.axes.remove()
if cg.ax_col_dendrogram is not None:
    cg.ax_col_dendrogram.axes.remove()
if cg.ax_col_colors is not None:
    cg.ax_col_colors.axes.remove()

cg.savefig(saveto / 'colourmap_zscore.png', dpi=1000)

plt.close('all')





#%%
## tierpsy_256 clustermap with colouring according to the
        #  'isolation_type', 'Ce_strain', and if the strain belongs to the 'CeMBIO' library of the bacteria



# Filter feat_df to only include features in tierpsy_256
feat = feat[tierpsy_256_filtered]

# Set save path for figures
#saveto = figures_dir / 'Clustermaps' / 'excluding_240710' / 'tierpsy_256' / 'excluding_jub134' / 'seq_info'
saveto = figures_dir / 'row_colours' / 'cembio_CeStrain_isolation_type'
saveto.mkdir(exist_ok=True)

# Removes nan's, bad wells, bad days and selected tierpsy features
feat_df, meta_df, featsets = filter_features(feat, meta)

# Exclude a specific date from meta_df
meta_df = meta_df[meta_df['date_yyyymmdd'] != 20240710]
meta_df = meta_df[meta_df['date_yyyymmdd'] != 20240720]

# Exclude JUb134
#meta_df = meta_df[meta_df['bacteria_strain'] != 'JUb134']

# Ensure CeMBIO column is interpreted as boolean
meta_df['CeMBIO'] = meta_df['CeMBIO'].astype(bool)


# Make a stimuli colour map/ look up table with sns
stim_cmap = sns.color_palette('Pastel1', 3)
stim_lut = dict(zip(STIMULI_ORDER.keys(), stim_cmap))

# Save colour maps as legends/figure keys for use in paper
plot_colormap(stim_lut)
plt.savefig(saveto / 'stim_cmap.png')

plot_cmap_text(stim_lut)
plt.savefig(saveto / 'stim_cmap_text.png')

feat_lut = {f: v for f in tierpsy_256_filtered for k, v in stim_lut.items() if k in f}

# Impute nans from feature dataframe
feat_nonan = impute_nan_inf(feat_df)

# Calculate Z score of features
featZ = pd.DataFrame(data=stats.zscore(feat_nonan[tierpsy_256_filtered], axis=0),
                    columns=tierpsy_256_filtered,
                    index=feat_nonan.index)

# Assert no nans
assert featZ.isna().sum().sum() == 0

featZ_grouped = pd.concat([featZ, meta_df], axis=1).groupby(['bacteria_strain']).mean()

# Make clustermap
plt.style.use(CUSTOM_STYLE)

N2clustered_features = {}
sns.set(font_scale=1.2)

# Filter meta to include only the unique bacteria_strain values present in featZ_grouped
unique_strains = featZ_grouped.index.unique()
filtered_meta = meta_df[meta_df['bacteria_strain'].isin(unique_strains)].drop_duplicates(subset='bacteria_strain')

# Set the index of filtered_meta to bacteria_strain
filtered_meta = filtered_meta.set_index('bacteria_strain').loc[unique_strains]

# Generate color maps for 'isolation_type', 'Ce_strain', and 'CeMBIO'
isolation_type_colors = sns.color_palette('Set1', len(filtered_meta['isolation_type'].dropna().unique()))
isolation_type_lut = dict(zip(filtered_meta['isolation_type'].dropna().unique(), isolation_type_colors))

Ce_strain_colors = sns.color_palette('Set2', len(filtered_meta['Ce_strain'].dropna().unique()))
Ce_strain_lut = dict(zip(filtered_meta['Ce_strain'].dropna().unique(), Ce_strain_colors))

CeMBIO_colors = sns.color_palette('Set3', len(filtered_meta['CeMBIO'].dropna().unique()))
CeMBIO_lut = dict(zip(filtered_meta['CeMBIO'].dropna().unique(), CeMBIO_colors))

# Map the colors to the corresponding rows
filtered_meta['isolation_type_color'] = filtered_meta['isolation_type'].map(lambda x: isolation_type_lut.get(x, 'grey') if pd.isna(x) else isolation_type_lut[x])
filtered_meta['Ce_strain_color'] = filtered_meta['Ce_strain'].map(lambda x: Ce_strain_lut.get(x, 'grey') if pd.isna(x) else Ce_strain_lut[x])
filtered_meta['CeMBIO_color'] = filtered_meta['CeMBIO'].map(lambda x: CeMBIO_lut.get(x, 'grey') if pd.isna(x) else CeMBIO_lut[x])

# Combine the color columns into a DataFrame for row_colors
row_colors = filtered_meta[['isolation_type_color', 'Ce_strain_color', 'CeMBIO_color']].rename(
    columns={
        'isolation_type_color': 'isolation_type',
        'Ce_strain_color': 'Ce_strain',
        'CeMBIO_color': 'CeMBIO'
    }
)

# Ensure the DataFrame is correctly indexed
row_colors.index = featZ_grouped.index

# Print row_colors DataFrame for debugging
print("row_colors DataFrame:")
print(row_colors)


def update_tick_labels(cg, strain_to_level, cmap_dict):
    for tick_label in cg.ax_heatmap.axes.get_yticklabels():
        tick_label.set_color('black')  # Set the color to black for all tick labels
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=2)
    cg.ax_heatmap.tick_params(axis='y', width=0.5)

for stim, fset in featsets.items():
    col_colors = featZ_grouped[fset].columns.map(feat_lut)
    # Map the bacteria_strain to their corresponding taxonomy colors
    #row_colors = featZ_grouped.index.map(strain_to_genus).map(GENUS_cmap) # Change this to 'genus', 'class', 'family' or 'phylum' depending on what to colour the bacteria_strain by
    row_colors = row_colors.applymap(lambda x: x if pd.notnull(x) else 'white')

    print("row_colors DataFrame:")
    print(row_colors)

    cg = sns.clustermap(featZ_grouped[fset],
                        col_colors=col_colors,
                        row_colors=row_colors,
                        vmin=-2,
                        vmax=2,
                        yticklabels=1,
                        cbar_pos=None,
                        figsize=(10,20))

    # remove feature labels from x axis
    cg.ax_heatmap.axes.set_xticklabels([])

    # customise y axis
    cg.ax_heatmap.axes.set_ylabel('')

    # Update tick labels with colors and reduce fontsize
    for tick_label in cg.ax_heatmap.axes.get_yticklabels():
        tick_label.set_fontsize(4)  # Set the fontsize

    cg.savefig(saveto / '{}_clustermap.png'.format(stim), dpi=1000)

    # get order of features
    N2clustered_features[stim] = np.array(fset)[cg.dendrogram_col.reordered_ind]

    # Write order of clustered features into .txt file
    for k, v in N2clustered_features.items():
        with open(saveto / 'clustered_features_{}.txt'.format(k), 'w+') as fid:
            for line in v:
                fid.write(line + '\n')

# Function to save color legend
def save_colour_legend(cmap, title, filename, data_column):
    # Create a copy of the cmap to avoid modifying the original
    cmap_copy = cmap.copy()
    
    # Check if there are NaN values in the data column
    if data_column.isna().any():
        cmap_copy[np.nan] = 'grey'  # Add grey color for NaN

    # Create a figure for the legend
    fig, ax = plt.subplots(figsize=(6, 4))
    patches = [mpatches.Patch(color=color, label='NaN' if pd.isna(label) else label) for label, color in cmap_copy.items()]
    ax.legend(handles=patches, loc='center', fontsize='small', title=title)
    ax.axis('off')  # Hide the axes
    plt.savefig(saveto / filename, bbox_inches='tight')
    plt.close(fig)

# Save color legends for 'isolation_type', 'Ce_strain', and 'CeMBIO'
save_colour_legend(isolation_type_lut, 'Isolation Type', 'isolation_type_legend.png', filtered_meta['isolation_type'])
save_colour_legend(Ce_strain_lut, 'Ce Strain', 'Ce_strain_legend.png', filtered_meta['Ce_strain'])
save_colour_legend(CeMBIO_lut, 'CeMBIO', 'CeMBIO_legend.png', filtered_meta['CeMBIO'])

# Check if the colorbar axis exists before setting the label
if cg.ax_cbar is not None:
    cg.ax_cbar.set_ylabel('Z score', rotation=90)
else:
    print("Warning: Colorbar axis (cg.ax_cbar) is None.")

# Plot colorbar only
if cg.ax_heatmap is not None:
    cg.ax_heatmap.axes.remove()
if cg.ax_row_dendrogram is not None:
    cg.ax_row_dendrogram.axes.remove()
if cg.ax_col_dendrogram is not None:
    cg.ax_col_dendrogram.axes.remove()
if cg.ax_col_colors is not None:
    cg.ax_col_colors.axes.remove()

cg.savefig(saveto / 'colourmap.png', dpi=1000)

plt.close('all')



#%% Make clustermaps of all 

    if do_clustermaps:
             
        # Set save path for figures
        saveto = figures_dir / 'Clustermaps' / 'excluding_240710'
        saveto.mkdir(exist_ok=True)
        
        # Removes nan's, bad wells, bad days and selected tierpsy features
        feat_df, meta_df, featsets = filter_features(feat,
                                                  meta)
        
        # Exclude a specific date from meta_df
        meta_df = meta_df[meta_df['date_yyyymmdd'] != 20240710]
        

        # Make a stimuli colour map/ look up table with sns
        stim_cmap = sns.color_palette('Pastel1',3)
        stim_lut = dict(zip(STIMULI_ORDER.keys(), stim_cmap))
        
        # Save colour maps as legends/figure keys for use in paper
        plot_colormap(stim_lut)
        plt.savefig(saveto / 'stim_cmap.png')
        
        plot_cmap_text(stim_lut)
        plt.savefig(saveto / 'stim_cmap_text.png')
        
        feat_lut = {f:v for f in featsets['all'] for k,v in stim_lut.items() if k in f}
        
        # TODO: Too big for 1 plot
        # # Save colour maps as legends/figure keys for use in paper
        # plot_colormap(STRAIN_cmap)
        # plt.tight_layout()
        # plt.savefig(saveto / 'strain_cmap.png', bbox_inches='tight')
        
        # plot_cmap_text(STRAIN_cmap)
        # plt.savefig(saveto / 'strain_cmap_text.png', bbox_inches='tight')
        
        # Impute nans from feature dataframe
        feat_nonan = impute_nan_inf(feat_df)
        
        # Calculate Z score of features
        featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], 
                                            axis=0),
                          columns=featsets['all'],
                          index=feat_nonan.index)
        # Assert no nans
        assert featZ.isna().sum().sum() == 0
        
        featZ_grouped = pd.concat([featZ,meta_df],axis=1).groupby(['worm_gene']).mean() 
        
        # Make clustermap
        plt.style.use(CUSTOM_STYLE)
        
        N2clustered_features = {}
        sns.set(font_scale=1.2)
        
        for stim, fset in featsets.items():
            col_colors = featZ_grouped[fset].columns.map(feat_lut)
            
            # plt.figure(figsize=[7.5,5])
            cg = sns.clustermap(featZ_grouped[fset],
                            col_colors=col_colors,
                            vmin=-2,
                            vmax=2,
                            yticklabels=1,
                            cbar_pos=None,
                            # metric ='cosine'
                            )
            
            # remove feature labels from x axis
            cg.ax_heatmap.axes.set_xticklabels([])
            
            # customise y axis
            cg.ax_heatmap.axes.set_ylabel('')
            
            for tick_label in cg.ax_heatmap.axes.get_yticklabels():
                # # get drug dose
                drug = tick_label.get_text()
                # colour labels accordsing to drug type
                colour = colours[drug]
                tick_label.set_color(STRAIN_cmap[colour])
            cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize = 3)
            cg.ax_heatmap.tick_params(axis='y',width = 0.5)
        
            cg.savefig(saveto / '{}_clustermap.png'.format(stim), dpi=1000)
            
            # get order of features
            N2clustered_features[stim] = np.array(featsets[stim])[cg.dendrogram_col.reordered_ind] 
        
            # Write order of clustered features into .txt file
            for k, v in N2clustered_features.items():
                with open(saveto / 'clustered_features_{}.txt'.format(k), 'w+') as fid:
                    for line in v:
                        fid.write(line + '\n')
                        
        # TODO: SAVE AS DICTIONARY FOR LATER USE
    
        # save colour bar for figures
        col_colors = featZ_grouped[featsets['all']].columns.map(feat_lut)
            
        # plt.figure(figsize=[7.5,5])
        cg = sns.clustermap(featZ_grouped[fset],
                        col_colors=col_colors,
                        vmin=-2,
                        vmax=2,
                        yticklabels=1)
        
        cg.ax_cbar.set_ylabel('Z score', rotation=90)
        
        # plot colorbar only
        cg.ax_heatmap.axes.remove()
        cg.ax_row_dendrogram.axes.remove()
        cg.ax_col_dendrogram.axes.remove()
        cg.ax_col_colors.axes.remove()
        
        cg.savefig(saveto / 'colourmap.png', dpi=1000)
        
        plt.close('all')
    
    # If not plotting control heatmaps, read cluster features file and make dict
    # for plotting heatmaps etc later on in script  
    
    # TODO: use previous clustermap order? or create new one
    
    else:
        clustered_features = {}
        for fset in STIMULI_ORDER.keys():
            clustered_features[fset] = []
            with open(figures_dir / 
                      'Clustermaps/clustered_features_{}.txt'.format(fset), 'r') as fid:
                clustered_features[fset] = [l.rstrip() 
                                              for l in fid.readlines()]
    
        with open(figures_dir / 'Clustermaps/clustered_features_{}.txt'.format('all'), 'r') as fid:
            clustered_features['all'] = [l.rstrip() for l in fid.readlines()]   

#%% tierpsy_256 clustermap

# Filter feat_df to only include features in tierpsy_256
feat = feat[tierpsy_256_filtered]

# Set save path for figures
saveto = figures_dir / 'Clustermaps' / 'excluding_240710' / 'tierpsy_256' / 'excluding_jub134'
saveto.mkdir(exist_ok=True)

# Removes nan's, bad wells, bad days and selected tierpsy features
feat_df, meta_df, featsets = filter_features(feat, meta)

# Exclude a specific date from meta_df
meta_df = meta_df[meta_df['date_yyyymmdd'] != 20240710]

# Exclude JUb134
meta_df = meta_df[meta_df['bacteria_strain'] != 'JUb134']

# Make a stimuli colour map/ look up table with sns
stim_cmap = sns.color_palette('Pastel1', 3)
stim_lut = dict(zip(STIMULI_ORDER.keys(), stim_cmap))

# Save colour maps as legends/figure keys for use in paper
plot_colormap(stim_lut)
plt.savefig(saveto / 'stim_cmap.png')

plot_cmap_text(stim_lut)
plt.savefig(saveto / 'stim_cmap_text.png')

feat_lut = {f: v for f in tierpsy_256_filtered for k, v in stim_lut.items() if k in f}

# Impute nans from feature dataframe
feat_nonan = impute_nan_inf(feat_df)

# Calculate Z score of features
featZ = pd.DataFrame(data=stats.zscore(feat_nonan[tierpsy_256_filtered], axis=0),
                    columns=tierpsy_256_filtered,
                    index=feat_nonan.index)

# Assert no nans
assert featZ.isna().sum().sum() == 0

featZ_grouped = pd.concat([featZ, meta_df], axis=1).groupby(['worm_gene']).mean()

# Make clustermap
plt.style.use(CUSTOM_STYLE)

N2clustered_features = {}
sns.set(font_scale=1.2)

for stim, fset in featsets.items():
    col_colors = featZ_grouped[fset].columns.map(feat_lut)

    cg = sns.clustermap(featZ_grouped[fset],
                        col_colors=col_colors,
                        vmin=-2,
                        vmax=2,
                        yticklabels=1,
                        cbar_pos=None,
                        figsize=(10,20))

    # remove feature labels from x axis
    cg.ax_heatmap.axes.set_xticklabels([])

    # customise y axis
    cg.ax_heatmap.axes.set_ylabel('')

    for tick_label in cg.ax_heatmap.axes.get_yticklabels():
        drug = tick_label.get_text()
        colour = colours[drug]
        tick_label.set_color(STRAIN_cmap[colour])
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=3)
    cg.ax_heatmap.tick_params(axis='y', width=0.5)

    cg.savefig(saveto / '{}_clustermap.png'.format(stim), dpi=1000)

    # get order of features
    N2clustered_features[stim] = np.array(fset)[cg.dendrogram_col.reordered_ind]

    # Write order of clustered features into .txt file
    for k, v in N2clustered_features.items():
        with open(saveto / 'clustered_features_{}.txt'.format(k), 'w+') as fid:
            for line in v:
                fid.write(line + '\n')

# save colour bar for figures
col_colors = featZ_grouped[tierpsy_256_filtered].columns.map(feat_lut)

cg = sns.clustermap(featZ_grouped[tierpsy_256_filtered],
                    col_colors=col_colors,
                    vmin=-2,
                    vmax=2,
                    yticklabels=1)

cg.ax_cbar.set_ylabel('Z score', rotation=90)

# plot colorbar only
cg.ax_heatmap.axes.remove()
cg.ax_row_dendrogram.axes.remove()
cg.ax_col_dendrogram.axes.remove()
cg.ax_col_colors.axes.remove()

cg.savefig(saveto / 'colourmap.png', dpi=1000)

plt.close('all')


#%% 
# Plotting a 256 clustermap where the values are normalised to that of OP50
# Filter feat_df to only include features in tierpsy_256
feat = feat[tierpsy_256_filtered]

# Set save path for figures
saveto = figures_dir / 'excluding_240720' / 'op50_normalised'
saveto.mkdir(exist_ok=True)

# Removes nan's, bad wells, bad days and selected tierpsy features
feat_df, meta_df, featsets = filter_features(feat, meta)

# Exclude a specific date from meta_df
meta_df = meta_df[meta_df['date_yyyymmdd'] != 20240710]
meta_df = meta_df[meta_df['date_yyyymmdd'] != 20240720]

# Exclude JUb134
#meta_df = meta_df[meta_df['bacteria_strain'] != 'JUb134']

# Make a stimuli colour map/ look up table with sns
stim_cmap = sns.color_palette('Pastel1', 3)
stim_lut = dict(zip(STIMULI_ORDER.keys(), stim_cmap))

# Save colour maps as legends/figure keys for use in paper
plot_colormap(stim_lut)
plt.savefig(saveto / 'stim_cmap.png')

plot_cmap_text(stim_lut)
plt.savefig(saveto / 'stim_cmap_text.png')

feat_lut = {f: v for f in tierpsy_256_filtered for k, v in stim_lut.items() if k in f}

# Impute nans from feature dataframe
feat_nonan = impute_nan_inf(feat_df)

# Calculate Z score of features
featZ = pd.DataFrame(data=stats.zscore(feat_nonan[tierpsy_256_filtered], axis=0),
                    columns=tierpsy_256_filtered,
                    index=feat_nonan.index)

# Assert no nans
assert featZ.isna().sum().sum() == 0
# Ensure the indices of featZ and meta_df are aligned
featZ = featZ.loc[meta_df.index]

# Normalize to OP50
op50_means = featZ[meta_df['bacteria_strain'] == 'OP50'].mean()
featZ_normalized = featZ - op50_means

featZ_grouped = pd.concat([featZ_normalized, meta_df], axis=1).groupby(['worm_gene']).mean()

# Make clustermap
plt.style.use(CUSTOM_STYLE)

N2clustered_features = {}
sns.set(font_scale=1.2)

for stim, fset in featsets.items():
    col_colors = featZ_grouped[fset].columns.map(feat_lut)

    cg = sns.clustermap(featZ_grouped[fset],
                        col_colors=col_colors,
                        vmin=-2,
                        vmax=2,
                        yticklabels=1,
                        cbar_pos=None,
                        figsize=(10,20))

    # remove feature labels from x axis
    cg.ax_heatmap.axes.set_xticklabels([])

    # customise y axis
    cg.ax_heatmap.axes.set_ylabel('')

    for tick_label in cg.ax_heatmap.axes.get_yticklabels():
        drug = tick_label.get_text()
        colour = colours[drug]
        tick_label.set_color(STRAIN_cmap[colour])
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=3)
    cg.ax_heatmap.tick_params(axis='y', width=0.5)

    cg.savefig(saveto / '{}_clustermap.png'.format(stim), dpi=1000)

    # get order of features
    N2clustered_features[stim] = np.array(fset)[cg.dendrogram_col.reordered_ind]

    # Write order of clustered features into .txt file
    for k, v in N2clustered_features.items():
        with open(saveto / 'clustered_features_{}.txt'.format(k), 'w+') as fid:
            for line in v:
                fid.write(line + '\n')

# save colour bar for figures
col_colors = featZ_grouped[tierpsy_256_filtered].columns.map(feat_lut)

cg = sns.clustermap(featZ_grouped[tierpsy_256_filtered],
                    col_colors=col_colors,
                    vmin=-2,
                    vmax=2,
                    yticklabels=1)

cg.ax_cbar.set_ylabel('Z score', rotation=90)

# plot colorbar only
cg.ax_heatmap.axes.remove()
cg.ax_row_dendrogram.axes.remove()
cg.ax_col_dendrogram.axes.remove()
cg.ax_col_colors.axes.remove()

cg.savefig(saveto / 'colourmap.png', dpi=1000)

plt.close('all')


#%%