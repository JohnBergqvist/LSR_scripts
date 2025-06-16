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
OD_file = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/OD/norm_od_compiled.csv')
# WINDOW_FILES = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Results/window_summaries')

# ANALYSIS_DIR = Path('/Users/bonnie/OneDrive - Imperial College London/Bode_compounds/Analysis')

# RAW_DATA_DIR = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Results')

# feats_plot = Path('/Volumes/behavgenom$/Bonnie/Bode_compounds/Initial/Analysis/Scripts')

figures_dir = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/Correct/Figures/tierpsy2/OD')

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

    # Drop wells that were recorded on two days - 20240710 and 20240720 (one excluded because of bad data, other excluded because of lack of OD measurements)
    meta = meta[(meta['date_yyyymmdd'] != 20240710) & (meta['date_yyyymmdd'] != 20240720)]

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

# %%

od_df = pd.read_csv(OD_file)
meta_copy = meta.copy()

# Merge OD data with metadata on 'bacteria_strain'
meta_copy = pd.merge(meta_copy, od_df, on='bacteria_strain', how='inner')

# drop wells with OP50
meta_copy = meta_copy[meta_copy['bacteria_strain'] != 'OP50']


#%%
combined_data = pd.concat([feat, meta_copy], axis=1)

# %% Check if there is any correlation between the speed feature and the OD measurements

# Calculate the correlation between 'speed_90th_bluelight' and 'norm_od'
correlation = combined_data['speed_90th_bluelight'].corr(combined_data['speed_90th_bluelight'])

print(f"Correlation between 'n_skeletons_bluelight' and 'speed_90th_bluelight': {correlation}")

# %%
# Calculate the correlation between each feature in 'tierpsy_256' and 'norm_od'
correlation_results = {}
for feature in tierpsy_256_filtered:
    if feature in combined_data.columns:
        correlation = combined_data[feature].corr(combined_data['norm_od'])
        correlation_results[feature] = correlation

# Print the correlation results
for feature, correlation in correlation_results.items():
    print(f"Correlation between '{feature}' and 'norm_od': {correlation}")


# %%
# Sort the correlation results from most negative to most positive
sorted_correlation_results = dict(sorted(correlation_results.items(), key=lambda item: item[1]))

# Plot the sorted correlation results
plt.figure(figsize=(15, 20))  # Increase the figure size for better readability
plt.barh(list(sorted_correlation_results.keys()), list(sorted_correlation_results.values()), color='skyblue')
plt.xlabel('Correlation with norm_od')
plt.title('Correlation of tierpsy_256 features with norm_od')
plt.grid(True)

# Adjust the font size of the labels
plt.yticks(fontsize=4)
plt.show()
# %%
# Find the feature with the maximum correlation
max_feature = max(correlation_results, key=correlation_results.get)
max_correlation = correlation_results[max_feature]
print(f"Feature with maximum correlation: {max_feature} ({max_correlation})")

# Find the feature with the minimum correlation
min_feature = min(correlation_results, key=correlation_results.get)
min_correlation = correlation_results[min_feature]
print(f"Feature with minimum correlation: {min_feature} ({min_correlation})")
# %%
