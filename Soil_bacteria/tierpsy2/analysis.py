#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:33:31 2023

Script for conducting in-depth stats analysis, calculating timeseries and
plotting window summaries of all positive hit compounds vs N2

@author: tobrien
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

#%%
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

    
#%% Plot Syngenta compounds

    if plot_syngenta: 

        # Select Syngenta compounds
    #    mask = meta['imaging_plate_drug_concentration'].isna()
    #    meta_df = meta[~mask]
    #    feat_df = feat.loc[meta_df.index]
        
        # get list of syngenta compounds
        compounds = list(meta['bacteria_strain'].unique())
        # Exclude a specific date from meta_df
        meta = meta[meta['date_yyyymmdd'] != 20240710]
        # remove DMSO
    #    del compounds[1]
        
        for c in compounds:
            saveto = figures_dir / 'tierpsy_16' / 'excluding_240710' / c
            saveto.mkdir(exist_ok=True)
            
            meta_select = meta[meta['bacteria_strain'].isin([c,'OP50'])]
            feat_select = feat.loc[meta_select.index]
            
            for t in tierpsy_16: 
                label_format = '{0:.4g}'
                plt.style.use(CUSTOM_STYLE)
                sns.set_style('ticks')
                plt.tight_layout()
                
                plt.figure(figsize=(5,10))
                ax = sns.boxplot(y=t,
                            x='bacteria_strain',
                            data=pd.concat([feat_select, meta_select],
                                            axis=1),       
                            palette='mako',
                            showfliers=False)
                plt.tight_layout()
                
                sns.swarmplot(y=t,
                              x='bacteria_strain',
                              data=pd.concat([feat_select, meta_select],
                                            axis=1),    
                              hue='date_yyyymmdd',
                              palette='Greys',
                              alpha=0.6)
                
                ax.set_ylabel(fontsize=22, ylabel=t)
                ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])
                ax.set_xlabel(xlabel= c + ' (uM)')
                plt.xticks(rotation=90)
                plt.legend(loc='upper right')
                plt.legend(title = 'date_yyyy-mm-dd', title_fontsize = 14,fontsize = 14, 
                            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                
                plt.tight_layout()
                plt.savefig(saveto / '{}_boxplot.png'.format(t),
                                        bbox_inches='tight',
                                        dpi=200)

 
    #%% Use glob to find all window summary results files in directory
    if 'bluelight' in ANALYSIS_TYPE:
        window_files = list(WINDOW_FILES.rglob('*_window_*'))
        window_feat_files = [f for f in window_files if 'features' in str(f)]
        window_feat_files.sort(key=find_window)
        window_fname_files = [f for f in window_files if 'filenames' in str(f)]
        window_fname_files.sort(key=find_window)
    
        assert (find_window(f[0]) == find_window(f[1]) for f in list(zip(
            window_feat_files, window_fname_files)))
        
    # Use Ida's helper function to read in window files and concat into DF
        feat_windows = []
        meta_windows = []
        for c,f in enumerate(list(zip(window_feat_files, window_fname_files))):
            _feat, _meta = read_disease_data(f[0],
                                             f[1],
                                             METADATA_FILE,
                                             drop_nans=False)
            # When set to True = empty dataframe 
            _meta['window'] = find_window(f[0])
            
            meta_windows.append(_meta)
            feat_windows.append(_feat)

        meta_windows = pd.concat(meta_windows)
        meta_windows.reset_index(drop=True,
                                 inplace=True)
        
        feat_windows = pd.concat(feat_windows)
        feat_windows.reset_index(drop=True,
                                 inplace=True)
        
 #%% 
 
 # TODO: make matching
        # Drop OP50 (not PFA treated) wells and drop empty wells
        mask = meta_windows['bacteria_strain'] == 'OP50'
        meta_windows = meta_windows[~mask]
        feat_windows = feat_windows[~mask]
        
        # Change wells missed by ViaFlo to 'empty'
        meta_windows.loc[meta_windows['worm_strain'].isna(), 'drug_type'] = 'Empty'
        
        # Drop empty wells
        mask = meta_windows['drug_type'] == 'Empty'
        meta_windows = meta_windows[~mask]
        feat_windows = feat_windows[~mask]
        
        # Drop bad wells
        mask = meta_windows['is_bad_well'] == True
        meta_windows = meta_windows[~mask]
        feat_windows = feat_windows[~mask]
        
        # Change formatting of date for easier plotting
        imaging_date_yyyymmdd = pd.DataFrame(meta_windows['date_yyyymmdd'])
        meta_windows['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd

        # Combine information about drug and dose
        meta_windows['analysis'] = meta_windows['drug_type'] + '_' + meta_windows['imaging_plate_drug_concentration']
        # Replace controls
        meta_windows.replace({'DMSO_0': 'DMSO', 'Water_0': 'Water'}, inplace=True)
        
        # Update worm gene column with new info to reuse existing functions
        meta_windows['worm_gene'] = meta_windows['analysis']
    
#%%
    if N2_analysis:
        control = 'OP50'
        
        # Set save path for figures
        saveto = figures_dir / control
        saveto.mkdir(exist_ok=True)
        
        feat_df, meta_df, idx, gene_list = select_strains([control],control,
                                                      feat_df=feat,
                                                      meta_df=meta)
        
        feat_df.drop_duplicates(inplace=True)
        meta_df.drop_duplicates(inplace=True)
        
        # Removes nan's, bad wells, bad days and selected tierpsy features
        feat_df, meta_df, featsets = filter_features(feat_df,
                                                  meta_df)

        # Exclude a specific date from meta_df
        meta_df = meta_df[meta_df['date_yyyymmdd'] != 20240710]

        # Make lut for imaging days
        days = meta['date_yyyymmdd'].unique()
        
        day_cmap = plt.get_cmap('Greys', days.size)
        greys_lut = [day_cmap(i) for i in range(days.size)]
        
        day_lut = dict(zip(days, greys_lut))
        
        # Save colour maps as legends/figure keys for use in paper
        plot_colormap(day_lut)
        plt.tight_layout()
        plt.savefig(saveto / 'day_cmap.png', bbox_inches='tight')
        
        plot_cmap_text(day_lut)
        plt.savefig(saveto / 'day_cmap_text.png', bbox_inches='tight')
        
        # Make a stimuli colour map/ look up table with sns
        stim_cmap = sns.color_palette('Pastel1',3)
        stim_lut = dict(zip(STIMULI_ORDER.keys(), stim_cmap))
        
        feat_lut = {f:v for f in featsets['all'] for k,v in stim_lut.items() if k in f}
        
        # Impute nans from feature dataframe
        feat_nonan = impute_nan_inf(feat_df)
        
        # Calculate Z score of features
        featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], 
                                            axis=0),
                          columns=featsets['all'],
                          index=feat_nonan.index)
        # Assert no nans
        assert featZ.isna().sum().sum() == 0
        
        # Plotting function that uses sns clustermap module- defines plot size
        # font etc... #TODO: Change params for neating up paper plots
        N2clustered_features = make_clustermaps(featZ,
                                            meta_df,
                                            featsets,
                                            strain_lut = day_lut,
                                            feat_lut=feat_lut,
                                            row_color = 'date_yyyymmdd',
                                            saveto=saveto)
        
        # Write order of clustered features into .txt file
        for k, v in N2clustered_features.items():
            with open(saveto / 'clustered_features_{}.txt'.format(k), 'w+') as fid:
                for line in v:
                    fid.write(line + '\n')
                    
        # Plot tierpsy16 boxplots
        for t in tierpsy_16: 
            label_format = '{0:.4g}'
            plt.style.use(CUSTOM_STYLE)
            sns.set_style('ticks')
            plt.tight_layout()
            
            plt.figure(figsize=(5,10))
            ax = sns.boxplot(y=t,
                        x='date_yyyymmdd',
                        data=pd.concat([feat_df, meta_df],
                                        axis=1),       
                        palette='mako',
                        showfliers=False)
            plt.tight_layout()
            
            sns.swarmplot(y=t,
                          x='date_yyyymmdd',
                          data=pd.concat([feat_df, meta_df],
                                        axis=1),    
                          hue='date_yyyymmdd',
                          palette='Greys',
                          alpha=0.6)
            
            ax.set_ylabel(fontsize=22, ylabel=t)
            ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])
            plt.xticks(rotation=90)
            ax.get_legend().remove()
            
            plt.tight_layout()
            plt.savefig(saveto / '{}_boxplot.png'.format(t),
                                    bbox_inches='tight',
                                    dpi=200)
            
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
saveto = figures_dir / 'Clustermaps' / 'excluding_240710' / 'tierpsy_256' / 'excluding_jub134' / 'normalised_to_OP50'
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


#%% Plotting blueight_speed_90th for all bacterial species that are significantly different from OP50

# Structuring data

    # set significane level
    alpha = 0.05


    combined_data = pd.concat([feat, meta], axis=1)
    combined_data = combined_data.loc[combined_data['date_yyyymmdd'].isin([20240720, 20240816, 20240830])]


    # Find starins that are significantly different from OP50 in speed_90th_bluelight
    significant_strains = []
    op50_speeds = combined_data[combined_data['bacteria_strain'] == 'OP50']['speed_90th_bluelight']

    for strain in combined_data['bacteria_strain'].unique():
        if strain != 'OP50':
            strain_speeds = combined_data[combined_data['bacteria_strain'] == strain]['speed_90th_bluelight']
            stat, p_value = mannwhitneyu(op50_speeds, strain_speeds, alternative='two-sided')
            
            # Check if the p-value is below the significance level
            if p_value < alpha:
                # Filter data for OP50 and the current strain
                filtered_data = combined_data[combined_data['bacteria_strain'].isin(['OP50', strain])]
                
                # Plot
                plt.figure(figsize=(8, 10))
                plt.style.use(CUSTOM_STYLE)
                sns.set_style('ticks')
                
                ax = sns.boxplot(y='speed_90th_bluelight', x='worm_gene', data=filtered_data, hue='bacteria_strain', showfliers=False)
                plt.title(f"Comparison of {strain} vs OP50\n{format_p_value(p_value)}")
                
                # Adjust legend and labels
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles[0:], labels=labels[0:], title='Bacteria Strain')
                ax.set_ylabel('Speed 90th Percentile Bluelight', fontsize=30)
                ax.set_xlabel('')

                sns.stripplot(y='speed_90th_bluelight', x='worm_gene', data=filtered_data, hue='bacteria_strain', dodge=False, jitter=True, color='black', alpha=0.5, ax=ax)

                handles, labels = ax.get_legend_handles_labels()
                l = plt.legend(handles[0:2], labels[0:2], title='Bacteria Strain')
                ax.set_xlabel('')

                plt.tight_layout()
                
                # Define the directory to save the figures
                save_dir = "/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/Figures/tierpsy2/speed_90th_bluelight/significant_bacteria"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                # Construct the file path
                file_path = os.path.join(save_dir, f"{strain}_vs_OP50_speed_90th_bluelight.png")
                
                # Save the plot
                plt.savefig(file_path, dpi=300)

                plt.show()


#%% Make interactive heatmap

#     cg = sns.clustermap(featZ_grouped[featsets['all']], 
#                                     vmin=-2,
#                                     vmax=2
#                                     # ,metric ='cosine'
#                                     )
                
#     plt.close('all')

#     # get order of features and bacteria strain/day in clustermap
#     row_order = cg.dendrogram_row.reordered_ind
#     col_order = cg.dendrogram_col.reordered_ind     

#     # re-order df to match clustering
#     clustered_df_final = featZ_grouped.loc[featZ_grouped.index[row_order], featZ_grouped.columns[col_order]]
    
#     # # create new index for y labels
#     # clustered_df_final = clustered_df_final.reset_index()
#     # label_values = clustered_df_final['worm_gene'].astype(str) + '_' + clustered_df_final['imaging_date_yyyymmdd'].astype(str)
#     # clustered_df_final = clustered_df_final.set_index(label_values)

#     # Define your heatmap
#     intheatmap = (
#         go.Heatmap(x=clustered_df_final.columns, 
#                    y=clustered_df_final.index, 
#                    z=clustered_df_final.values,  
#                    colorscale='Inferno', # Try RdBu or something
#                    zmin=-2,
#                    zmax=2,
#                    showscale=True)
#     )
    
#     intfig_cl = go.Figure(data=intheatmap)
#     intfig_cl.update_xaxes(showticklabels=False)  
#     intfig_cl.update_yaxes(showticklabels=False, autorange="reversed") 
    
#     # Define your layout, adjusting colorbar size
#     intfig_cl.update_layout({
#         'width': 1200,
#         'height': 550,
#         'margin': dict(l=0,r=0,b=0,t=0),
#         'showlegend': True,
#         'hovermode': 'closest',
#     })
    
#     plot(intfig_cl, filename = str(saveto / "InteractiveClustermap.html"),
#          config={"displaylogo": False,
#                  "displayModeBar": True,
#                  "scale":10},
#          auto_open=False 
# )

            
#%%
    # make dictionary of compounds and plate controls
    genes = (dict(zip(meta.worm_gene,meta.control)))
    
    # drop nan values e.g. DMSO
    genes = {key: value for key, value in genes.items() if value is not None and not isinstance(value, float) or not math.isnan(value)}
    
    # drop Syngenta compounds
    genes = {key: value for key, value in genes.items() if value != 'DMSO'}
    
    # only keep compounds that have controls
    drugs = list(meta['worm_gene'].unique())
    genes = {key: value for key, value in genes.items() if value in drugs}
    
#%%
    # Counting timer of individual gene selected for analysis
    for count, g in enumerate(genes):
        print('Analysing {} {}/{}'.format(g, count+1, len(genes)))
        candidate_gene = g
        CONTROL_STRAIN = genes[g]
        
        # Set save path for figures
        saveto = figures_dir / candidate_gene / 'plate_control'
        saveto.mkdir(exist_ok=True)
        
        # Make a colour map for control and target drug
        colour = colours[candidate_gene] 
        
        strain_lut = {CONTROL_STRAIN:STRAIN_cmap[CONTROL_STRAIN],
                        candidate_gene: STRAIN_cmap[colour]}

        if 'all_stim' in ANALYSIS_TYPE:
            print ('all stim plots for {}'.format(candidate_gene))

       #Uses Ida's helper to again select individual strain for analysis
       # TODO: only selects controls from same imaging days as candidate
       # Changed in function
            feat_df, meta_df, idx, gene_list = select_strains([candidate_gene],
                                                              CONTROL_STRAIN,
                                                              feat_df=feat,
                                                              meta_df=meta)
       # Again filter out bad wells, nans and unwanted features
            feat_df_1, meta_df_1, featsets = filter_features(feat_df,
                                                         meta_df)

            strain_lut_old, stim_lut, feat_lut = make_colormaps(gene_list,
                                                            featlist=featsets['all'],
                                                            CONTROL_STRAIN = CONTROL_STRAIN,
                                                            idx=idx,
                                                            candidate_gene=[candidate_gene],
                                                            )
        # Save colour maps as legends/figure keys for use in paper
            plot_colormap(strain_lut)
            plt.tight_layout()
            plt.savefig(saveto / 'strain_cmap.png', bbox_inches='tight')
            
            plot_cmap_text(strain_lut)
            plt.tight_layout()
            plt.savefig(saveto / 'strain_cmap_text.png',  bbox_inches='tight')

            plot_colormap(stim_lut, orientation='horizontal')
            plt.savefig(saveto / 'stim_cmap.png',  bbox_inches='tight')
            plot_cmap_text(stim_lut)
            plt.savefig(saveto / 'stim_cmap_text.png',  bbox_inches='tight')

            plt.close('all')

            #%% Impute nan's and calculate Z scores of features for strains
            feat_nonan = impute_nan_inf(feat_df)

            featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], 
                                                   axis=0),
                                 columns=featsets['all'],
                                 index=feat_nonan.index)

            assert featZ.isna().sum().sum() == 0
            
            # #%% Make a nice clustermap of features for strain & N2
            # Plotting helper saves separate cluster maps for: prestim, postim, bluelight and all conditions
            (saveto / 'clustermaps').mkdir(exist_ok=True)

            clustered_features = make_clustermaps(featZ=featZ,
                                                  meta=meta_df,
                                                  featsets=featsets,
                                                  strain_lut=strain_lut,
                                                  feat_lut=feat_lut,
                                                  saveto=saveto / 'clustermaps')
            plt.close('all')
            
            # Make a copy of the cluster map for plotting pvals and selected
            # features later on in this script without overwriting plot
            clustered_features_copy = clustered_features.copy()
                
            #%% Calculate top 100 significant feats and make boxplots vs N2
            if exploratory:
                (saveto / 'ksig_feats').mkdir(exist_ok=True)
                sns.set_style('white')
                label_format = '{:.4f}'
                kfeats = {}
                # Looks at feature sets for each stimuli periods
                for stim, fset in featsets.items():
                    kfeats[stim], scores, support = k_significant_feat(
                        feat_nonan[fset],
                        meta_df.worm_gene,
                        k=100,
                        plot=False)
                # Formatting 20 boxplots on one figure:
                    for i in range(0,5):
                        fig, ax = plt.subplots(4, 5, sharex=True, figsize = [20,20])
                        for c, axis in enumerate(ax.flatten()):
                            counter=(20*i)-(20-c)
                            sns.boxplot(x=meta_df['worm_gene'],
                                        order = [CONTROL_STRAIN, candidate_gene],
                                        y=feat_df[kfeats[stim][counter]],
                                        palette=strain_lut.values(),
                                        ax=axis)
                            axis.set_ylabel(fontsize=8,
                                            ylabel=kfeats[stim][counter])
                            axis.set_yticklabels(labels=[label_format.format(x) for x in axis.get_yticks()],
                                                 fontsize=6)
                            axis.set_xlabel('')
                        plt.tight_layout()
                        fig.fontsize=11
                        plt.savefig(saveto / 'ksig_feats' / '{}_{}_ksig_feats.png'.format(i*20, stim),
                                    dpi=400)
                        plt.close('all')

            else:
                (saveto / 'heatmaps').mkdir(exist_ok=True)
                (saveto / 'heatmaps_N2ordered').mkdir(exist_ok=True)
                (saveto / 'boxplots').mkdir(exist_ok=True)

            # If we're not printing the top 100 significant feats, calculate
            # stats using permutation t-tests or LMM tierpsy functions:
                # (Note that we're only interested in pvals and rejects)
                if do_stats:
                    if which_stat_test == 'permutation_ttest':
                                                # Set save path for figures
                        # saveto = saveto / 'permutation_ttest'
                        # saveto.mkdir(exist_ok=True)
                        
                        _, unc_pvals, unc_reject = univariate_tests(
                            feat_nonan, y=meta_df['worm_gene'],
                            control=CONTROL_STRAIN,
                            test='t-test',
                            comparison_type='binary_each_group',
                            multitest_correction=None,
                            # TODO: SET TO 10000 permutations once working
                            # n_permutation_test=1,
                            n_permutation_test=100000,
                            perm_blocks=meta_df['imaging_date_yyyymmdd'],
                            )
                        reject, pvals = _multitest_correct(
                            unc_pvals, 'fdr_by', 0.05)
                        unc_pvals = unc_pvals.T
                        pvals = pvals.T
                        reject = reject.T
                        
                    elif which_stat_test == 'LMM':
                        # lmm is ok with nans, other tests did not
                        _, _, _, reject, pvals = compounds_with_low_effect_univariate(
                            feat_df, meta_df['worm_gene'],
                            drug_dose=None,
                            random_effect=meta_df['imaging_date_yyyymmdd'],
                            control='N2',
                            test='LMM',
                            comparison_type='binary_each_dose',
                            multitest_method='fdr_by',
                            fdr=0.05,
                            n_jobs=-1
                            )
                    
                    elif which_stat_test == 't-test':
                        # # Set save path for figures
                        # saveto = saveto / 't-test'
                        # saveto.mkdir(exist_ok=True)
        
                        _, unc_pvals, unc_reject = univariate_tests(
                            feat_nonan, y=meta_df['worm_gene'],
                            control=CONTROL_STRAIN,
                            test='t-test',
                            comparison_type='binary_each_group',
                            multitest_correction=None,
                            perm_blocks=meta_df['imaging_date_yyyymmdd'],
                            )
                        reject, pvals = _multitest_correct(
                            unc_pvals, 'fdr_by', 0.05)
                        unc_pvals = unc_pvals.T
                        pvals = pvals.T
                        reject = reject.T
                        
                    else:
                        raise ValueError((
                            f'Invalid value "{which_stat_test}"'
                            ' for which_stat_test'))
                    # massaging data to be in keeping with downstream analysis
                    assert pvals.shape[0] == 1, 'the output is supposed to be one line only I thought'
                    assert all(reject.columns == pvals.columns)
                    assert reject.shape == pvals.shape
                    # set the pvals over threshold to NaN - These are set to nan for convinence later on
                    bhP_values = pvals.copy(deep=True)
                    bhP_values.loc[:, ~reject.iloc[0, :]] = np.nan
                    bhP_values['worm_gene'] = candidate_gene
                    bhP_values.index = ['p<0.05']

                    # check the right amount of features was set to nan
                    assert reject.sum().sum() == bhP_values.notna().sum().sum()-1
                    
                    # also save the corrected and uncorrected pvalues, without
                    # setting the rejected ones to nan, just keeping the same
                    # dataframe format as bhP_values
                    for p_df in [unc_pvals, pvals]:
                        p_df['worm_gene'] = candidate_gene
                        p_df.index = ['p value']
                    unc_pvals.to_csv(
                        saveto/f'{candidate_gene}_uncorrected_pvals.csv',
                        index=False)
                    pvals.to_csv(
                        saveto/f'{candidate_gene}_fdrby_pvals.csv',
                        index=False)
                    # Save total number of significant feats as .txt file
                    with open(saveto / 'sig_feats.txt', 'w+') as fid:
                        fid.write(str(bhP_values.notna().sum().sum()-1) + ' significant features out of \n')
                        fid.write(str(bhP_values.shape[1]-1))

                    bhP_values.to_csv(saveto / '{}_stats.csv'.format(candidate_gene),
                                      index=False)
                    
                # If not calculating stats, read the .csv file for plotting
                else:
                    # TODO: change for vs. strain control or permutation test
                    bhP_values = pd.read_csv(saveto / '{}_fdrby_pvals.csv'.format(candidate_gene),
                                             index_col=False)
                    bhP_values.rename(mapper={0:'p<0.05'},
                                      inplace=True)
                #%%
                #Import features to be plotted from a .txt file and make
                #swarm/boxplots and clustermaps showing stats and feats etc
                
                if 'all' in feats_to_plot:
                    # get list of features to make boxplots for
                    sig_feats = bhP_values.dropna(axis=1)
                    sig_feats = sig_feats.drop(columns='worm_gene')
                    all_stim_selected_feats = list(sig_feats.columns)
                    
                     # if you don't want stars on clustermaps
                    selected_feats=[]
                
                elif 'select' in feats_to_plot:
                   
                   # # Find .txt file and generate list of all feats to plot
                    selected_feats = []
                    with open(saveto / 
                          'feats_to_plot.txt', 'r') as fid:
                        for l in fid.readlines():
                            selected_feats.append(l.rstrip().strip(','))
                       
                    all_stim_selected_feats = selected_feats
                    
                elif 'tierpsy_16' in feats_to_plot:
                  
                    all_stim_selected_feats = []
                    with open(
                            '/Volumes/behavgenom$/Bonnie/Scripts/Tierpsy_16.csv','r') as fid:
                      for l in fid.readlines():
                            all_stim_selected_feats.append(l.rstrip().strip(','))

                      selected_feats=[]
        

                # Make a cluster map of strain vs N2
                clustered_barcodes(clustered_features, selected_feats,
                                    featZ,
                                    meta_df,
                                    bhP_values,
                                    saveto / 'heatmaps')

                # Use the copy of the N2 cluster map (made earlier) and plot
                # cluster map with pVals of all features alongside an asterix
                # denoting the selected features used to make boxplots
                clustered_barcodes(clustered_features_copy, selected_feats,
                                    featZ,
                                    meta_df,
                                    bhP_values,
                                    saveto / 'heatmaps_N2ordered')

                # Generate boxplots of selected features containing correct
                # pValues and formatted nicely
                for f in  all_stim_selected_feats:
                    feature_box_plots(f,
                                      feat_df,
                                      meta_df,
                                      strain_lut,
                                      show_raw_data='date',
                                      bhP_values_df=bhP_values
                                      )
                    plt.tight_layout()
                    plt.savefig(saveto / 'boxplots' / '{}_boxplot.png'.format(f),
                                bbox_inches='tight',
                                dpi=200)
                plt.close('all')
    

        #%% Using window feature summaries to look at bluelight conditions
        if 'bluelight' in ANALYSIS_TYPE:
            print ('all window_plots for {}'.format(candidate_gene))
            
           # Call dataframes window specific dataframes (made earlier)
            feat_windows_df, meta_windows_df, idx, gene_list = select_strains(
                                                          [candidate_gene],
                                                          CONTROL_STRAIN,
                                                          meta_windows,
                                                          feat_windows)

            # Filter out only the bluelight features
            bluelight_feats = [f for f in feat_windows_df.columns if 'bluelight' in f]
            feat_windows_df = feat_windows_df.loc[:,bluelight_feats]

            feat_windows_df, meta_windows_df, featsets = filter_features(feat_windows_df,
                                                                    meta_windows_df)
            
            bluelight_feats = list(feat_windows_df.columns)
            
            strain_lut_old, stim_lut, feat_lut = make_colormaps(gene_list,
                                                            featlist=bluelight_feats,
                                                            CONTROL_STRAIN = CONTROL_STRAIN,
                                                            idx=idx,
                                                            candidate_gene=[candidate_gene],
                                                            )

            strain_lut_bluelight = strain_lut
            
            #%% Fill nans and calculate Zscores of window feats
            feat_nonan = impute_nan_inf(feat_windows_df)

            featZ = pd.DataFrame(data=stats.zscore(feat_nonan[bluelight_feats], axis=0),
                                  columns=bluelight_feats,
                                  index=feat_nonan.index)

            assert featZ.isna().sum().sum() == 0

            #%%
            # Find top significant feats that differentiate between prestim and bluelight
            
            #make save directory and set layout for plots using dictionary
            (saveto / 'windows_features').mkdir(exist_ok=True)
            meta_windows_df['light'] = [x[1] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
            meta_windows_df['window_sec'] = [x[0] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
            meta_windows_df['stim_number'] = [x[2] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]

            y_classes = ['{}, {}'.format(r.worm_gene, r.light) for i,r in meta_windows_df.iterrows()]

            # # Using tierpsytools to find top 100 signifcant feats
            # kfeats, scores, support = k_significant_feat(
            #         feat_nonan,
            #         y_classes,
            #         # k=100,
            #         k=1,
            #         plot=False,
            #         score_func='f_classif')
                
            # Grouping by stimulation number and line making plots for entire
            # experiment and each individual burst window
            stim_groups = meta_windows_df.groupby('stim_number').groups
            
            # Find .txt file and generate list of tierpsy16 features to plot
            feat_to_plot_fname = list(feats_plot.rglob('feats_to_plot.txt'))[0]
            selected_feats = []
            with open(feat_to_plot_fname, 'r') as fid:
                for l in fid.readlines():
                    selected_feats.append(l.rstrip().strip(','))
            # add speed
            selected_feats.append('speed_90th_bluelight')
            
            # for f in kfeats[:50]:
            for f in selected_feats:
                (saveto / 'windows_features' / f).mkdir(exist_ok=True)
                window_errorbar_plots(feature=f,
                                      feat=feat_windows_df,
                                      meta=meta_windows_df,
                                      cmap_lut=strain_lut,
                                      plot_legend=True)
                plt.savefig(saveto / 'windows_features' / f / 'allwindows_{}'.format(f), dpi=200)
                plt.close('all')

            for stim,locs in stim_groups.items():
                    window_errorbar_plots(feature=f,
                                          feat=feat_windows_df.loc[locs],
                                          meta=meta_windows_df.loc[locs],
                                          cmap_lut=strain_lut)
                    plt.savefig(saveto / 'windows_features' / f / 'window{}_{}'.format(stim,f),
                                dpi=200)
                    plt.close('all')

            #%% Calculating motion modes from bluelight features and making 
            # plots of these- saved in a sub-folder within bluelight analysis
            if motion_modes:
                mm_feats = [f for f in bluelight_feats if 'motion_mode' in f]
                (saveto / 'windows_features' / 'motion_modes').mkdir(exist_ok=True)
                sns.set_style('ticks')
                for f in mm_feats:
                    window_errorbar_plots(feature=f,
                                          feat=feat_windows_df,
                                          meta=meta_windows_df,
                                          cmap_lut=strain_lut)
                    plt.savefig(saveto / 'windows_features' / 'motion_modes' / '{}'.format(f),
                                dpi=200)
                    plt.close('all')
                    for stim,locs in stim_groups.items():
                        window_errorbar_plots(feature=f,
                                              feat=feat_windows_df.loc[locs],
                                              meta=meta_windows_df.loc[locs],
                                              cmap_lut=strain_lut)
                        plt.savefig(saveto / 'windows_features' / 'motion_modes' / 'window{}_{}'.format(stim,f),
                                    dpi=200)
                        plt.close('all')

        #%% Make timerseries plots

        if 'timeseries' in ANALYSIS_TYPE:
            print ('timeseries plots for {}'.format(candidate_gene))       
            
            #Transfer raw data to external hdd, featuresN, 
            # run this section with RAWDATA_DIR as hDD. 
            # Set hdf5 timeseries to save to local disk and transfer back to HDD. 
            # Once _timeseries.HDF5 generated set reload results to false, then re-run.

            TS_METADATA_FILE = METADATA_FILE
            
            timeseries_fname = Path(
                '/Users/bonnie/OneDrive - Imperial College London/Bode_compounds/Analysis/Figures/TimeSeries'
                ) / '{}_timeseries.hdf5'.format(candidate_gene)

            meta_ts = pd.read_csv(TS_METADATA_FILE,
                                  index_col=None)
            
            imaging_date_yyyymmdd = meta_ts['date_yyyymmdd']
            imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
            meta_ts['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd 
            
            meta_ts.loc[:,'imaging_date_yyyymmdd'] = meta_ts[
                'imaging_date_yyyymmdd'].apply(lambda x: str(int(x)))
            
            # Drop OP50 (not PFA treated) wells and drop empty wells
            mask = meta_ts['bacteria_strain']=='OP50'
            meta_ts = meta_ts[~mask]
            
            # Drop empty wellsx
            mask = meta_ts['drug_type']=='Empty'
            meta_ts = meta_ts[~mask]
    
            # Combine information about drug and dose
            meta_ts['analysis'] = meta_ts['drug_type'] + '_' + meta_ts['imaging_plate_drug_concentration']
            # Replace controls
            meta_ts.replace({'DMSO_0':'DMSO','Water_0':'Water'},inplace=True)
                
            
            # Update worm gene column with new info to reuse existing functions
            meta_ts['worm_gene'] = meta_ts['analysis']
                    
            meta_ts['number_worms_per_well'] = 3
            meta_ts = meta_ts.drop(columns= ['date_bleached_yyyymmdd', 'date_refed_yyyymmdd'])
            meta_ts = fix_dtypes(meta_ts)

            # Select candidate gene
            meta_ts, idx, gene_list = select_strains([candidate_gene],
                                                      CONTROL_STRAIN,
                                                      meta_df=meta_ts)
            # Make strain and stimuli colour maps
            strain_lut_old, stim_lut = make_colormaps(gene_list,
                                                  [],
                                                  CONTROL_STRAIN,
                                                  idx,
                                                  [candidate_gene])
            
            strain_lut_ts = strain_lut
  
            # # Make a strain dictionary
            # strain_dict = strain_gene_dict(meta_ts)
            # gene_dict = {v:k for k,v in strain_dict.items()}

            # Align by bluelight conditions 
            meta_ts = align_bluelight_meta(meta_ts)

            if is_reload_timeseries_from_results:
                # Uses tierpsy under hood to calculate timeseries and motion modes
                timeseries_df, hires_df  = load_bluelight_timeseries_from_results(
                                    meta_ts,
                                    RAW_DATA_DIR,
                                    saveto=None)
                                    # save to disk when calculating (takes long time to run)
                try:
                    timeseries_df.to_hdf(timeseries_fname, 'timeseries_df', format='table')
                    hires_df.to_hdf(timeseries_fname, 'hires_df', format='table') 
                    #'fixed' may help hires_df be copied correctly, 
                    # issues with files not being saved properly due to time it takes and HDD connection
                except Exception:
                    print ('error creating {} HDF5 file'.format(candidate_gene))
                    
            else:  
                # Read hdf5 data and make DF
                timeseries_df = pd.read_hdf(timeseries_fname, 'timeseries_df')
                hires_df = pd.read_hdf(timeseries_fname, 'hires_df')

            #%% Add info about the replicates for each day/plate
            # NB: the layout below is for DM_01->05 plates, 
            date_to_repl = pd.DataFrame({
                                        'date_yyyymmdd': [
                                                          '20230302',
                                                          '20230309',
                                                          '20230310'
                                                            ],
                                      'replicate': [1, 2, 3]
                                      })     
            # Merge replicates into single df
            timeseries_df = pd.merge(timeseries_df, date_to_repl,
                                      how='left',
                                      on='date_yyyymmdd')

            # timeseries_df['worm_strain'] = timeseries_df['worm_gene'].map(gene_dict)
            # # For working with hires rawdata, not needed for this analysis:
            # hires_df['worm_strain'] = hires_df['worm_gene'].map(gene_dict)

            #make d/v signed features absolute as in hydra d/v is not assigned
            timeseries_df = make_feats_abs(timeseries_df)

            # %%% Plot hand-picked features from the downsampled dataframe

            plt.close('all')
            (saveto / 'ts_plots').mkdir(exist_ok=True)
            feats_toplot = ['speed',
                            'abs_speed',
                            'angular_velocity',
                            'abs_angular_velocity',
                            'relative_to_body_speed_midbody',
                            'abs_relative_to_body_speed_midbody',
                            'abs_relative_to_neck_angular_velocity_head_tip',
                            'speed_tail_base',
                            'length',
                            'major_axis',
                            'd_speed',
                            'head_tail_distance',
                            'abs_angular_velocity_neck',
                            'abs_angular_velocity_head_base',
                            'abs_angular_velocity_hips',
                            'abs_angular_velocity_tail_base',
                            'abs_angular_velocity_midbody',
                            'abs_angular_velocity_head_tip',
                            'abs_angular_velocity_tail_tip',
                            'd_curvature_std_head'
                            ]

            # Uses plotting helper to make lineplots with confidence intervals
            # of handpicked features over time (includes bluelight bursts on fig)
            plot_strains_ts(timeseries_df=timeseries_df,
                            strain_lut=strain_lut_ts,
                            CONTROL_STRAIN=CONTROL_STRAIN,
                            features=feats_toplot,
                            SAVETO=saveto / 'ts_plots')
            plt.close('all')
            #%% Plot entire motion modes
            # Calculates motion modes (fraction of paused, fws/bck worms) and save as .hdf5
            tic = time.time()
            if is_recalculate_frac_motion_modes:
                motion_modes, frac_motion_modes_with_ci = get_motion_modes(hires_df,
                                                                            saveto=timeseries_fname
                                                                            )
            # If motion modes already calculated, reload them 
            else:
                frac_motion_modes_with_ci = pd.read_hdf(timeseries_fname,
                                                        'frac_motion_mode_with_ci')

            frac_motion_modes_with_ci['worm_strain'] = frac_motion_modes_with_ci['worm_gene'].map(gene_dict)

            fps = 25
            frac_motion_modes_with_ci = frac_motion_modes_with_ci.reset_index()
            frac_motion_modes_with_ci['time_s'] = (frac_motion_modes_with_ci['timestamp']
                                                  / fps)
            print('Time elapsed: {}s'.format(time.time()-tic))
                
            #%% Utilising Luigi's boostraping functions to make ts plot
            # plot forwawrd,backward and stationary on one plot for each strain
            # plots are coloured by cmap defined earlier on
            
            for ii, (strain, df_g) in enumerate(frac_motion_modes_with_ci.groupby('worm_gene')):
                plot_frac_all_modes(df_g, strain, strain_lut_ts)
                plt.savefig(saveto / '{}_ts_motion_modes_coloured_by_strain.png'.format(strain), dpi=200)
                
            #%% Same as above, but each motion mode coloured differently
                  
            for iii, (strain, df_g) in enumerate(frac_motion_modes_with_ci.groupby('worm_gene')):
                plot_frac_all_modes_coloured_by_motion_mode(df_g, strain, strain_lut_ts)
                plt.savefig(saveto / '{}_ts_coloured_by_motion_modes.png'.format(strain), dpi=200)      
                
            #%% Plot each motion mode separately
            
            for motion_mode in MODECOLNAMES:
                plot_frac_by_mode(df=frac_motion_modes_with_ci, 
                                  strain_lut=strain_lut_ts, 
                                  modecolname=motion_mode)
                plt.savefig(saveto / '{}_ts.png'.format(motion_mode), dpi=200)
            #%% First stimuli plots
            time_drop = frac_motion_modes_with_ci['time_s']>160
            frac_motion_modes_with_ci = frac_motion_modes_with_ci.loc[~time_drop,:]             
            for motion_mode in MODECOLNAMES:
                short_plot_frac_by_mode(df=frac_motion_modes_with_ci, 
                                        strain_lut=strain_lut_ts, 
                                        modecolname=motion_mode)
                plt.savefig(saveto / '{}_first_stimuli_ts.png'.format(motion_mode), dpi=200)     
                
                        
                