#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:33:31 2023

Script for conducting in-depth stats analysis, calculating timeseries and
plotting window summaries of all positive hit compounds vs N2

@author: tobrien
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tierpsytools.analysis.significant_features import k_significant_feat
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.read_data.hydra_metadata import (align_bluelight_conditions,
                                                   read_hydra_metadata)
from tierpsytools.hydra.platechecker import fix_dtypes
import time
from tierpsytools.drug_screenings.filter_compounds import (
    compounds_with_low_effect_univariate)
from tierpsytools.analysis.statistical_tests import (univariate_tests,
                                                     _multitest_correct)

sys.path.insert(0, '/Users/tobrien/analysis_repo/DiseaseModelling/hydra_screen/phenotype_summary')
from helper import (read_disease_data,
                    select_strains,
                    filter_features,
                    make_colormaps,
                    find_window,
                    strain_gene_dict,
                    BLUELIGHT_WINDOW_DICT,
                    STIMULI_ORDER)
from plotting_helper import  (plot_colormap,
                              plot_cmap_text,
                              make_clustermaps,
                              clustered_barcodes,
                              feature_box_plots,
                              average_feature_box_plots,
                              clipped_feature_box_plots,
                              window_errorbar_plots,
                              CUSTOM_STYLE)
from ts_helper import (align_bluelight_meta,
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

from luigi_helper import load_bluelight_timeseries_from_results
# from strain_cmap import full_strain_cmap as STRAIN_cmap
#%% Set paths for data and contol strains etc

# TODO: Test then set apropiate parameters below
N2_analysis=True
ANALYSIS_TYPE = [
                # 'all_stim',
                  # 'timeseries',
                    'bluelight'
                 ] #options:['all_stim','timeseries','bluelight']
motion_modes=True
exploratory=False
do_stats=True
is_reload_timeseries_from_results = True
is_recalculate_frac_motion_modes = True

keep_percipitation = True

# Define file locations, save directory and control strain
ROOT_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/Analysis')
FEAT_FILE =  Path('/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/Results/features_summary_tierpsy_plate_20230212_162626.csv') 
FNAME_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/Results/filenames_summary_tierpsy_plate_20230212_162626.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/AuxiliaryFiles/wells_updated_metadata.csv')

WINDOW_FILES = Path('/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/Results/window_summaries')
ANALYSIS_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/Analysis')

# TODO: Sort out file paths once data transfer complete
RAW_DATA_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/Results')
ANALYSIS_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/Results/StrainTimeseries')

feats_plot = Path('/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/Analysis/selected_feats')

CONTROL_STRAIN = 'N2'  

# Strains already analysed (removes them from re-analysis), comment out the
# strains that are going to be analysed
strains_done = [
    # 'unc-80+Abitrexate',
    # 'unc-80+Carbenicillin disodium',
    # 'unc-80+D-Cycloserine',
    # 'unc-80+Ivabradine HCl',
    # 'unc-80+Mesalamine',
    # 'unc-80+Ofloxacin',
    # 'unc-80+Olanzapine',
    # 'unc-80+Rizatriptan benzoate',
    # 'unc-80+Rofecoxib',
    # 'unc-80+Sulfadoxine',
    # 'unc-80+Sulindac',
    'unc-80',
    'N2',
    # 'unc-80+Amitriptyline HCl',
    # 'unc-80+Atorvastatin calcium',
    # 'unc-80+Azatadine dimaleate',
    # 'unc-80+Ciprofloxacin',
    # 'unc-80+Clozapine',
    # 'unc-80+Daunorubicin HCl',
    # 'unc-80+Detomidine HCl',
    # 'unc-80+Idarubicin',
    # 'unc-80+Iloperidone',
    # 'unc-80+Medetomidine HCl',
    # 'unc-80+Mirtazapine',
    # 'unc-80+Moxifloxacin',
    # 'unc-80+Norfloxacin',
    # 'unc-80+Vinblastine ',
    
    # 'unc-80+Mitotane',
    # 'unc-80+Loratadine',
    # 'unc-80+Fenofibrate',
    # 'unc-80+Liranaftate',    
    # 'unc-80+Ziprasidone hydrochloride'
    ]
#%%Setting plotting styles, filtering data & renaming strains
if __name__ == '__main__':
    
    # Set stats test to be performed and set save directory for output
    which_stat_test = 'permutation_ttest'  # permutation or LMM
    if which_stat_test == 'permutation_ttest':
        figures_dir = Path('/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/Analysis/individual_compounds/N2_as_control')

    
    # CUSTOM_STYLE= mplt style card ensuring figures are consistent for papers
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    
    # Read in data and align by bluelight with tierpsy tools functions
    feat, meta = read_hydra_metadata(
        FEAT_FILE,
        FNAME_FILE,
        METADATA_FILE)

    feat, meta = align_bluelight_conditions(feat, meta, how='inner')

    # Converting metadata date into nicer format when plotting
    meta['date_yyyymmdd'] = pd.to_datetime(
        meta['date_yyyymmdd'], format='%Y%m%d').dt.date
    
    # Filter out nan's within specified columns and print .csv of these    
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'date_yyyymmdd']]

    nan_worms.to_csv(
        METADATA_FILE.parent / 'nan_worms.csv', index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)            

    if keep_percipitation==True:
        mask = meta['well_label'].isin([1.0, 3.0])
        
    meta = meta[mask]    
    feat = feat[mask]

    # Combine information about strain and drug treatment
    meta['analysis'] = meta['worm_gene'] + '+' + meta['drug_type']
    # Rename controls for ease of use
    meta['analysis'].replace({'unc-80+DMSO':'unc-80',
                              'N2+DMSO':'N2'},
                             inplace=True)
    
    # Drop water only controls (edges of plate)- we will look at DMSO only
    mask = meta['analysis'].isin(['unc-80+water', 'N2+water',
                                  'unc-80+no compound', 'N2+no compound'])
    meta = meta[~mask]
    feat = feat[~mask]
    
    # Some of the compound names have their brand name in the metadata, 
    # simply rename these to  help plots fit onto axes
    meta['analysis'].replace({
        'unc-80+Atorvastatin calcium (Lipitor)':'unc-80+Atorvastatin calcium',
        'unc-80+Daunorubicin HCl (Daunomycin HCl)':'unc-80+Daunorubicin HCl',
        'unc-80+Ciprofloxacin (Cipro)':'unc-80+Ciprofloxacin',
        'unc-80+Ivabradine HCl (Procoralan)':'unc-80+Ivabradine HCl',
        'unc-80+Iloperidone (Fanapt)':'unc-80+Iloperidone',
        'unc-80+Clozapine (Clozaril)':'unc-80+Clozapine',
        'unc-80+Mitotane (Lysodren)':'unc-80+Mitotane',
        'unc-80+Abitrexate (Methotrexate)': 'unc-80+Abitrexate',
        'unc-80+Sulindac (Clinoril)':'unc-80+Sulindac',
        'unc-80+Mesalamine (Lialda)':'unc-80+Mesalamine',
        'unc-80+Fenofibrate (Tricor, Trilipix)':'unc-80+Fenofibrate'
                            },
                             inplace=True)       
    
    # Update worm gene column with new info to reuse existing functions
    meta['worm_gene'] = meta['analysis']

    # Print out number of features processed
    feat_filters = [line for line in open(FNAME_FILE) 
                     if line.startswith("#")]
    print ('Features summaries were processed \n {}'.format(
           feat_filters))
    
    # Make summary .txt file of feats
    with open(ROOT_DIR / 'feat_filters_applied.txt', 'w+') as fid:
        fid.writelines(feat_filters)
    
    # Extract genes in metadata different from control strain and make a list
    # of the total number of straisn
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    
    # Remove strains done from gene list, so we're only analysing the strains
    # we want to
    genes = list(set(genes) - set(strains_done))
    genes.sort()
    strain_numbers = []

    imaging_date_yyyymmdd = meta['date_yyyymmdd']
    imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
    meta['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd
    
    # Select date only (remove time format) to make nice plots
    # meta['date_yyyymmdd'] = meta['date_yyyymmdd'].dt.date
    # meta['imaging_date_yyyymmdd'] = meta['imaging_date_yyyymmdd'].dt.date                                        
    
    #%% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())

    
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
                                             drop_nans=True)
            _meta['window'] = find_window(f[0])
            
            meta_windows.append(_meta)
            feat_windows.append(_feat)

        meta_windows = pd.concat(meta_windows)
        meta_windows.reset_index(drop=True,
                                 inplace=True)
        
        feat_windows = pd.concat(feat_windows)
        feat_windows.reset_index(drop=True,
                                 inplace=True)

        # Combine information about strain and drug treatment
        meta_windows['analysis'] = meta_windows['worm_gene'] + '+' + meta_windows['drug_type']
        # Rename controls for ease of use
        meta_windows['analysis'].replace({'unc-80+DMSO':'unc-80',
                                  'N2+DMSO':'N2'},
                                 inplace=True)
        
        # Drop water only controls (edges of plate)- we will look at DMSO only
        mask = meta_windows['analysis'].isin(['unc-80+water', 'N2+water',
                                      'unc-80+no compound', 'N2+no compound'])
        meta_windows = meta_windows[~mask]
        feat_windows = feat_windows[~mask]
        
        # Some of the compound names have their brand name in the metadata, 
        # simply rename these to  help plots fit onto axes
        meta_windows['analysis'].replace({
            'unc-80+Atorvastatin calcium (Lipitor)':'unc-80+Atorvastatin calcium',
            'unc-80+Daunorubicin HCl (Daunomycin HCl)':'unc-80+Daunorubicin HCl',
            'unc-80+Ciprofloxacin (Cipro)':'unc-80+Ciprofloxacin',
            'unc-80+Ivabradine HCl (Procoralan)':'unc-80+Ivabradine HCl',
            'unc-80+Iloperidone (Fanapt)':'unc-80+Iloperidone',
            'unc-80+Clozapine (Clozaril)':'unc-80+Clozapine',
            'unc-80+Mitotane (Lysodren)':'unc-80+Mitotane',
            'unc-80+Abitrexate (Methotrexate)': 'unc-80+Abitrexate',
            'unc-80+Sulindac (Clinoril)':'unc-80+Sulindac',
            'unc-80+Mesalamine (Lialda)':'unc-80+Mesalamine',
            'unc-80+Fenofibrate (Tricor, Trilipix)':'unc-80+Fenofibrate'
                                },
                                 inplace=True)       
        
        # Update worm gene column with new info to reuse existing functions
        meta_windows['worm_gene'] = meta_windows['analysis']
    

#%% N2 analysis only- makes cluster maps of N2 features

    # Ida's function strain selection function (in this case N2 only)
    if N2_analysis:
        feat_df, meta_df, idx, gene_list = select_strains(['N2'],
                                                          CONTROL_STRAIN,
                                                          feat_df=feat,
                                                          meta_df=meta)

        feat_df.drop_duplicates(inplace=True)
        meta_df.drop_duplicates(inplace=True)

        # Removes nan's, bad wells, bad days and selected tierpsy features
        feat_df, meta_df, featsets = filter_features(feat_df,
                                                     meta_df)
        
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
                                                strain_lut={'N2': 
                                                            (0.6, 0.6, 0.6)},
                                                feat_lut=feat_lut,
                                                saveto=figures_dir)

        # Write order of clustered features into .txt file
        for k, v in N2clustered_features.items():
            with open(figures_dir / 'N2_clustered_features_{}.txt'.format(k), 'w+') as fid:
                for line in v:
                    fid.write(line + '\n')
        # If not plotting heatmaps, read cluster features file and make dict
        # for plotting strain heatmaps etc later on in script
    else:
        N2clustered_features = {}
        for fset in STIMULI_ORDER.keys():
            N2clustered_features[fset] = []
            with open(figures_dir / 
                     'N2_clustered_features_{}.txt'.format(fset), 'r') as fid:
                N2clustered_features[fset] = [l.rstrip() 
                                              for l in fid.readlines()]

        with open(figures_dir / 'N2_clustered_features_{}.txt'.format('all'), 'r') as fid:
            N2clustered_features['all'] = [l.rstrip() for l in fid.readlines()]

#%%
    # if do_stats:
        # np.random.seed(seed=42)  # Uses seed to reproduce 'randomness' for each strain when testing code. Remove for 'real' analysis

    # Counting timer of individual gene selected for analysis
    for count, g in enumerate(genes):
        print('Analysing {} {}/{}'.format(g, count+1, len(genes)))
        candidate_gene = g
        
        # Set save path for figres
        saveto = figures_dir / candidate_gene
        saveto.mkdir(exist_ok=True)
        
        # Make a colour map for control and target strain- Here I use a
        # hardcoded strain cmap to keep all figures consistent for paper
        strain_lut = {}
        # candidate_gene_colour = STRAIN_cmap[candidate_gene]
        strain_lut = {CONTROL_STRAIN:(0.0, 0.0, 0.0),
                        candidate_gene: (0.6, 0.6, 0.6)}

        if 'all_stim' in ANALYSIS_TYPE:
            print ('all stim plots for {}'.format(candidate_gene))

       #Uses Ida's helper to again select individual strain for analysis
            feat_df, meta_df, idx, gene_list = select_strains([candidate_gene],
                                                              CONTROL_STRAIN,
                                                              feat_df=feat,
                                                              meta_df=meta)
       # Again filter out bad wells, nans and unwanted features
            feat_df_1, meta_df_1, featsets = filter_features(feat_df,
                                                         meta_df)

            strain_lut_old, stim_lut, feat_lut = make_colormaps(gene_list,
                                                            featlist=featsets['all'],
                                                            idx=idx,
                                                            candidate_gene=[candidate_gene],
                                                            )
        # Save colour maps as legends/figure keys for use in paper
            plot_colormap(strain_lut)
            plt.tight_layout()
            plt.savefig(saveto / 'strain_cmap.png')
            plot_cmap_text(strain_lut)
            plt.tight_layout()
            plt.savefig(saveto / 'strain_cmap_text.png')

            plot_colormap(stim_lut, orientation='horizontal')
            plt.savefig(saveto / 'stim_cmap.png')
            plot_cmap_text(stim_lut)
            plt.savefig(saveto / 'stim_cmap_text.png')

            plt.close('all')

            #%% Impute nan's and calculate Z scores of features for strains
            feat_nonan = impute_nan_inf(feat_df)

            featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], 
                                                   axis=0),
                                 columns=featsets['all'],
                                 index=feat_nonan.index)

            assert featZ.isna().sum().sum() == 0
            #%% Make a nice clustermap of features for strain & N2
            # Plotting helper saves separate cluster maps for: prestim, postim, bluelight and all conditions
            (saveto / 'clustermaps').mkdir(exist_ok=True)

            clustered_features = make_clustermaps(featZ=featZ,
                                                  meta=meta_df,
                                                  featsets=featsets,
                                                  strain_lut=strain_lut,
                                                  feat_lut=feat_lut,
                                                  saveto=saveto / 'clustermaps')
            plt.close('all')
            
            # Make a copy of the cluster map for plotting pVals and selected
            # features later on in this script without overwriting plot
            N2clustered_features_copy = N2clustered_features.copy()
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
                        _, unc_pvals, unc_reject = univariate_tests(
                            feat_nonan, y=meta_df['worm_gene'],
                            control='N2',
                            test='t-test',
                            comparison_type='binary_each_group',
                            multitest_correction=None,
                            # TODO: SET TO 10000 permutations once working
                            n_permutation_test=100000,
                            # n_permutation_test=10,
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
                    bhP_values = pd.read_csv(saveto / '{}_stats.csv'.format(candidate_gene),
                                             index_col=False)
                    bhP_values.rename(mapper={0:'p<0.05'},
                                      inplace=True)
                #%%
                #Import features to be plotted from a .txt file and make
                #swarm/boxplots and clustermaps showing stats and feats etc
                
                # Find .txt file and generate list of all feats to plot
                feat_to_plot_fname = list(feats_plot.rglob('feats_to_plot.txt'))[0]
                selected_feats = []
                with open(feat_to_plot_fname, 'r') as fid:
                    for l in fid.readlines():
                        selected_feats.append(l.rstrip().strip(','))

                all_stim_selected_feats=[]
                for s in selected_feats:
                    all_stim_selected_feats.extend([f for f in featsets['all'] if '_'.join(s.split('_')[:-1])=='_'.join(f.split('_')[:-1])])

                # Make a cluster map of strain vs N2
                clustered_barcodes(clustered_features, selected_feats,
                                    featZ,
                                    meta_df,
                                    bhP_values,
                                    saveto / 'heatmaps')

                # Use the copy of the N2 cluster map (made earlier) and plot
                # cluster map with pVals of all features alongside an asterix
                # denoting the selected features used to make boxplots
                clustered_barcodes(N2clustered_features_copy, selected_feats,
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
                
                # for f in  all_stim_selected_feats:
                #     average_feature_box_plots(f,
                #                       feat_df,
                #                       meta_df,
                #                       strain_lut,
                #                       show_raw_data='date',
                #                       bhP_values_df=bhP_values
                #                       )
                #     plt.tight_layout()
                #     plt.savefig(saveto / 'average_boxplots' / '{}_boxplot.png'.format(f),
                #                 bbox_inches='tight',
                #                 dpi=200)
                # plt.close('all')

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
            
            strain_lut_bluelight, stim_lut, feat_lut = make_colormaps(gene_list,
                                                            featlist=bluelight_feats,
                                                            idx=idx,
                                                            candidate_gene=[candidate_gene],
                                                            )

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

            # Using tierpsytools to find top 100 signifcant feats
            kfeats, scores, support = k_significant_feat(
                    feat_nonan,
                    y_classes,
                    k=100,
                    plot=False,
                    score_func='f_classif')
            
            # Grouping by stimulation number and line making plots for entire
            # experiment and each individual burst window
            stim_groups = meta_windows_df.groupby('stim_number').groups
            for f in kfeats[:50]:
                (saveto / 'windows_features' / f).mkdir(exist_ok=True)
                window_errorbar_plots(feature=f,
                                      feat=feat_windows_df,
                                      meta=meta_windows_df,
                                      cmap_lut=strain_lut)
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
            print ('timeseries plots for {}'.format(candidate_gene))                #Transfer raw data to external hdd, featuresN, run this section with RAWDATA_DIR as hDD. Set hdf5 timeseries to save to local disk and transfer back to HDD. Once _timeseries.HDF5 generated set reload results to false, then re-run.

            TS_METADATA_FILE = METADATA_FILE

            timeseries_fname = Path('/Users/tobrien/Data/disease_modelling_December2022/RawData/StrainTimeseries') / '{}_timeseries.hdf5'.format(candidate_gene)

            meta_ts = pd.read_csv(TS_METADATA_FILE,
                                  index_col=None)
            
            imaging_date_yyyymmdd = meta_ts['date_yyyymmdd']
            imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
            meta_ts['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd 
            
            meta_ts.loc[:,'imaging_date_yyyymmdd'] = meta_ts['imaging_date_yyyymmdd'].apply(lambda x: str(int(x)))
            #drop nan wells
            meta_ts.dropna(axis=0,
                        subset=['worm_gene'],
                        inplace=True) 
            # remove data from dates to exclude
            # good_date = meta_ts.query('@DATES_TO_DROP not in date_yyyymmdd').index
            
            if keep_percipitation==True:
                mask = meta_ts['well_label'].isin([1.0, 3.0])
                
            meta_ts = meta_ts[mask]    
        
            # Combine information about strain and drug treatment
            meta_ts['analysis'] = meta_ts['worm_gene'] + '+' + meta_ts['drug_type']
            # Rename controls for ease of use
            meta_ts['analysis'].replace({'unc-80+DMSO':'unc-80',
                                      'N2+DMSO':'N2'},
                                     inplace=True)
            
            # Drop water only controls (edges of plate)- we will look at DMSO only
            mask = meta_ts['analysis'].isin(['unc-80+water', 'N2+water',
                                          'unc-80+no compound', 'N2+no compound'])
            meta_ts = meta_ts[~mask]
            
            # Some of the compound names have their brand name in the metadata, 
            # simply rename these to  help plots fit onto axes
            meta_ts['analysis'].replace({
                'unc-80+Atorvastatin calcium (Lipitor)':'unc-80+Atorvastatin calcium',
                'unc-80+Daunorubicin HCl (Daunomycin HCl)':'unc-80+Daunorubicin HCl',
                'unc-80+Ciprofloxacin (Cipro)':'unc-80+Ciprofloxacin',
                'unc-80+Ivabradine HCl (Procoralan)':'unc-80+Ivabradine HCl',
                'unc-80+Iloperidone (Fanapt)':'unc-80+Iloperidone',
                'unc-80+Clozapine (Clozaril)':'unc-80+Clozapine',
                'unc-80+Mitotane (Lysodren)':'unc-80+Mitotane',
                'unc-80+Abitrexate (Methotrexate)': 'unc-80+Abitrexate',
                'unc-80+Sulindac (Clinoril)':'unc-80+Sulindac',
                'unc-80+Mesalamine (Lialda)':'unc-80+Mesalamine',
                'unc-80+Fenofibrate (Tricor, Trilipix)':'unc-80+Fenofibrate'
                                    },
                                     inplace=True)       
            
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
            strain_lut_ts, stim_lut = make_colormaps(gene_list,
                                                  [],
                                                  idx,
                                                  [candidate_gene])
  
            # Make a strain dictionary
            strain_dict = strain_gene_dict(meta_ts)
            gene_dict = {v:k for k,v in strain_dict.items()}

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
                    hires_df.to_hdf(timeseries_fname, 'hires_df', format='table') #'fixed' may help hires_df be copied correctly, issues with files not being saved properly due to time it takes and HDD connection
                except Exception:
                    print ('error creating {} HDF5 file'.format(candidate_gene))
                    
            else:  
                # Read hdf5 data and make DF
                timeseries_df = pd.read_hdf(timeseries_fname, 'timeseries_df')
                # hires_df = pd.read_hdf(timeseries_fname, 'hires_df')

            #%% Add info about the replicates for each day/plate
            # NB: the layout below is for DM_01->05 plates, 
            date_to_repl = pd.DataFrame({
                                        'date_yyyymmdd': [
                                                          '20230208',
                                                          '20230209',
                                                          '20230210'
                                                           ],
                                     'replicate': [1, 2, 3]
                                     })     
            # Merge replicates into single df
            timeseries_df = pd.merge(timeseries_df, date_to_repl,
                                     how='left',
                                     on='date_yyyymmdd')

            timeseries_df['worm_strain'] = timeseries_df['worm_gene'].map(gene_dict)
            # For working with hires rawdata, not needed for this analysis:
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
                
                        
                