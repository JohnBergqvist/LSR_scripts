#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script for plotting pixel difference 

@author: John Bergqvist 2025
"""

#%%
import numpy as np
from natsort import natsorted
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
#from skimage import filters, measure
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict
from IPython.display import display
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols



#%%

# Load the data
# Pre-normalised data
all_data_df = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/batch1_batch2_batch3_batch4_data.csv')


# Normalised data
df_norm = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/batch1_batch2_batch3_batch4_data_normalised.csv')


#%%
# Set which plots to print for making plots of pixel difference (not statistical analysis)
by_strain = False
by_drug = False

by_strain_norm = False
by_drug_norm = False
heatmap_norm = False

#%%
# Plotting dependent on the selected plots in the above code block

# Define color map for each drug
color_map = {
    'Aldicarb': 'red',
    'Levamisole': 'blue',
    'Ivermectin': 'green',
    'Chlorpromazine': 'purple',
    'S.H2O': 'gray',
    'DMSO': 'gray'
}

# Function to create a directory if it doesn't exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to map drug names to colors
def get_color(drug_concentration):
    drug = drug_concentration.split()[0]
    return color_map.get(drug, 'black')

# Function to create a color palette for the stripplot based on unique dates
def create_date_color_palette(dates):
    unique_dates = dates.unique()
    if len(unique_dates) == 1:
        return {unique_dates[0]: 'black'}
    else:
        palette = sns.color_palette("colorblind", len(unique_dates))
        return dict(zip(unique_dates, palette))

# Plot the pixel variance data for each strain (All imaging days for N2 combined)
# Plot all drugs and concentrations in one plot per strain
if by_strain == True:
    strains = all_data_df['strain'].unique()
    for strain in strains:
        plt.figure(figsize=(12, 8))
        
        strain_data = all_data_df[all_data_df['strain'] == strain]
        
        if strain == 'N2':
            # Extract unique dates and create a color palette
            date_palette = create_date_color_palette(strain_data['date_yyyymmdd'])
            
            # Create a combined column for drug and concentration
            strain_data['drug_concentration'] = strain_data['drug'] + ' ' + strain_data['concentration'].astype(str) + 'µM'
            
            # Create a color palette for the drug_concentration column
            unique_drug_concentrations = strain_data['drug_concentration'].unique()
            palette = {dc: get_color(dc) for dc in unique_drug_concentrations}
            
            # Create the boxplot
            boxplot = sns.boxplot(data=strain_data, x='drug_concentration', y='value', palette=palette)
            
            # Create the striplot to show all data points with colors based on date
            striplot = sns.stripplot(data=strain_data, x='drug_concentration', y='value', hue='date_yyyymmdd', palette=date_palette, jitter=True, marker='o', alpha=0.8)
            
            # Plot all values for wells in row E (S.H2O) and row F (DMSO)
            #sh2o_values = all_data_df[(all_data_df['strain'] == strain) & all_data_df['well'].str.startswith('E')]['value']
            #dmso_values = all_data_df[(all_data_df['strain'] == strain) & all_data_df['well'].str.startswith('F')]['value']
            
            # Add labels and title
            plt.xlabel('Drug and Concentration')
            plt.ylabel('Pixel Variation - not normalised')
            plt.title(f"Strain: {strain}")
            
            # Change x-axis label orientation
            plt.xticks(rotation=90)
            plt.legend(title='Date')

            # Save the plot with high DPI
            #strain = plate_data['strain'].iloc[0]  # Assuming 'strain' column exists
            plt.savefig(SAVE_FIG_DIR / 'all' / 'by_strain' / f'{strain}_pixelvariance_all_batches.png', dpi=300, bbox_inches='tight')

            #plt.show()  

        else:
                # Extract unique dates
            unique_dates = strain_data['date_yyyymmdd'].unique()
            
            for date in unique_dates:
                plt.figure(figsize=(12, 8))
                
                date_data = strain_data[strain_data['date_yyyymmdd'] == date]
                
                # Create a combined column for drug and concentration
                date_data['drug_concentration'] = date_data['drug'] + ' ' + date_data['concentration'].astype(str) + 'µM'
                
                # Create a color palette for the drug_concentration column
                unique_drug_concentrations = date_data['drug_concentration'].unique()
                palette = {dc: get_color(dc) for dc in unique_drug_concentrations}
                
                # Create the boxplot
                boxplot = sns.boxplot(data=date_data, x='drug_concentration', y='value', palette=palette)
                
                # Create the striplot to show all data points with colors based on date
                striplot = sns.stripplot(data=date_data, x='drug_concentration', y='value', hue='date_yyyymmdd', palette={date: 'black'}, jitter=True, marker='o', alpha=0.8)
                
                # Add labels and title
                plt.xlabel('Drug and Concentration')
                plt.ylabel('Pixel Variation - not normalised')
                plt.title(f"Strain: {strain}")
                
                # Change x-axis label orientation
                plt.xticks(rotation=90)
                plt.legend(title='Date')

                # Save the plot with high DPI
                date_dir = SAVE_FIG_DIR / 'all' / 'by_strain' / date
                create_directory(date_dir)
                plt.savefig(date_dir / f'{date}_{strain}_pixelvariance.png', dpi=300, bbox_inches='tight')
                #plt.show()

    

# Plot pixel variance data by drug and concentration (All imaging days for N2 combined)
if by_drug == True:
    # Get unique combinations of drug and concentration
    drug_concentrations = all_data_df[['drug', 'concentration']].drop_duplicates()

    # Determine the range of concentrations
    concentration_values = drug_concentrations['concentration'].unique()
    concentration_values_sorted = np.sort(concentration_values)

    # Create a mapping from concentration to alpha value
    alpha_map = {concentration: (i + 1) / len(concentration_values_sorted) for i, concentration in enumerate(concentration_values_sorted)}
    
    # Extract unique dates
    unique_dates = all_data_df['date_yyyymmdd'].unique()
    
    # Plot all strains for each drug and concentration
    for _, row in drug_concentrations.iterrows():
        drug = row['drug']
        concentration = row['concentration']
        for date in unique_dates: # Make plots for each unique date
            plt.figure(figsize=(12, 10))
            plt.title(f"Drug: {drug}, Concentration: {concentration}µM, Date: {date}")
            
            # Filter data for the current drug, concentration, and date
            drug_concentration_data = all_data_df[(all_data_df['drug'] == drug) & 
                                                  (all_data_df['concentration'] == concentration) & 
                                                  (all_data_df['date_yyyymmdd'] == date)]
            
            # Drop rows with NaNs
            drug_concentration_data = drug_concentration_data.dropna()
            
            # Skip if no data is left after dropping NaNs
            if drug_concentration_data.empty:
                print(f"No data for drug: {drug}, concentration: {concentration}, date: {date}, skipping...")
                continue

            # Calculate the mean value for each strain
            strain_means = drug_concentration_data.groupby('strain')['value'].mean()
            
            # Calculate the mean value for N2
            n2_mean = strain_means['N2']
            
            # Calculate the difference to the mean of N2
            strain_diffs = strain_means - n2_mean
            
            # Sort strains based on the difference
            sorted_strains = strain_diffs.abs().sort_values().index.tolist()
            
            # Ensure N2 is the first in the order
            order = ['N2'] + [strain for strain in sorted_strains if strain != 'N2']
            
            # Determine the transparency based on concentration
            alpha = alpha_map[concentration]
            
            # Create the boxplot
            boxplot = sns.boxplot(data=drug_concentration_data, x='strain', y='value', order=order, color=get_color(drug), fliersize=0)
            
            # Set the transparency for the boxplot elements
            for patch in boxplot.patches:
                r, g, b, _ = patch.get_facecolor()
                patch.set_facecolor((r, g, b, alpha))
            
            # Create the striplot to show all data points
            striplot = sns.stripplot(data=drug_concentration_data, x='strain', y='value', color='black', jitter=True, marker='o', alpha=0.5, order=order)
            
            # Add labels and title
            plt.xlabel('Strain, Drug and Concentration')
            plt.ylabel('Pixel Variation - not normalised')

            # Change x-axis label orientation
            plt.xticks(rotation=90)

            plt.ylim(0, 0.1)

            # Format the filename
            drug_formatted = drug.lower().replace(' ', '_')
            concentration_formatted = str(concentration).replace('.', '_')
            filename = f'{drug_formatted}_{concentration_formatted}_pixelvariance.png'
            
            # Save the plot with high DPI
            date_dir = SAVE_FIG_DIR / 'all' / 'by_drug' / date
            create_directory(date_dir)
            plt.savefig(date_dir / filename, dpi=300, bbox_inches='tight')

            #plt.show()
    

    
            # Plot all strains for each drug and concentration with all dates combined
        
        # Filter data for the current drug and concentration across all dates
        drug_concentration_data = all_data_df[(all_data_df['drug'] == drug) & 
                                              (all_data_df['concentration'] == concentration)]

        plt.figure(figsize=(20, 10))
        plt.title(f"Drug: {drug}, Concentration: {concentration}µM (All Dates Combined)")
        
        # Calculate the mean value for each strain
        strain_means = drug_concentration_data.groupby('strain')['value'].mean()
        
        # Calculate the mean value for N2
        n2_mean = strain_means['N2']
            
        # Calculate the difference to the mean of N2
        strain_diffs = strain_means - n2_mean
        
        # Sort strains based on the difference
        sorted_strains = strain_diffs.abs().sort_values().index.tolist()
        
        # Ensure N2 is the first in the order
        order = ['N2'] + [strain for strain in sorted_strains if strain != 'N2']
        
        # Determine the transparency based on concentration
        alpha = alpha_map[concentration]
        
        # Create the boxplot
        boxplot = sns.boxplot(data=drug_concentration_data, x='strain', y='value', order=order, color=get_color(drug), fliersize=0)
        
        # Set the transparency for the boxplot elements
        for patch in boxplot.patches:
            r, g, b, _ = patch.get_facecolor()
            patch.set_facecolor((r, g, b, alpha))
        
        # Create the striplot to show all data points
        striplot = sns.stripplot(data=drug_concentration_data, x='strain', y='value', color='black', jitter=True, marker='o', alpha=0.5, order=order)
        
        # Add labels and title
        plt.xlabel('Strain, Drug and Concentration')
        plt.ylabel('Pixel Variation - not normalised')

        # Change x-axis label orientation
        plt.xticks(rotation=90)

        plt.ylim(0, 0.1)

        # Format the filename
        drug_formatted = drug.lower().replace(' ', '_')
        concentration_formatted = str(concentration).replace('.', '_')
        filename = f'{drug_formatted}_{concentration_formatted}_pixelvariance_all_dates.png'
        
        # Save the plot with high DPI
        save_dir = Path("/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/figures/mutant_screen/all/by_drug/all_dates")
        create_directory(save_dir)
        plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')

        #plt.show()


# Normalised values


if by_strain_norm == True:
    strains = df_norm['strain'].unique()
    for strain in strains:
        plt.figure(figsize=(12, 8))
        
        strain_data = df_norm[df_norm['strain'] == strain]
        
        if strain == 'N2':
            # Extract unique dates and create a color palette
            date_palette = create_date_color_palette(strain_data['date_yyyymmdd'])
            
            # Create a combined column for drug and concentration
            strain_data['drug_concentration'] = strain_data['drug'] + ' ' + strain_data['concentration'].astype(str) + 'µM'
            
            # Create a color palette for the drug_concentration column
            unique_drug_concentrations = strain_data['drug_concentration'].unique()
            palette = {dc: get_color(dc) for dc in unique_drug_concentrations}
            
            # Create the boxplot
            boxplot = sns.boxplot(data=strain_data, x='drug_concentration', y='normalized_value', palette=palette)
            
            # Create the striplot to show all data points with colors based on date
            striplot = sns.stripplot(data=strain_data, x='drug_concentration', y='normalized_value', hue='date_yyyymmdd', palette=date_palette, jitter=True, marker='o', alpha=0.8)
            
            # Plot all values for wells in row E (S.H2O) and row F (DMSO)
            #sh2o_values = df_norm[(df_norm['strain'] == strain) & df_norm['well'].str.startswith('E')]['normalized_value']
            #dmso_values = df_norm[(df_norm['strain'] == strain) & df_norm['well'].str.startswith('F')]['normalized_value']
            
            # Add labels and title
            plt.xlabel('Drug and Concentration')
            plt.ylabel('Pixel Variation - normalised')
            plt.title(f"Strain: {strain} - normalised")
            
            # Change x-axis label orientation
            plt.xticks(rotation=90)
            plt.legend(title='Date')

            # Save the plot with high DPI
            #strain = plate_data['strain'].iloc[0]  # Assuming 'strain' column exists
            plt.savefig(SAVE_FIG_DIR / 'all' / 'by_strain' / 'normalised' / f'{strain}_pixelvariance_all_batches.png', dpi=300, bbox_inches='tight')

            #plt.show()  

        else:
                # Extract unique dates
            unique_dates = strain_data['date_yyyymmdd'].unique()
            
            for date in unique_dates:
                plt.figure(figsize=(12, 8))
                
                date_data = strain_data[strain_data['date_yyyymmdd'] == date]
                
                # Create a combined column for drug and concentration
                date_data['drug_concentration'] = date_data['drug'] + ' ' + date_data['concentration'].astype(str) + 'µM'
                
                # Create a color palette for the drug_concentration column
                unique_drug_concentrations = date_data['drug_concentration'].unique()
                palette = {dc: get_color(dc) for dc in unique_drug_concentrations}
                
                # Create the boxplot
                boxplot = sns.boxplot(data=date_data, x='drug_concentration', y='normalized_value', palette=palette)
                
                # Create the striplot to show all data points with colors based on date
                striplot = sns.stripplot(data=date_data, x='drug_concentration', y='normalized_value', hue='date_yyyymmdd', palette={date: 'black'}, jitter=True, marker='o', alpha=0.8)
                
                # Add labels and title
                plt.xlabel('Drug and Concentration')
                plt.ylabel('Pixel Variation - normalised')
                plt.title(f"Strain: {strain} - normalised")
                
                # Change x-axis label orientation
                plt.xticks(rotation=90)
                plt.legend(title='Date')

                # Save the plot with high DPI
                #plt.savefig(SAVE_FIG_DIR / 'all' / 'by_strain' / 'normalised' / f'{date}_{strain}_pixelvariance.png', dpi=300, bbox_inches='tight')

                # Save the plot with high DPI
                date_dir = SAVE_FIG_DIR / 'all' / 'by_strain' / 'normalised' / date
                create_directory(date_dir)
                plt.savefig(date_dir / f'{date}_{strain}_pixelvariance.png', dpi=300, bbox_inches='tight')

                #plt.show()




if by_drug_norm == True:
        # Get unique combinations of drug and concentration
    drug_concentrations = df_norm[['drug', 'concentration']].drop_duplicates()

    # Determine the range of concentrations
    concentration_values = drug_concentrations['concentration'].unique()
    concentration_values_sorted = np.sort(concentration_values)

    # Create a mapping from concentration to alpha value
    alpha_map = {concentration: (i + 1) / len(concentration_values_sorted) for i, concentration in enumerate(concentration_values_sorted)}

    # Extract unique dates to plot by drug and concentration, and by day 
    unique_dates = df_norm['date_yyyymmdd'].unique()
    # Plot day by day
    
    for _, row in drug_concentrations.iterrows():
        drug = row['drug']
        concentration = row['concentration']
        for date in unique_dates:
            plt.figure(figsize=(12, 10))
            plt.title(f"Drug: {drug}, Concentration: {concentration}µM")
            
            # Filter data for the current drug, concentration, and date
            drug_concentration_data = df_norm[(df_norm['drug'] == drug) & 
                                                  (df_norm['concentration'] == concentration) & 
                                                  (df_norm['date_yyyymmdd'] == date)]
            
            # Drop rows with NaNs
            drug_concentration_data = drug_concentration_data.dropna()
            
            # Skip if no data is left after dropping NaNs
            if drug_concentration_data.empty:
                print(f"No data for drug: {drug}, concentration: {concentration}, date: {date}, skipping...")
                continue

            # Calculate the mean value for each strain
            strain_means = drug_concentration_data.groupby('strain')['value'].mean()
            
            # Calculate the mean value for N2
            n2_mean = strain_means['N2']
            
            # Calculate the difference to the mean of N2
            strain_diffs = strain_means - n2_mean
            
            # Sort strains based on the difference
            sorted_strains = strain_diffs.abs().sort_values().index.tolist()
            
            # Ensure N2 is the first in the order
            order = ['N2'] + [strain for strain in sorted_strains if strain != 'N2']
            
            # Determine the transparency based on concentration
            alpha = alpha_map[concentration]
            
            # Create the boxplot
            boxplot = sns.boxplot(data=drug_concentration_data, x='strain', y='normalized_value', order=order, color=get_color(drug), fliersize=0)
            
            # Set the transparency for the boxplot elements
            for patch in boxplot.patches:
                r, g, b, _ = patch.get_facecolor()
                patch.set_facecolor((r, g, b, alpha))
            
            # Create the striplot to show all data points
            striplot = sns.stripplot(data=drug_concentration_data, x='strain', y='normalized_value', color='black', jitter=True, marker='o', alpha=0.5, order=order)
            
            # Add labels and title
            plt.xlabel('Strain, Drug and Concentration')
            plt.ylabel('Pixel Variation -  normalised')

            # Change x-axis label orientation
            plt.xticks(rotation=90)

            #plt.ylim(0, 1.4)

            # Format the filename
            drug_formatted = drug.lower().replace(' ', '_')
            concentration_formatted = str(concentration).replace('.', '_')
            filename = f'{drug_formatted}_{concentration_formatted}_pixelvariance.png'
            
            # Save the plot with high DPI
            #plt.savefig(SAVE_FIG_DIR / 'all' / 'by_drug' / 'normalised' / f'{filename}', dpi=300, bbox_inches='tight')

            # Save the plot with high DPI
            for date in drug_concentration_data['date_yyyymmdd'].unique():
                date_dir = SAVE_FIG_DIR / 'all' / 'by_drug' / 'normalised' / date
                create_directory(date_dir)
                plt.savefig(date_dir / filename, dpi=300, bbox_inches='tight')
            #plt.show()

    # Plot all strains together for each drug and concentration       
        # Filter data for the current drug and concentration
        drug_concentration_data = df_norm[(df_norm['drug'] == drug) & 
                                          (df_norm['concentration'] == concentration)]
        
        plt.figure(figsize=(20, 10))
        plt.title(f"Drug: {drug}, Concentration: {concentration}µM")

        # Calculate the mean value for each strain
        strain_means = drug_concentration_data.groupby('strain')['value'].mean()
        
        # Calculate the mean value for N2
        n2_mean = strain_means['N2']
        
        # Calculate the difference to the mean of N2
        strain_diffs = strain_means - n2_mean
        
        # Sort strains based on the difference
        sorted_strains = strain_diffs.abs().sort_values().index.tolist()
        
        # Ensure N2 is the first in the order
        order = ['N2'] + [strain for strain in sorted_strains if strain != 'N2']
        
        # Determine the transparency based on concentration
        alpha = alpha_map[concentration]
        
        # Create the boxplot
        boxplot = sns.boxplot(data=drug_concentration_data, x='strain', y='normalized_value', order=order, color=get_color(drug), fliersize=0)
        
        # Set the transparency for the boxplot elements
        for patch in boxplot.patches:
            r, g, b, _ = patch.get_facecolor()
            patch.set_facecolor((r, g, b, alpha))
        
        # Create the striplot to show all data points
        striplot = sns.stripplot(data=drug_concentration_data, x='strain', y='normalized_value', color='black', jitter=True, marker='o', alpha=0.5, order=order)
        
        # Add labels and title
        plt.xlabel('Strain, Drug and Concentration')
        plt.ylabel('Pixel Variation - normalised')

        # Change x-axis label orientation
        plt.xticks(rotation=90)

        #plt.ylim(0, 1.4)

        # Format the filename
        drug_formatted = drug.lower().replace(' ', '_')
        concentration_formatted = str(concentration).replace('.', '_')
        filename = f'{drug_formatted}_{concentration_formatted}_pixelvariance.png'
        
        # Save the plot with high DPI
        plt.savefig(SAVE_FIG_DIR / 'all' / 'by_drug' / 'normalised' / 'all_dates' / f'{filename}', dpi=300, bbox_inches='tight')



if heatmap_norm == True:

    unique_dates = df_norm['date_yyyymmdd'].unique()

    def plot_heatmap(data, title, filename, figsize=(12, 20)):
        df_pivot_norm_all = data.copy()

        # Pivot the DataFrame
        df_pivot_norm_all = df_pivot_norm_all.pivot_table(index='strain', columns=['drug', 'concentration'], values='normalized_value')

        # Flatten the multi-level column index for better readability in the heatmap
        df_pivot_norm_all.columns = [f'{drug}-{concentration}' for drug, concentration in df_pivot_norm_all.columns]

        # Define the custom order of the drugs
        custom_order = ['Aldicarb', 'Levamisole', 'Ivermectin', 'Chlorpromazine']

        # Reorder the columns based on the custom order
        ordered_columns = []
        for drug in custom_order:
            ordered_columns.extend([col for col in df_pivot_norm_all.columns if col.startswith(drug)])
        df_pivot_norm_all = df_pivot_norm_all[ordered_columns]

        # Reorder the rows to place 'N2' at the top and strains with missing values at the bottom
        strains_with_missing_values = df_pivot_norm_all.index[df_pivot_norm_all.isnull().any(axis=1)].tolist()
        strains_with_complete_values = df_pivot_norm_all.index.difference(strains_with_missing_values).tolist()
        
        # Ensure 'N2' is at the top
        if 'N2' in strains_with_complete_values:
            strains_with_complete_values.remove('N2')
            ordered_strains = ['N2'] + strains_with_complete_values + strains_with_missing_values
        else:
            ordered_strains = strains_with_complete_values + strains_with_missing_values

        df_pivot_norm_all = df_pivot_norm_all.loc[ordered_strains]

        colors = [(0, 'blue'), (1, 'white')]
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

        # Create a heatmap
        plt.figure(figsize=figsize)
        ax = sns.heatmap(df_pivot_norm_all, cmap=custom_cmap, annot=True, fmt=".2f", linewidths=.5, vmin=0, vmax=1, 
                        cbar=False)

        # Customize the plot
        plt.title(title)
        plt.xlabel('Drugs and Concentrations')
        plt.ylabel('Strains')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Draw vertical lines between different drugs
        drug_names = df_pivot_norm_all.columns
        unique_drugs = df_pivot_norm_all.columns.str.split('-').str[0].unique()
        for drug in unique_drugs[:-1]:  # Skip the last drug
            last_col_index = np.where(df_pivot_norm_all.columns.str.startswith(drug))[0][-1]
            ax.axvline(x=last_col_index + 1, color='black', linewidth=2)

        # Show the plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        #plt.show()

        # Create a color bar for the heatmap
        fig, ax = plt.subplots(figsize=(1, 5))
        cmap = custom_cmap
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('Normalised Pixel Variation\nhigh to low ', rotation=270, labelpad=20)
        cb1.set_ticks([0, 1])
        cb1.set_ticklabels(['0', '1'])

        # Save the color bar
        fig.savefig(filename.parent / f'colorbar_{filename.stem}.png', dpi=300, bbox_inches='tight')
        #plt.show()

    # Plot for all data
    plot_heatmap(df_norm, 'All Strains - Normalised Pixel Variation to Drug Control', SAVE_FIG_DIR / 'all' / 'heatmap' / 'all_days' / 'heatmap_pixelvariance_norm.png')

    # Plot for each unique date
    for date in unique_dates:
        date_data = df_norm[df_norm['date_yyyymmdd'] == date]
        date_dir = SAVE_FIG_DIR / 'all' / 'heatmap' / date
        create_directory(date_dir)
        plot_heatmap(date_data, f'All Strains - Normalised Pixel Variation to Drug Control ({date})', date_dir / f'heatmap_pixelvariance_norm_{date}.png')
    
    # Define the custom order for N2 strains
    n2_order = [
        'N2_2025-03-14',
        'N2_2025-03-21',
        'N2_2025-03-28',
        'N2_2025-04-01'
    ]

    for control in df_norm:
        # Filter data for N2 strain and supplement strain name with date
        n2_data = df_norm[df_norm['strain'] == 'N2'].copy()
        n2_data['strain'] = n2_data['strain'] + '_' + n2_data['date_yyyymmdd'].astype(str)

        # Convert the strain column to a categorical type with the specified order
        n2_data['strain'] = pd.Categorical(n2_data['strain'], categories=n2_order, ordered=True)

        # Sort the N2 data by the categorical strain column
        n2_data = n2_data.sort_values(by='strain')

        # Plot heatmap for N2 strain with supplemented date
        plot_heatmap(n2_data, 'N2 Strain - Normalised Pixel Variation to Drug Control (All Days)', SAVE_FIG_DIR / 'all' / 'heatmap' / 'control' / 'heatmap_pixelvariance_norm_N2_all_days.png', figsize=(12, 4))

plt.close('all')

#%%

# Statistical Analysis 1.
## The goal of this two-way ANOVA analysis is to check for an interaction effect between the worm strain and a drug at a specific concentration. 
## This requires comparing the pixel difference of N2 and the worm strain in the no drug treatment (S.H2O) and N2 and the worm strain in the drug treatment, at a specific concentration.

## The analysis is performed only on N2 from the day of recording that corresponds to the day of recording of the strain being tested

# (1) CHS11246 as a control for the statistical analysis - ensuring that the analysis can determine whether there is a strain:drug_concentration interaction effect or not.
#   This strain is slow-moving in both S. H20 and the drug, meaning there should not be a significant strain:drug_concentration interaction effect, but there should be a strain effect.
#   There are three types of models being applied: 
#       1) Only a strain effect -  tested by comparing N2 and the worm strain in no drug treatment (S.H2O)        
#       2) Only a drug effect   -  tested by comparing N2 and the worm strain in one drug treatment (one single concentration)
#       3) Interaction effect   -  tested by comparing the combination of no drug treatement and drug treatment between N2 and the worm strain


all_data_df_stats = all_data_df.copy()

control_strains = ['N2', 'CHS11246']

# Filter the DataFrame to include only the control strains
n2_strain_stats = all_data_df_stats[all_data_df_stats['strain'].isin(control_strains)]

# Filter out N2 data that is not from the date corresponding to the strain being tested
# Get the unique dates for the other strain (excluding 'N2')
other_strain_dates = n2_strain_stats.loc[
    n2_strain_stats['strain'] != 'N2', 'date_yyyymmdd'
].unique()

# Filter 'N2' rows to only include matching dates (analasying only to the N2 on that day)
n2_strain_stats = n2_strain_stats[
    (n2_strain_stats['strain'].isin(control_strains)) & (n2_strain_stats['date_yyyymmdd'].isin(other_strain_dates))
]

control_wells = ['S.H2O']
aldicarb = ['Aldicarb']
aldicarb_conc = [7.5]

# Filter the DataFrame to include only the control_wells and aldicarb and aldicarb_conc
chs11246_aldi_analysis = n2_strain_stats[(all_data_df_stats['drug'].isin(control_wells)) | 
                                        (all_data_df_stats['drug'].isin(aldicarb)) & 
                                        (all_data_df_stats['concentration'].isin(aldicarb_conc))]


# 1. Control Data Analysis (Strain Effect)
control_data = n2_strain_stats[n2_strain_stats['drug'].isin(control_wells)]
model_control = ols('value ~ strain', data=control_data).fit()
anova_control = sm.stats.anova_lm(model_control, typ=3) # unbalanced ANOVA due to higher number of replictaes in teh S.H2O treatment
print("Control Data - S.H2O (Strain Effect):")
print(anova_control)

# 2. Aldicarb 7.5 µM Analysis (Drug Effect)
aldicarb_data = n2_strain_stats[
    (n2_strain_stats['drug'].isin(aldicarb)) & 
    (n2_strain_stats['concentration'].isin(aldicarb_conc))
]
model_aldicarb = ols('value ~ strain', data=aldicarb_data).fit()
anova_aldicarb = sm.stats.anova_lm(model_aldicarb, typ=3) # unbalanced ANOVA due to higher number of replictaes in teh S.H2O treatment
print("\nAldicarb 7.5 µM (Drug Effect):")
print(anova_aldicarb)

# 3. Combined Analysis (Interaction Effect)
combined_data = chs11246_aldi_analysis
model_combined = ols('value ~ strain * drug', data=combined_data).fit()
anova_combined = sm.stats.anova_lm(model_combined, typ=3) # unbalanced ANOVA due to higher number of replictaes in teh S.H2O treatment
print("\nCombined Analysis (Interaction Effect):")
print(anova_combined)

# Determine which test has the main effect
def get_main_effect_test():
    # Extract p-values for each effect from the ANOVA tables
    control_p = anova_control.loc['strain', 'PR(>F)'] if 'strain' in anova_control.index else float('inf')
    aldicarb_p = anova_aldicarb.loc['strain', 'PR(>F)'] if 'strain' in anova_aldicarb.index else float('inf')
    strain_p = anova_combined.loc['strain', 'PR(>F)'] if 'strain' in anova_combined.index else float('inf')
    drug_p = anova_combined.loc['drug', 'PR(>F)'] if 'drug' in anova_combined.index else float('inf')
    interaction_p = anova_combined.loc['strain:drug', 'PR(>F)'] if 'strain:drug' in anova_combined.index else float('inf')

    # Create a dictionary of p-values for each effect
    p_values = {
        'strain (control)': control_p,
        'strain (aldicarb)': aldicarb_p,
        'strain (combined)': strain_p,
        'drug': drug_p,
        'strain:drug': interaction_p
    }

    # Find the effect with the smallest p-value
    main_effect_test = min(p_values, key=p_values.get)

    # Return the test name and its p-value
    return main_effect_test, p_values[main_effect_test]

# Get the main effect test and its p-value
main_effect_test, main_effect_p = get_main_effect_test()

# Print the significant main effect
print(f"\nSignificant main effect: {main_effect_test} (p = {main_effect_p:.3e})")




# Plot the results of the ANOVA by indicating significance in a boxplot

# Define the order of the hue categories
hue_order = ['S.H2O', 'Aldicarb']

# Plot the results of the ANOVA by indicating significance in a boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='strain', y='value', hue='drug', data=chs11246_aldi_analysis, palette='colorblind', hue_order=hue_order)
plt.xlabel('Strain')
plt.ylabel('Pixel Variation - not normalised')
plt.legend(title='Drug')

# Calculate the maximum y-value across all datasets
max_y_value = max(
    control_data['value'].max(),
    aldicarb_data['value'].max(),
    combined_data['value'].max()
)

# Adjust the y-axis limit dynamically to include space for significance markers
extra_space = 0.01  # Add extra space above the maximum value for significance markers
plt.ylim(0, max_y_value + extra_space + 0.1)

# Function to determine the significance stars
def get_significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


# 1. Add significance for 'anova_control'
for i, row in anova_control.iterrows():
    stars = get_significance_stars(row['PR(>F)'])
    if stars:
        x1, x2 = -0.2, 0.8  # Replace with the indices of the boxplots being compared
        y = max_y_value + 0.02  # Adjust y-position for the bar
        plt.plot([x1, x2], [y, y], color='black', linewidth=1.5)  # Draw the horizontal bar
        plt.plot([x1, x1], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x1
        plt.plot([x2, x2], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x2
        plt.text((x1 + x2) / 2, y + 0.005, stars, ha='center', va='bottom', color='black')  # Add significance text

# 2. Add significance for 'anova_aldicarb'
for i, row in anova_aldicarb.iterrows():
    stars = get_significance_stars(row['PR(>F)'])
    if stars:
        x1, x2 = 0.2, 1.2  # Replace with the indices of the boxplots being compared
        y = max_y_value + 0.045  # Adjust y-position for the bar
        plt.plot([x1, x2], [y, y], color='black', linewidth=1.5)  # Draw the horizontal bar
        plt.plot([x1, x1], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x1
        plt.plot([x2, x2], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x2
        plt.text((x1 + x2) / 2, y + 0.005, stars, ha='center', va='bottom', color='black')  # Add significance text

# 3. Add significance for 'anova_combined'
for i, row in anova_combined.iterrows():
    stars = get_significance_stars(row['PR(>F)'])
    if stars:
        if ':' in row.name:  # Check if the row name contains a colon
            x1, x2 = 0, 2  # Replace with the indices of the boxplots being compared
            y = max_y_value + 0.06  # Adjust y-position for the bar
            plt.plot([x1, x2], [y, y], color='black', linewidth=1.5)  # Draw the horizontal bar
            plt.plot([x1, x1], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x1
            plt.plot([x2, x2], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x2
            plt.text((x1 + x2) / 2, y + 0.01, stars, ha='center', va='bottom', color='black')  # Add significance text

plt.show()

#%%
# Statistical Analysis 2.
## The goal of this two-way ANOVA analysis is to check for an interaction effect between the worm strain and a drug at a specific concentration. 
## This requires comparing the pixel difference of N2 and the worm strain in the no drug treatment (S.H2O) and N2 and the worm strain in the drug treatment, at a specific concentration.


# (2) CHS11227_LKG as a control for the statistical analysis - ensuring that the analysis can determine whether there is a strain:drug_concentration interaction effect or not.
#   This strain looks like N2 in the no drug treatment control wells and a stronger reduction in pixel variance in 10 µM Chlorpromazine, meaning there might be a significant strain:drug_concentration interaction effect. 
#   There are three types of models being applied: 
#       1) Only a strain effect -  tested by comparing N2 and the worm strain in no drug treatment (S.H2O)        
#       2) Only a drug effect   -  tested by comparing N2 and the worm strain in one drug treatment (one single concentration)
#       3) Interaction effect   -  tested by comparing the combination of no drug treatement and drug treatment between N2 and the worm strain


all_data_df_stats = all_data_df.copy()

control_strains = ['N2', 'CHS11227_LKG']

# Filter the DataFrame to include only the control strains
n2_strain_stats = all_data_df_stats[all_data_df_stats['strain'].isin(control_strains)]

# Get the unique dates for the other strain (excluding 'N2')
other_strain_dates = n2_strain_stats.loc[
    n2_strain_stats['strain'] != 'N2', 'date_yyyymmdd'
].unique()

# Filter 'N2' rows to only include matching dates (analasying only to the N2 on that day)
n2_strain_stats = n2_strain_stats[
    (n2_strain_stats['strain'].isin(control_strains)) & (n2_strain_stats['date_yyyymmdd'].isin(other_strain_dates))
]


control_wells = ['S.H2O']
drug = ['Chlorpromazine']
drug_conc = [10]

# Define the order of the hue categories (plotting)
hue_order = ['S.H2O', 'Chlorpromazine']

# Filter the DataFrame to include only the controll_wells and drug and drug_conc
strain_drug_analysis = n2_strain_stats[(all_data_df_stats['drug'].isin(control_wells)) | 
                                        (all_data_df_stats['drug'].isin(drug)) & 
                                        (all_data_df_stats['concentration'].isin(drug_conc))]


# 1. Control Data Analysis (Strain Effect)
control_data = n2_strain_stats[n2_strain_stats['drug'].isin(control_wells)]
model_control = ols('value ~ strain', data=control_data).fit()
anova_control = sm.stats.anova_lm(model_control, typ=3) # unbalanced ANOVA due to higher number of replictaes in teh S.H2O treatment
print("Control Data - S.H2O (Strain Effect):")
print(anova_control)

# 2. Drug at one concentration (µM) Analysis (Drug Effect)
drug_data = n2_strain_stats[
    (n2_strain_stats['drug'].isin(drug)) & 
    (n2_strain_stats['concentration'].isin(drug_conc))
]
model_drug = ols('value ~ strain', data=drug_data).fit()
anova_drug = sm.stats.anova_lm(model_drug, typ=3) # unbalanced ANOVA due to higher number of replictaes in teh S.H2O treatment
print("\Chlorpromazine 10 µM (Drug Effect):")
print(anova_drug)

# 3. Combined Analysis (Interaction Effect)
combined_data = strain_drug_analysis
model_combined = ols('value ~ strain * drug', data=combined_data).fit()
anova_combined = sm.stats.anova_lm(model_combined, typ=3) # unbalanced ANOVA due to higher number of replictaes in teh S.H2O treatment
print("\nCombined Analysis (Interaction Effect):")
print(anova_combined)


# Determine which test has the main effect
def get_main_effect_test():
    # Extract p-values for each effect from the ANOVA tables
    control_p = anova_control.loc['strain', 'PR(>F)'] if 'strain' in anova_control.index else float('inf')
    drug_indep_p = anova_drug.loc['strain', 'PR(>F)'] if 'strain' in anova_drug.index else float('inf')
    strain_p = anova_combined.loc['strain', 'PR(>F)'] if 'strain' in anova_combined.index else float('inf')
    drug_p = anova_combined.loc['drug', 'PR(>F)'] if 'drug' in anova_combined.index else float('inf')
    interaction_p = anova_combined.loc['strain:drug', 'PR(>F)'] if 'strain:drug' in anova_combined.index else float('inf')

    # Create a dictionary of p-values for each effect
    p_values = {
        'strain (control)': control_p,
        'strain (drug)': drug_indep_p,
        'strain (combined)': strain_p,
        'drug': drug_p,
        'strain:drug': interaction_p
    }

    # Find the effect with the smallest p-value
    main_effect_test = min(p_values, key=p_values.get)

    # Check if the strain:drug interaction is significant
    interaction_significance = "significant" if interaction_p < 0.05 else "not significant"

    # Return the test name and its p-value
    return main_effect_test, p_values[main_effect_test], interaction_p, interaction_significance

# Get the main effect test and its p-value
main_effect_test, main_effect_p, interaction_p, interaction_significance = get_main_effect_test()

# Print the significant main effect
print(f"\nSignificant main effect: {main_effect_test} (p = {main_effect_p:.3e})")

# Print whether the strain:drug interaction is significant
print(f"\nThe strain:drug interaction is {interaction_significance} (p = {interaction_p:.3e}).")




# Plot the results of the ANOVA by indicating significance in a boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='strain', y='value', hue='drug', data=strain_drug_analysis, palette='colorblind', hue_order=hue_order)
plt.xlabel('')
plt.title(f"{drug[0]} {drug_conc[0]} µM")
plt.ylabel('Pixel Variation - not normalised')
plt.legend(title='')

# Calculate the maximum y-value across all datasets
max_y_value = max(
    control_data['value'].max(),
    drug_data['value'].max(),
    combined_data['value'].max()
)

# Adjust the y-axis limit dynamically to include space for significance markers
extra_space = 0.01  # Add extra space above the maximum value for significance markers
plt.ylim(0, max_y_value + extra_space + 0.1)

# Function to determine the significance stars
def get_significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


# 1. Add significance for 'anova_control' to plot
for i, row in anova_control.iterrows():
    stars = get_significance_stars(row['PR(>F)'])
    if stars:
        x1, x2 = -0.2, 0.8  # Replace with the indices of the boxplots being compared
        y = max_y_value + 0.02  # Adjust y-position for the bar
        plt.plot([x1, x2], [y, y], color='black', linewidth=1.5)  # Draw the horizontal bar
        plt.plot([x1, x1], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x1
        plt.plot([x2, x2], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x2
        plt.text((x1 + x2) / 2, y + 0.0025, stars, ha='center', va='bottom', color='black')  # Add significance text


# 2. Add significance for 'anova_drug' to plot
for i, row in anova_drug.iterrows():
    stars = get_significance_stars(row['PR(>F)'])
    if stars:
        x1, x2 = 0.2, 1.2  # Replace with the indices of the boxplots being compared
        y = max_y_value + 0.045  # Adjust y-position for the bar
        plt.plot([x1, x2], [y, y], color='black', linewidth=1.5)  # Draw the horizontal bar
        plt.plot([x1, x1], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x1
        plt.plot([x2, x2], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x2
        plt.text((x1 + x2) / 2, y + 0.0025, stars, ha='center', va='bottom', color='black')  # Add significance text


# 3. Add significance for 'anova_combined' to plot
for i, row in anova_combined.iterrows():
    stars = get_significance_stars(row['PR(>F)'])
    if stars:
        if ':' in row.name:  # Check if the row name contains a colon
            x1, x2 = -0.4, 1.4  # Replace with the indices of the boxplots being compared
            y = max_y_value + 0.005  # Adjust y-position for the bar
            plt.plot([x1, x2], [y, y], color='black', linewidth=1.5)  # Draw the horizontal bar
            plt.plot([x1, x1], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x1
            plt.plot([x2, x2], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x2
            plt.text((x1 + x2) / 2, y + 0.0025, stars, ha='center', va='bottom', color='black')  # Add significance text


plt.show()
plt.close()


# %%

# Perform the same two-way ANOVA analysis as done above for every strain and drug_concentration combination in your dataset.
# It will produce a plot only if there is a significant strain:drug_interaction. But better check the plots that it makes as I found it wasn't perfect. 

 # Ensure the directory exists
output_dir = '/Users/jb3623/Documents/250511_Analysis/NHR_drugs/significance/type_III_anova'
os.makedirs(output_dir, exist_ok=True)

all_data_df_stats = all_data_df.copy()

# Remove rows with 'Ivermectin' in the 'drug' column from the dataset because bonkers data...
all_data_df_stats = all_data_df_stats[all_data_df_stats['drug'] != 'Ivermectin']

# merge the 'drug' and 'concentration' columns into a single column
all_data_df_stats['drug_conc'] = all_data_df_stats['drug'] + ' ' + all_data_df_stats['concentration'].astype(str) + ' µM'
# Revert the values for 'S.H2O' and 'DMSO' back to their original names
all_data_df_stats.loc[all_data_df_stats['drug'] == 'S.H2O', 'drug_conc'] = 'S.H2O'
all_data_df_stats.loc[all_data_df_stats['drug'] == 'DMSO', 'drug_conc'] = 'DMSO'


# Define the control strain
control_strain = 'N2'
control_wells = 'S.H2O'
unique_strains = all_data_df_stats['strain'].unique()
uniuqe_strains = [strain for strain in unique_strains if strain != control_strain]
unique_drugs = all_data_df_stats['drug_conc'].unique()

# Function to determine the significance stars
def get_significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


anova_results = []

# Initialize a list to store all ANOVA results for csv export
anova_results_list = []

# Loop through each strain
for strain in unique_strains:
    # Skip the control strain
    if strain == control_strain:
        continue
    # Print the current strain being analyzed
    print(f"\nAnalyzing strain: {strain}")

    # Filter data for the current strain and corresponding N2 rows with the same date to compare only to the N2 of the day of the strain being recorded
    strain_data = all_data_df_stats[all_data_df_stats['strain'] == strain]
    n2_data = all_data_df_stats[
        (all_data_df_stats['strain'] == control_strain) &
        (all_data_df_stats['date_yyyymmdd'].isin(strain_data['date_yyyymmdd']))
    ]

    # Combine the strain data and N2 data
    combined_data = pd.concat([strain_data, n2_data])

    for drug in unique_drugs:
        # Skip the control treatment ('S.H2O') as a drug
        if drug == control_wells:
            continue

        print(f"\nAnalyzing drug: {drug}")

        # Filter the combined data for the current drug
        drug_data = combined_data[combined_data['drug_conc'] == drug]

        control_treatment = combined_data[combined_data['drug_conc'] == control_wells]

        drug_data = pd.concat([drug_data, control_treatment])

        # Perform ANOVA
        model = ols('value ~ strain * drug', data=drug_data).fit()
        anova_results = sm.stats.anova_lm(model, typ=3) # unbalanced ANOVA due to higher number of replictaes in teh S.H2O treatment

        if 'strain:drug' in anova_results.index and anova_results.loc['strain:drug', 'PR(>F)'] < 0.05:
            print(f"\nSignificant interaction effect found for {strain} with {drug}:")
            print(anova_results)

            # Append significant result to the list
            anova_results_list.append({
                'strain': strain,
                'drug': drug,
                'p_value': anova_results.loc['strain:drug', 'PR(>F)'],
                'result': 'Significant strain:drug interaction'
            })

            # Define the order for plotting
            drug_data['strain'] = pd.Categorical(
                drug_data['strain'], categories=[control_strain, strain], ordered=True
            )
            drug_data['drug_conc'] = pd.Categorical(
                drug_data['drug_conc'], categories=[control_wells, drug], ordered=True
            )

            # Generate the plot
            plt.figure(figsize=(12, 8))
            sns.boxplot(
                x='strain', y='value', hue='drug_conc', 
                data=drug_data, palette='colorblind'
            )
            plt.title(f"Significant Interaction: {strain} with {drug}")
            plt.xlabel('Strain')
            plt.ylabel('Value')
            plt.legend(title='Drug Concentration')

            # Calculate the maximum y-value across all datasets
            max_y_value = drug_data['value'].max()

            # Adjust the y-axis limit dynamically to include space for significance markers
            extra_space = 0.01  # Add extra space above the maximum value for significance markers
            plt.ylim(0, max_y_value + extra_space + 0.1)

            # Add significance markers
            for i, row in anova_results.iterrows():
                stars = get_significance_stars(row['PR(>F)'])
                if stars:
                    if ':' in row.name:  # Check if the row name contains a colon
                        x1, x2 = -0.4, 1.4  # Replace with the indices of the boxplots being compared
                        y = max_y_value + 0.005  # Adjust y-position for the bar
                        plt.plot([x1, x2], [y, y], color='black', linewidth=1.5)  # Draw the horizontal bar
                        plt.plot([x1, x1], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x1
                        plt.plot([x2, x2], [y, y - 0.01], color='black', linewidth=1.5)  # Add vertical overhang at x2
                        plt.text((x1 + x2) / 2, y + 0.0025, stars, ha='center', va='bottom', color='black')  # Add significance text

            # Save the plot
            plot_filename = f"{strain}_{drug}_interaction_plot.png".replace('/', '_')
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()  # Close the plot to free memory

        else:
            print(f"\nNo significant interaction effect found for {strain} with {drug}.")

            # Append non-significant result to the list
            anova_results_list.append({
                'strain': strain,
                'drug': drug,
                'p_value': anova_results.loc['strain:drug', 'PR(>F)'] if 'strain:drug' in anova_results.index else None,
                'result': 'No significant strain:drug interaction'
            })

# Convert the list of results to a DataFrame
anova_results_df = pd.DataFrame(anova_results_list)

# Save the DataFrame to a CSV file
output_csv_path = os.path.join(output_dir, 'anova_results.csv')
anova_results_df.to_csv(output_csv_path, index=False)

print(f"\nANOVA results saved to {output_csv_path}")

plt.close()
# %%
