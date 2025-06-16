
# -*- coding: utf-8 -*-
'''
This script plots the pixel variance values for all runs of the mobility assay.
It uses the seaborn library to create a FacetGrid of line plots, where each plot represents a different label (strain) and shows the pixel variance over different days.


author: John Bergqvist
date: 2025-06-16
'''

#%%
#import libraries
import os
import numpy as np
import cv2
import pandas as pd
import seaborn as sns
from pathlib import Path
from skimage import filters, measure
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.draw import disk
from natsort import natsorted
from collections import defaultdict
from IPython.display import display



import statsmodels.api as sm
import statsmodels.formula.api as smf

#%%

## Plotting the pixel variance values for everything

#Load the combined_df
combined_df = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/data/final_merged_pixel_difference_results.csv')

pixvar_plotting = combined_df.copy()

# Ensure the 'day' column is ordered as a categorical variable
pixvar_plotting['day'] = pd.Categorical(
    pixvar_plotting['day'], 
    categories=["D0", "D3", "D4", "D5", "D6", "D7"], 
    ordered=True
)

# If plotting by runs for a specific label:
pixvar_plotting['run'] = pd.Categorical(
    pixvar_plotting['run'], 
    categories=["run1", "run2", "run3", "run4"], 
    ordered=True
)

# Remove the well with no plasmid (TWIST error) (mol6481)
pixvar_plotting = pixvar_plotting[pixvar_plotting['label_for_plotting'] != 'mol6481']

# Decide which runs to plot
#pixvar_plotting = pixvar_plotting[pixvar_plotting['run'] == 'run3']

# Decide which labels to plot
#pixvar_plotting = pixvar_plotting[pixvar_plotting['label_for_plotting'].isin(['empty_vector_pet24_control'])] #empty_vector_pet24_control #cry6A #no_bacteria_control #mScarlet_IPTG_control
#
## Create a new column for a combination of 'run' and 'source_plate'
#pixvar_plotting['run_source'] = pixvar_plotting['run'].astype(str) + "_" + pixvar_plotting['source_plate'].astype(str)
#pixvar_plotting['run_source'] = pd.Categorical(
#    pixvar_plotting['run_source'], 
#    categories=["run1_cult_1", "run1_cult_2", "run2_cult_1", "run2_cult_2", "run3_cult_1", "run3_cult_2", "run4_cult_1", "run4_cult_2"], 
#    ordered=True
#)

# Filter for day == 'D7' and compute mean pixvar 'value' for each label
d7_means = (
    pixvar_plotting[pixvar_plotting["day"] == "D7"]
    .groupby("label_for_plotting")["value"]
    .mean()
)

# Sort labels by the mean pixvar 'value' at D7 in ascending order
sorted_labels = d7_means.sort_values(ascending=True).index.tolist()

# Specify the labels to place at the start
priority_labels = ['cry6A', 'no_bacteria_control', 'mScarlet_IPTG_control', 'empty_vector_pet24_control']

# Remove priority labels from the sorted list (if they exist)
remaining_labels = [label for label in sorted_labels if label not in priority_labels]

# Combine priority labels with the remaining sorted labels
final_sorted_labels = priority_labels + remaining_labels

# Create the FacetGrid with the updated structure
g = sns.FacetGrid(
    pixvar_plotting,
    col="label_for_plotting",
    col_wrap=8,
    height=4,
    sharey=True,  # Share the y-axis across plots
    col_order=final_sorted_labels,  # You can specify a sorted order if needed
)
g.map_dataframe(sns.lineplot, x="day", y="value")
g.set_titles("{col_name}")
g.set_axis_labels("Day", "Pixel Variance")
g.add_legend()
plt.tight_layout()
plt.show()

#%%
# save the figure
fig = g.fig
fig.savefig("/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/Analysis/figures/pixvar/pixvar_all_runs.png", dpi=900, bbox_inches='tight')


# %%

# Filtering to plot specific runs

pixvar_plotting = combined_df.copy()
# Filter for run4
pixvar_plotting = pixvar_plotting[pixvar_plotting['run'] == 'run4']

# Remove the well with no plasmid (TWIST error) (mol6481)
pixvar_plotting = pixvar_plotting[pixvar_plotting['label_for_plotting'] != 'mol6481']

# Ensure the 'day' column is ordered as a categorical variable
pixvar_plotting['day'] = pd.Categorical(
    pixvar_plotting['day'], 
    categories=["D0", "D3", "D4", "D5", "D6", "D7"], 
    ordered=True
)

# Filter for day == 'D4' and compute mean pixvar 'value' for each label
d7_means = (
    pixvar_plotting[pixvar_plotting["day"] == "D7"]
    .groupby("label_for_plotting")["value"]
    .mean()
)

# Sort labels by the mean pixvar 'value' at D7 in ascending order
sorted_labels = d7_means.sort_values(ascending=True).index.tolist()

# Specify the labels to place at the start
priority_labels = ['cry6A', 'no_bacteria_control', 'mScarlet_IPTG_control', 'empty_vector_pet24_control']

# Remove priority labels from the sorted list (if they exist)
remaining_labels = [label for label in sorted_labels if label not in priority_labels]

# Combine priority labels with the remaining sorted labels
final_sorted_labels = priority_labels + remaining_labels

# Ensure final_sorted_labels only includes labels present in the filtered data
final_sorted_labels = [label for label in final_sorted_labels if label in pixvar_plotting['label_for_plotting'].unique()]

# Create the FacetGrid with the updated structure
g = sns.FacetGrid(
    pixvar_plotting,
    col="label_for_plotting",
    col_wrap=10,
    height=4,
    sharey=True,  # Share the y-axis across plots
    col_order=final_sorted_labels,  # Use the filtered and sorted labels
)
g.map_dataframe(sns.lineplot, x="day", y="value")
g.set_titles("{col_name}")
g.set_axis_labels("Day", "Pixel Variance")
g.add_legend()
plt.tight_layout()
plt.show()
# %%
fig = g.fig
fig.savefig("/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/Analysis/figures/pixvar/pixvar_run4.png", dpi=900, bbox_inches='tight')

#%%

# Plot the U-bottom runs (Run 1 and Run 2)
pixvar_plotting = combined_df.copy()
# Filter for runs 1 and 2
pixvar_plotting = pixvar_plotting[pixvar_plotting['run'].isin(['run1', 'run2'])]

# Remove the well with no plasmid (TWIST error) (mol6481)
pixvar_plotting = pixvar_plotting[pixvar_plotting['label_for_plotting'] != 'mol6481']

# Ensure the 'day' column is ordered as a categorical variable
pixvar_plotting['day'] = pd.Categorical(
    pixvar_plotting['day'], 
    categories=["D0", "D3", "D4", "D5", "D6", "D7"], 
    ordered=True
)

# Filter for day == 'D4' and compute mean pixvar 'value' for each label
d7_means = (
    pixvar_plotting[pixvar_plotting["day"] == "D7"]
    .groupby("label_for_plotting")["value"]
    .mean()
)

# Sort labels by the mean pixvar 'value' at D7 in ascending order
sorted_labels = d7_means.sort_values(ascending=True).index.tolist()

# Specify the labels to place at the start
priority_labels = ['cry6A', 'no_bacteria_control', 'mScarlet_IPTG_control', 'empty_vector_pet24_control']

# Remove priority labels from the sorted list (if they exist)
remaining_labels = [label for label in sorted_labels if label not in priority_labels]

# Combine priority labels with the remaining sorted labels
final_sorted_labels = priority_labels + remaining_labels

# Ensure final_sorted_labels only includes labels present in the filtered data
final_sorted_labels = [label for label in final_sorted_labels if label in pixvar_plotting['label_for_plotting'].unique()]

# Create the FacetGrid with the updated structure
g = sns.FacetGrid(
    pixvar_plotting,
    col="label_for_plotting",
    col_wrap=10,
    height=4,
    sharey=True,  # Share the y-axis across plots
    col_order=final_sorted_labels,  # Use the filtered and sorted labels
)
g.map_dataframe(sns.lineplot, x="day", y="value")
g.set_titles("{col_name}")
g.set_axis_labels("Day", "Pixel Variance")
g.add_legend()
plt.tight_layout()



# Save the figure
fig = g.fig
fig.savefig("/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/Analysis/figures/pixvar/pixvar_run1_and_run2.pdf", dpi=1200, bbox_inches='tight')

plt.show()

#%%

# Plot the flat-bottom runs (Run 3 and Run 4)
pixvar_plotting = combined_df.copy()
# Filter for runs 1 and 2
pixvar_plotting = pixvar_plotting[pixvar_plotting['run'].isin(['run3', 'run4'])]

# Remove the well with no plasmid (TWIST error) (mol6481)
pixvar_plotting = pixvar_plotting[pixvar_plotting['label_for_plotting'] != 'mol6481']

# Ensure the 'day' column is ordered as a categorical variable
pixvar_plotting['day'] = pd.Categorical(
    pixvar_plotting['day'], 
    categories=["D0", "D3", "D4", "D5", "D6", "D7"], 
    ordered=True
)

# Filter for day == 'D4' and compute mean pixvar 'value' for each label
d7_means = (
    pixvar_plotting[pixvar_plotting["day"] == "D7"]
    .groupby("label_for_plotting")["value"]
    .mean()
)

# Sort labels by the mean pixvar 'value' at D7 in ascending order
sorted_labels = d7_means.sort_values(ascending=True).index.tolist()

# Specify the labels to place at the start
priority_labels = ['cry6A', 'no_bacteria_control', 'mScarlet_IPTG_control', 'empty_vector_pet24_control']

# Remove priority labels from the sorted list (if they exist)
remaining_labels = [label for label in sorted_labels if label not in priority_labels]

# Combine priority labels with the remaining sorted labels
final_sorted_labels = priority_labels + remaining_labels

# Ensure final_sorted_labels only includes labels present in the filtered data
final_sorted_labels = [label for label in final_sorted_labels if label in pixvar_plotting['label_for_plotting'].unique()]

# Create the FacetGrid with the updated structure
g = sns.FacetGrid(
    pixvar_plotting,
    col="label_for_plotting",
    col_wrap=10,
    height=4,
    sharey=True,  # Share the y-axis across plots
    col_order=final_sorted_labels,  # Use the filtered and sorted labels
)
g.map_dataframe(sns.lineplot, x="day", y="value")
g.set_titles("{col_name}")
g.set_axis_labels("Day", "Pixel Variance")
g.add_legend()
plt.tight_layout()



# Save the figure
fig = g.fig
fig.savefig("/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/experiments/MoilityAssay/Analysis/figures/pixvar/pixvar_run3_and_run4.pdf", dpi=1200, bbox_inches='tight')

plt.show()

# %%
pixvar_plotting = combined_df.copy()
# Filter for run4
pixvar_plotting = pixvar_plotting[pixvar_plotting['run'] == 'run4']

# Remove the well with no plasmid (TWIST error) (mol6481)
pixvar_plotting = pixvar_plotting[pixvar_plotting['label_for_plotting'] != 'mol6481']

# Ensure the 'day' column is ordered as a categorical variable
pixvar_plotting['day'] = pd.Categorical(
    pixvar_plotting['day'], 
    categories=["D0", "D3", "D4", "D5", "D6", "D7"], 
    ordered=True
)

# Filter for day == 'D4' and compute mean pixvar 'value' for each label
d7_means = (
    pixvar_plotting[pixvar_plotting["day"] == "D6"]
    .groupby("label_for_plotting")["value"]
    .mean()
)

# Sort labels by the mean pixvar 'value' at D7 in ascending order
sorted_labels = d7_means.sort_values(ascending=False).index.tolist()

# Specify the labels to place at the start
priority_labels = ['cry6A', 'no_bacteria_control', 'mScarlet_IPTG_control', 'empty_vector_pet24_control']

# Remove priority labels from the sorted list (if they exist)
remaining_labels = [label for label in sorted_labels if label not in priority_labels]

# Combine priority labels with the remaining sorted labels
final_sorted_labels = priority_labels + remaining_labels

# Ensure final_sorted_labels only includes labels present in the filtered data
final_sorted_labels = [label for label in final_sorted_labels if label in pixvar_plotting['label_for_plotting'].unique()]

import plotly.express as px

# Create an interactive line plot
fig = px.line(
    pixvar_plotting,
    x="day",
    y="value",
    color="label_for_plotting",  # Different colors for each strain
    hover_name="label_for_plotting",  # Display strain name on hover
    title="Pixel Variance Across Strains Over Days",
    labels={"day": "Day", "value": "Pixel Variance"}
)

# Customize layout
fig.update_layout(
    xaxis_title="Day",
    yaxis_title="Pixel Variance",
    legend_title="Strains",
    template="plotly_white"
)

# Show the interactive plot
fig.show(renderer="browser")


# %%
