'''
(C) John Bergqvist 2025

Script to plot the number of wells with a bacterial lawn visibly present from the soil bacteria, ordered according to the top-bottom clustering of 
Tierpsy 256 features during bluelight stimulation

'''

#%%
# Import libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/bacteria_presence/bacteria_presence.csv')

# Remove rows with 'FALSE' values in 'keep' column
df = df[df['keep'].astype(bool) == True]
#df = df[df['keep'] == 'TRUE']
#print(df['keep'].unique())
#%%
counts = df.groupby(['bacteria_strain', 'bacteria_visibly_present']).size().unstack(fill_value=0)

strain_order = ['B35', 'B28', 'B44', 'JUb134', 'B92', 'B69', 'B90', 'B45', 'B64', 'B21', 'B30', 'B88', 'B82', 'B95', 'B91', 'B87', 'B96', 'OP50']

# Plot by 'bacteria_strain' as a barplot where 'TRUE' is counted in one of the barplots for the strain and 'FALSE' in the other
# Set the style to 'whitegrid'
sns.set(style="whitegrid")

counts = counts.reindex(strain_order, axis=0)  # Reorder the index to match strain_order

# Plot the barplot
ax = counts.plot(kind='bar', figsize=(14, 8), stacked=False)
plt.xlabel('')
plt.ylabel('Number of wells', fontsize=16)

plt.yticks(range(0, int(counts.values.max()) + 4, 2))  # Set custom y-axis ticks
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14)
plt.title('Visible Presence of Bacterial Lawns', fontsize=20)
plt.legend(fontsize=14)
plt.tight_layout()

## Add text underneath each x-tick
#for i, strain in enumerate(strain_order):
#    total_wells = counts.loc[strain].sum() if strain in counts.index else 0
#    ax.text(i, -1.5, 
#            f'(N={total_wells})', 
#            ha='center', 
#            fontsize=10)  # Position text slightly below the x-ticks

# Save the plot
plt.savefig('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data/Analysis/correct/Figures/lawn_presence/bacterial_lawn_presence_cluster_order.pdf', bbox_inches='tight', dpi=600)

plt.show()


# %%
