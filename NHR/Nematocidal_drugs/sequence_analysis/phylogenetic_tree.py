'''
John Bergqvist 2025 (C)

Script to visualise phylogenetic trees.

The Newick string here is from EMBL where all the Uniprot NHR sequences were queried (including duplicates) 

'''

#%%
# Import libraries
from ete3 import Tree, TreeStyle, NodeStyle
import pandas as pd
import re

#%%
# Load the phylogenetic tree .txt file (made from EMBL Clsutal Omega)
tree_file = '/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/sequence_analysis/all_c_elegans_longest_unique_nhr_tree.txt'


# Read the tree from the file
with open(tree_file, 'r') as file:
    newick_string = file.read().strip()

# Remove everything after the '_' in the gene names
modified_newick_string = re.sub(r'_(\w+):', r':', newick_string)

# Create a Tree object from the modified Newick string
t = Tree(modified_newick_string, format=1)

# Optional: Style
ts = TreeStyle()
ts.show_leaf_name = True
ts.scale = 100  # Adjust scale as needed


# Show the tree
t.show(tree_style=ts)


# %%
# Save the tree to a png
t.render("/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/figures/mutant_screen/sequence_analysis/all_nhrs_tree.png", w=600)



# %%
# Extract the names of the NHRs involved in the significant reduction of 
# pixel difference when exposed to nematocidal drugs

# Load the CSV with significant NHR genes
df = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/sequence_analysis/type_III_twoway_anova_sig_nhrs.csv')

# Extract and normalize significant gene names with their associated stat_sig_type
sig_nhr_dict = df.dropna().set_index('gene')['stat_sig_type'].str.strip().str.lower().to_dict()

# Create NodeStyles for each stat_sig_type
styles = {
    'mutant sensitive': NodeStyle(),
    'mutant resistant': NodeStyle(),
    'mutant gain': NodeStyle()
}

styles['mutant sensitive']["fgcolor"] = "red"
styles['mutant sensitive']["size"] = 12

styles['mutant resistant']["fgcolor"] = "orange"
styles['mutant resistant']["size"] = 12

styles['mutant gain']["fgcolor"] = "limegreen"
styles['mutant gain']["size"] = 12

def style_sig_nhr_nodes(node):
    """Style tree nodes based on their associated stat_sig_type."""
    # Normalize node name for matching
    node_base_name = node.name.split('_')[0].strip().lower()
    if node_base_name in sig_nhr_dict:
        stat_sig_type = sig_nhr_dict[node_base_name]
        if stat_sig_type in styles:
            node.set_style(styles[stat_sig_type])

# Apply style to all nodes
for node in t.traverse():
    style_sig_nhr_nodes(node)

# Show the styled tree
t.show(tree_style=ts)

# %%
#save tree
t.render("/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/figures/mutant_screen/sequence_analysis/all_longest_unique_nhrs_tree.pdf", w=1200)
# %%

# %%
