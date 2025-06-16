#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Checking the number of strains that are more sensisitive to more than one drug.

# Load the data

df = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/statistical_analyses/all/twoway_anova/type_III_anova_result/type_III_anova_result.csv')

# Filter the df to include on the rows with 'mutant sensitive' in the 'stat_sig-type' column
df_filtered = df[df['stat_sig_type'] == 'mutant sensitive']

# Count the number of unique strains that are sensitive to more than one drug
s = df_filtered.groupby('strain')['drug'].nunique()
# Get the strains that are sensitive to more than one drug
sensitive_strains = s[s > 1].index
# Get the number of sensitive strains
num_sensitive_strains = len(sensitive_strains)
# Print the number of sensitive strains
print(f'Number of strains sensitive to more than one drug: {num_sensitive_strains}')
# Plot the number of sensitive strains
plt.figure(figsize=(10, 6))
plt.bar(sensitive_strains, s[sensitive_strains], color='blue')
plt.xlabel('Strain')
plt.ylabel('Number of drugs')
plt.title('Number of drugs each strain is sensitive to')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %%
# Filter the dataframe to include only rows with 'mutant resistant' in the 'stat_sig_type' column
df_filtered = df[df['stat_sig_type'] == 'mutant resistant']

# Count the number of drugs each strain is resistant to
resistant_counts = df_filtered.groupby('strain')['drug'].count()

# Plot all strains that are resistant to a drug
plt.figure(figsize=(10, 6))
plt.bar(resistant_counts.index, resistant_counts.values, color='red')
plt.xlabel('Strain')
plt.ylabel('Number of drugs')
plt.title('Number of drugs each strain is resistant to')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# %%
