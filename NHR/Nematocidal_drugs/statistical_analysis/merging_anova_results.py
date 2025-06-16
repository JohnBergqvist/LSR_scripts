#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# Merge the raw ANOVA results with the annotated ANOVA, as well as with the full_genotype metadata file to make a table for the LSR

raw_anova = pd.read_csv('/Users/jb3623/Documents/250511_Analysis/NHR_drugs/significance/type_III_anova/anova_results.csv', encoding='latin1')
annotated_anova = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/statistical_analyses/all/twoway_anova/type_III_anova_result/type_III_anova_result.csv')


# Path to the genotype metadata file
file_path = "/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Auxiliary_files/plate_to_strains_all_runs.xlsx"
# Read the first sheet of the Excel file
data = pd.read_excel(file_path, sheet_name=0)
# Extract the 'strain' and 'full_genotype' columns
full_genotype_data = data[['strain', 'full_genotype']]
# Display the extracted data
print(f"number of rows in {full_genotype_data} before removing nans and duplicates: {len(full_genotype_data)}")


# Remove Nans from the full_genotype_data
full_genotype_data = full_genotype_data.dropna(subset=['full_genotype'])

print(f"number of rows in {full_genotype_data} after removing nans: {len(full_genotype_data)}")

# Keep only unique rows in the full_genotype_data
full_genotype_data = full_genotype_data.drop_duplicates(subset=['strain', 'full_genotype'])
print(f"number of rows in {full_genotype_data} after removing nans and duplicates: {len(full_genotype_data)}")

#%%

print(f"Number of line in raw anova before removing insignificant rows", len(raw_anova))
# Drop insignificant rows from the raw anova results ('result': 'No significant strain:drug interaction')
raw_anova = raw_anova[raw_anova['result'] != 'No significant strain:drug interaction']
print(f"Number of line in raw anova after removing insignificant rows", len(raw_anova))

#%%
# Change the annotated anova drug column to match the raw anova drug column by merging the 'drug' and the 'concentration' column and adding a ' µM' at the end (name of new column: 'drug')
annotated_anova['drug'] = annotated_anova['drug'] + ' ' + annotated_anova['concentration'].astype(str) + ' µM'
# Drop the 'concentration' column as it is no longer needed
annotated_anova.drop(columns=['concentration'], inplace=True)
# change the drug column to match the raw anova drug column by removing the ' µM' at the end of the 'DMSO' drug
annotated_anova['drug'] = annotated_anova['drug'].str.replace('DMSO 0.0 µM', 'DMSO', regex=False)


print("Number of lines in annotated anova:", len(annotated_anova))
#%%

# Merge the dataframes on the 'strain' column
merged_data = pd.merge(raw_anova, annotated_anova, on=['strain', 'drug'], how='left')

print(f"number of lines before merging with annotated ANOVA:", len(merged_data))

#%%
# Merge with the annotated_anova results
merged_data = pd.merge(merged_data, full_genotype_data, on='strain', how='left')
print(f"number of lines after merging with annotated ANOVA:", len(merged_data))

# %%

# remove strange symbol in the drug column
merged_data['drug'] = merged_data['drug'].str.replace('¬', '', regex=False)


# save the merged data to a csv file
merged_data.to_csv('/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/statistical_analyses/all/twoway_anova/type_III_anova_result/genotype_anova_results.csv', index=False)



# %%
# Filter for strains not in the list of strains that we already know that we want to check: 

# Make a list of the strains we already want to study
decided_strains = [
    "CHS10799_GUM", "CHS10771_UIN", "CHS10807_RLC", "CHS10526_QJJ", "CHS10796_NIL",
    "CHS11227_LKG", "CHS10542_SJA", "CHS10776_LDH", "CHS11137_QOW", "CHS10806_QLX",
    "CHS10938_GKE", "CHS10781_IJV", "CHS10973_HYV", "CHS10979_LIP", "CHS10798_YAC",
    "CHS10810_AHR", "CHS10822_YPQ", "CHS10941_EDU", "CHS10940_TMC", "CHS10820_BUV",
    "CHS10982_AEU", "CHS10981_CUC", "CHS10825_TXS", "CHS10883_ZWT", "CHS10778_HBD",
    "CHS11268", "CHS11136_XKP", "CHS10808_XUZ", "CHS11224_JTR", "CHS11243"
]
# Convert the list to a set for faster lookup
decided_strains = set(decided_strains)

# %%

# Remove the strains with the associated 'control difference' value in 'stat_sig_type' column in the 'merged_data' df
merged_data = merged_data[merged_data['stat_sig_type'] != 'control difference']

#%%
# Filter the merged data to keep only strains not in the strains_to_check list
filtered_data = merged_data[~merged_data['strain'].isin(decided_strains)]
# Save the filtered data to a csv file
filtered_data.to_csv('/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/statistical_analyses/all/twoway_anova/type_III_anova_result/genotype_anova_results_filtered_strains_to_check.csv', index=False)
# %%
