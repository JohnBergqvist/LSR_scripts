# -*- coding: utf-8 -*-
#%%

# %%
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = '/Volumes/behavgenom$/John/data_exp_info/NHR/Survival/Summary/20240829-20241008/nhr_survival_0829-1008_compiled_censor_corrected.csv'
df = pd.read_csv(file_path)

# Filter the data for the bacterium 'jub134'
#df_jub134 = df[df['bacteria'] == 'JUb134']
#df_one_strain = df_jub134.copy()

# Or filter the data for the bacterium 'OP50'
df_op50 = df[df['bacteria'] == 'OP50']
df_one_strain = df_op50.copy()

# Dictionary to map original worm strain names to new names
worm_name_mapping = {
    "N2": "N2",
    "CHS10968_VWU": "nhr-66",
    "CHS10507_JWJ": "nhr-49;nhr-80",
    "CHS11246": "nhr-49"
}

# Apply the worm name mapping
df_one_strain['worm'] = df_one_strain['worm'].map(worm_name_mapping)

# Function to sort dataframe by 'time' and 'status'
def sort_by_time_and_status(df):
    return df.sort_values(by=['time', 'status'], ascending=[True, False])

# Create separate dataframes for each worm strain, sorted by time and status
df_n2 = sort_by_time_and_status(df_one_strain[df_one_strain['worm'] == 'N2'])
df_nhr66 = sort_by_time_and_status(df_one_strain[df_one_strain['worm'] == 'nhr-66'])
df_nhr49_nhr80 = sort_by_time_and_status(df_one_strain[df_one_strain['worm'] == 'nhr-49;nhr-80'])
df_nhr49 = sort_by_time_and_status(df_one_strain[df_one_strain['worm'] == 'nhr-49'])

# Function to print the number of rows with status '1' for each timepoint
def print_dead_counts(df, worm_name):
    dead_counts = df[df['status'] == 1].groupby('time').size()
    print(f"\n{worm_name} DataFrame:")
    print(dead_counts)

# Function to print the number of rows with status '0' for each timepoint
def print_censored_counts(df, worm_name):
    censored_counts = df[df['status'] == 0].groupby('time').size()
    print(f"\n{worm_name} DataFrame:")
    print(censored_counts)

# Function to print the number of worms at risk at each timepoint
def print_at_risk_counts(df, worm_name):
    at_risk_counts = df.groupby('time').size().cumsum()
    print(f"\n{worm_name} DataFrame:")
    print(at_risk_counts)

# Print the number of rows with status '1' for each timepoint in the four different dataframes
print_dead_counts(df_n2, 'N2')
print_dead_counts(df_nhr66, 'nhr-66')
print_dead_counts(df_nhr49_nhr80, 'nhr-49;nhr-80')
print_dead_counts(df_nhr49, 'nhr-49')

# Print the number of rows with status '0' for each timepoint in the four different dataframes
print_censored_counts(df_n2, 'N2')
print_censored_counts(df_nhr66, 'nhr-66')
print_censored_counts(df_nhr49_nhr80, 'nhr-49;nhr-80')
print_censored_counts(df_nhr49, 'nhr-49')

# Print the number of worms at risk at each timepoint in the four different dataframes
print_at_risk_counts(df_n2, 'N2')
print_at_risk_counts(df_nhr66, 'nhr-66')
print_at_risk_counts(df_nhr49_nhr80, 'nhr-49;nhr-80')
print_at_risk_counts(df_nhr49, 'nhr-49')


# %%
print(f'nhr49_nhr80 - 1:', df_nhr49_nhr80[df_nhr49_nhr80['status'] == 1].groupby('time').size())
print(f'nhr49_nhr80 - 0:', df_nhr49_nhr80[df_nhr49_nhr80['status'] == 0].groupby('time').size())
print(f'nhr49_nhr80 - total:', df_nhr49_nhr80.groupby('time').size())

print(f'nhr49 - 1:', df_nhr49[df_nhr49['status'] == 1].groupby('time').size())
print(f'nhr49 - 0:', df_nhr49[df_nhr49['status'] == 0].groupby('time').size())
print(f'nhr49 - total:', df_nhr49.groupby('time').size())

print(f'nhr66 - 1:', df_nhr66[df_nhr66['status'] == 1].groupby('time').size())
print(f'nhr66 - 0:', df_nhr66[df_nhr66['status'] == 0].groupby('time').size())
print(f'nhr66 - total:', df_nhr66.groupby('time').size())

print(f'n2 - 1:', df_n2[df_n2['status'] == 1].groupby('time').size())
print(f'n2 - 0:', df_n2[df_n2['status'] == 0].groupby('time').size())
print(f'n2 - total:', df_n2.groupby('time').size())

#%%
# Statistical analyis - mutant vs N2


# Fit Kaplan-Meier estimators for each worm strain
kmf_n2 = KaplanMeierFitter()
kmf_n2.fit(durations=df_n2['time'], event_observed=df_n2['status'], label='N2')

kmf_nhr66 = KaplanMeierFitter()
kmf_nhr66.fit(durations=df_nhr66['time'], event_observed=df_nhr66['status'], label='nhr-66')

kmf_nhr49 = KaplanMeierFitter()
kmf_nhr49.fit(durations=df_nhr49['time'], event_observed=df_nhr49['status'], label='nhr-49')

kmf_nhr49_nhr80 = KaplanMeierFitter()
kmf_nhr49_nhr80.fit(durations=df_nhr49_nhr80['time'], event_observed=df_nhr49_nhr80['status'], label='nhr-49;nhr-80')

# Perform Log-Rank Test between N2 and each other strain
results_n2_vs_nhr66 = logrank_test(df_n2['time'], df_nhr66['time'], event_observed_A=df_n2['status'], event_observed_B=df_nhr66['status'])
results_n2_vs_nhr49 = logrank_test(df_n2['time'], df_nhr49['time'], event_observed_A=df_n2['status'], event_observed_B=df_nhr49['status'])
results_n2_vs_nhr49_nhr80 = logrank_test(df_n2['time'], df_nhr49_nhr80['time'], event_observed_A=df_n2['status'], event_observed_B=df_nhr49_nhr80['status'])

# Print the p-values to determine statistical significance
print(f'Log-Rank Test p-value (N2 vs nhr-66): {results_n2_vs_nhr66.p_value}')
print(f'Log-Rank Test p-value (N2 vs nhr-49): {results_n2_vs_nhr49.p_value}')
print(f'Log-Rank Test p-value (N2 vs nhr-49;nhr-80): {results_n2_vs_nhr49_nhr80.p_value}')


# %%
# Function to calculate the number of worms at risk at each timepoint
def calculate_at_risk(df, initial_count):
    at_risk_counts = []
    for time in sorted(df['time'].unique()):
        dead = df[df['status'] == 1].groupby('time').size().get(time, 0)
        censored = df[df['status'] == 0].groupby('time').size().get(time, 0)
        at_risk = initial_count - dead
        at_risk_counts.append((time, at_risk, dead, censored))
    return at_risk_counts


# Function to print the number of worms at risk at each timepoint
def print_at_risk_table(at_risk_counts, worm_name):
    print(f"\n{worm_name} DataFrame:")
    print("Time\tAt Risk\tDead\tCensored")
    for time, at_risk, dead, censored in at_risk_counts:
        print(f"{time}\t{at_risk}\t{dead}\t{censored}")

# Calculate and print the number of worms at risk at each timepoint for each dataframe
initial_count = 270
at_risk_n2 = calculate_at_risk(df_n2, initial_count)
at_risk_nhr66 = calculate_at_risk(df_nhr66, initial_count)
at_risk_nhr49_nhr80 = calculate_at_risk(df_nhr49_nhr80, initial_count)
at_risk_nhr49 = calculate_at_risk(df_nhr49, initial_count)


# Combine the dataframes into a single dataframe
df_combined = pd.concat([df_n2, df_nhr66, df_nhr49_nhr80, df_nhr49])

# List to store fitted KaplanMeierFitter objects
kmf_list = []

plt.figure(figsize=(10, 6))  # Increase the width of the plot

# Sorting worms for the legend
sorted_worms = ['N2', 'nhr-66', 'nhr-49', 'nhr-49;nhr-80']

# Loop through each worm type in the sorted order and fit the Kaplan-Meier estimator
for worm in sorted_worms:
    worm_data = df_combined[df_combined['worm'] == worm]
    
    # Create a new KaplanMeierFitter object for each worm strain
    kmf = KaplanMeierFitter()
    kmf.fit(durations=worm_data['time'], event_observed=worm_data['status'], label=worm)
    
    # Add the fitted KaplanMeierFitter object to the list
    kmf_list.append(kmf)

# Customize the plot
plt.title('Survival on S. molluscorum Over Time')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.grid(True)

# Define the time points of recording
time_points = sorted(df_combined['time'].unique())

# Add a small horizontal offset to each line plot
offsets = np.linspace(-0.4, 0.4, len(kmf_list))

# Plot each Kaplan-Meier curve with a small horizontal offset
for kmf, offset in zip(kmf_list, offsets):
    plt.step(kmf.survival_function_.index + offset, kmf.survival_function_, where='post', label=kmf._label)

# Set the x-ticks to show only the time points of recording
plt.xticks(time_points)
# set the y-axis limits
plt.ylim(0, 1)

# Add the legend in the sorted order
plt.legend(title='Worm', labels=sorted_worms)

# Create a table with the number of worms at risk, dead, and censored for each timepoint
table_data = {
    'N2': at_risk_n2,
    'nhr-66': at_risk_nhr66,
    'nhr-49;nhr-80': at_risk_nhr49_nhr80,
    'nhr-49': at_risk_nhr49
}

# Add the table to the plot
# Create the table with worms as rows and time points as columns
cell_text = []
columns = ['Worms at Risk'] + [f'{time}h' for time in time_points]  # Adjust the time points as needed
rows = ['N2', 'nhr-66', 'nhr-49', 'nhr-49;nhr-80']

for worm in rows:
    cell_row = [worm]
    for time in time_points:
        at_risk_data = next((x for x in table_data[worm] if x[0] == time), (time, 0, 0, 0))
        cell_row.append(at_risk_data[1])  # Append the 'At Risk' count
    cell_text.append(cell_row)

# Create the table with increased font size
table = plt.table(cellText=cell_text, colLabels=columns, loc='bottom', cellLoc='center', bbox=[0.0, -0.5, 1.0, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)

# Make the column labels bold
for key, cell in table.get_celld().items():
    if key[0] == 0:  # This is the header row
        cell.set_text_props(fontproperties={'weight': 'bold'})

# Add p-values to the plot
p_values = {
    'N2 vs nhr-66': results_n2_vs_nhr66.p_value,
    'N2 vs nhr-49': results_n2_vs_nhr49.p_value,
    'N2 vs nhr-49;nhr-80': results_n2_vs_nhr49_nhr80.p_value
}

# Position for p-values text
text_x = max(time_points) * 0.015
text_y = 0.35

for label, p_value in p_values.items():
    plt.text(text_x, text_y, f'{label}: p={p_value:.4f}', fontsize=10)
    text_y -= 0.05

# Adjust layout to make room for the table
plt.subplots_adjust(bottom=0.3)

# Save figure
plt.savefig('/Volumes/behavgenom$/John/data_exp_info/NHR/Survival/Summary/20240829-20241008/nhr_survival_op50.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# %%
