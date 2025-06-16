#%%
import pandas as pd
from pathlib import Path
import csv

#%%

to_twist_1 = Path('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/7_order/orders_split/to_twist/FS5_1.csv')
to_twist_2 = Path('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/7_order/orders_split/to_twist/FS5_2.csv')
to_twist_3 = Path('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/7_order/orders_split/to_twist/FS5_3.csv')
from_twist_1 = Path('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/7_order/orders_split/from_twist/FS5_1.csv')
from_twist_2 = Path('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/7_order/orders_split/from_twist/FS5_2.csv')
from_twist_3 = Path('/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/7_order/orders_split/from_twist/FS5_3.csv')

#%%
# I want to make sure that the 'Name' and 'Insert sequence' columns are the same in the files ending with the same number

to_twist_1_df = pd.read_csv(to_twist_1)
to_twist_2_df = pd.read_csv(to_twist_2)
to_twist_3_df = pd.read_csv(to_twist_3)
from_twist_1_df = pd.read_csv(from_twist_1)
from_twist_2_df = pd.read_csv(from_twist_2)
from_twist_3_df = pd.read_csv(from_twist_3)

#%%
def check_columns(df1, df2):
    for i in range(len(df1)):
        if df1.loc[i, 'Name'] != df2.loc[i, 'Name'] or df1.loc[i, 'Insert sequence'] != df2.loc[i, 'Insert sequence']:
            return False
    return True

# Check the columns for each pair of DataFrames
result_1 = check_columns(to_twist_1_df, from_twist_1_df)
result_2 = check_columns(to_twist_2_df, from_twist_2_df)
result_3 = check_columns(to_twist_3_df, from_twist_3_df)

print(f"Check result for FS5_1: {result_1}")
print(f"Check result for FS5_2: {result_2}")
print(f"Check result for FS5_3: {result_3}")
# %%
