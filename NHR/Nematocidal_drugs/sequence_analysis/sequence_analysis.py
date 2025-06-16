'''
(C) John Bergqvist 2025

Extract the protein fasta sequences of the NHRs that were included in the mutants
that the two-way ANOVA analysis showed to be significantly more sensitive to nematocidal drugs.

'''

#%%
import os
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from pathlib import Path

import requests
import time

#%%

def download_uniprot_ids(query, fields, size, output_file, base_url="https://rest.uniprot.org/uniprotkb/search"):
    """
    Downloads UniProt IDs based on a query and saves them to a file.

    Parameters:
        query (str): UniProt search query.
        fields (str): Fields to include in the search results (e.g., 'accession,gene_names').
        size (int): Maximum number of entries per page (UniProt's max is 500).
        output_file (str): Path to the file where results will be saved.
        base_url (str): Base URL for UniProt REST API (default is 'https://rest.uniprot.org/uniprotkb/search').
    """
    # Parameters for the API request
    params = {
        "query": query,
        "fields": fields,
        "format": "tsv",
        "size": size
    }

    print(f"Searching UniProt for query: {query}...")
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        # Save the response text to the output file
        with open(output_file, "w") as f:
            f.write(response.text)
        lines = response.text.strip().split("\n")
        print(f"Found {len(lines) - 1} entries.")
        print(f"IDs saved to {output_file}")
    else:
        print(f"Error: {response.status_code} - {response.text}")



def extract_accession_numbers(file_path):
    """
    Extracts accession numbers (Entry column) from a .txt file.

    Parameters:
        file_path (str or Path): Path to the .txt file.

    Returns:
        list: A list of accession numbers from the 'Entry' column.
    """
    accession_numbers = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip the header line
        for line in lines:
            columns = line.strip().split('\t')  # Split by tab
            accession_numbers.append(columns[0])  # Append the 'Entry' column
    return accession_numbers



def save_fasta_sequences(uniprot_ids, fasta_url, output_file, delay=0.5):
    """
    Fetches FASTA sequences for a list of UniProt IDs and saves them to a file.

    Parameters:
        uniprot_ids (list): List of UniProt IDs to fetch sequences for.
        fasta_url (str): URL template for fetching FASTA sequences (e.g., "https://www.uniprot.org/uniprot/{}.fasta").
        output_file (str): Path to the output file where sequences will be saved.
        delay (float): Delay between requests to avoid overwhelming the server (default is 0.5 seconds).
    """
    with open(output_file, "w") as f:
        for uid in uniprot_ids:
            try:
                response = requests.get(fasta_url.format(uid), timeout=10)
                if response.status_code == 200:
                    f.write(response.text)
                    print(f"Retrieved: {uid}")
                else:
                    print(f"Failed for {uid}: Status {response.status_code}")
            except Exception as e:
                print(f"Error retrieving {uid}: {e}")
            time.sleep(delay)  # polite delay

    print(f"\nAll sequences saved to {output_file}")


def modify_fasta_headers(input_fasta, output_fasta):
    """
    Modifies the headers of a FASTA file to include both the gene name and accession number,
    and reorders the sequences in ascending order based on the 'number' part of the gene name.

    Parameters:
        input_fasta (str or Path): Path to the input FASTA file.
        output_fasta (str or Path): Path to the output FASTA file with modified headers.
    """
    records = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        # Extract accession number from the record ID
        accession_number = record.id.split('|')[1] if '|' in record.id else record.id

        # Extract gene name using 'GN=' as the identifier
        description = record.description
        gene_name = None
        if "GN=" in description:
            gene_name = description.split("GN=")[1].split()[0]  # Extract the gene name after 'GN='
        else:
            gene_name = "unknown"  # Fallback if 'GN=' is not found
            print(f"Warning: 'GN=' not found in description for {record.id}. Using 'unknown' as gene name.")

        # Combine gene name and accession number in the new header
        new_header = f"{gene_name}_{accession_number}"

        # Create a new record with the updated header
        new_record = SeqRecord(Seq(str(record.seq)), id=new_header, description="")
        records.append((gene_name, new_record))

    # Sort records based on the 'number' part of the gene name
    def extract_number(gene_name):
        if gene_name.startswith("nhr-") and gene_name[4:].isdigit():
            return int(gene_name[4:])
        return float('inf')  # Place non-nhr genes at the end

    records.sort(key=lambda x: extract_number(x[0]))

    # Write modified and reordered records to the output FASTA file
    SeqIO.write([record[1] for record in records], output_fasta, "fasta")
    print(f"Modified and reordered headers saved to {output_fasta}")


#%%
# Defined directories
output_dir = Path('/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/sequence_analysis')

#%%

# 1. Download all the NHR protein UniProt IDs for C. elegans

# Search parameters
query = 'organism_id:6239 AND (gene:nhr-)'
fields = 'accession,gene_names'
size = 500  
output_file = output_dir / 'c_elegans_nhr_uniprot_ids.txt'

download_uniprot_ids(query, fields, size, output_file)


#%%
# 2. Download all the NHR protein sequences in FASTA format for C. elegans

# File paths
output_file = output_dir / 'c_elegans_nhr_uniprot_ids.txt'
fasta_url = "https://rest.uniprot.org/uniprotkb/{}.fasta"
output_fasta_file = output_dir / 'all_c_elegans_nhr_sequences.fasta'

# Extract accession numbers
uniprot_ids = extract_accession_numbers(output_file)

# Save FASTA sequences
save_fasta_sequences(uniprot_ids, fasta_url, output_fasta_file)



#%%
#3. Modify the FASTA headers of all the NHR protein sequences to include only the gene name as the header

input_fasta = output_dir / 'all_c_elegans_nhr_sequences.fasta'
output_fasta = output_dir / 'all_c_elegans_nhr_sequences_modified.fasta'
modify_fasta_headers(input_fasta, output_fasta)

'''Send this fasta file to EMBL to get the phylogenetic tree of all the NHRs in C. elegans'''


#%%
# 4. Keep only unique NHRs (the longest sequence if there are duplicate nhrs with the same nhr- name) and save them in a new FASTA file

# Input and output file paths
input_fasta = output_dir / "all_c_elegans_nhr_sequences_modified.fasta"
output_fasta = output_dir / "longest_unique_nhr_sequences.fasta"

# Dictionary to store the longest sequence for each prefix
longest_sequences = {}

# Read the FASTA file
for record in SeqIO.parse(input_fasta, "fasta"):
    # Extract the prefix (everything before the underscore)
    prefix = record.id.split('_')[0]
    sequence_length = len(record.seq)
    
    # Check if this sequence is longer than the current longest for the prefix
    if prefix not in longest_sequences or sequence_length > len(longest_sequences[prefix].seq):
        longest_sequences[prefix] = record

# Write the longest sequences to a new FASTA file
with open(output_fasta, "w") as output_handle:
    SeqIO.write(longest_sequences.values(), output_handle, "fasta")

print(f"Longest sequences saved to {output_fasta}")

# Verify the number of unique sequences
unique_sequences = set(str(record.seq) for record in longest_sequences.values())
print(f"Number of unique sequences: {len(unique_sequences)}")

# Compare with the length of longest_sequences
if len(unique_sequences) == len(longest_sequences):
    print("The number of unique sequences matches the number of unique NHRs.")
else:
    print("Mismatch detected: The number of unique sequences does not match the number of unique NHRs.")

'''Send this fasta file to EMBL to get the phylogenetic tree of all the NHRs in C. elegans'''

#%%
# 5. Get the protein sequences for only the nhr of strains with significant strain:drug interactions in fasta format
# Load the csv file with NHRs and uniprot ids

df = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/sequence_analysis/uniprotid_sig_nhrs.csv')

uniprot_ids = df['uniprot_id'].tolist()

fasta_url = "https://rest.uniprot.org/uniprotkb/{}.fasta"

# Output file
output_file = output_dir / 'sig_nhrs_seq.fasta'

save_fasta_sequences(uniprot_ids, fasta_url, output_file)


#%%




# %%
# The above significant NHR mutants come from the two-way ANOVA analysis with Type II sum of squares. 
# Given unequal sample sizes between the control condition and the drug condition, Type III sum of squares is more appropriate.
# As such, I should redo the above steps using the data from the ANOVA with Type III sum of squares.

# Load the csv file with the significant NHRs from the two-way ANOVA with Type III sum of squares

df = pd.read_csv('/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/statistical_analyses/all/twoway_anova/type_III_anova_result/type_III_anova_result.csv')
print(f"Before removing control diffrence and unsure results:", len(df))

# Remove rows that in the column 'stat_sig_type' has thee string 'control difference' or 'unsure'
df = df[~df['stat_sig_type'].isin(['control difference', 'unsure'])]

# Load the specific sheet and extract the required columns
strain_genotype = pd.read_excel(
    '/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Auxiliary_files/plate_to_strains_all_runs.xlsx',
    sheet_name='Sheet1'
)

# Select only the 'strain' and 'full_genotype' columns
strain_genotype = strain_genotype[['strain', 'full_genotype']]

# Display the extracted data
print(strain_genotype)

# combine the two dataframes on the 'strain' column
df = pd.merge(df, strain_genotype, on='strain', how='left')
print(f"after removing control difference and unsure results, and merging with strain_genotype:", len(df))

print(f"Number of unique worm strains:", df['strain'].nunique())

# %%
import re

output_dir = Path('/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/sequence_analysis')

# Extract all unique genes from the 'full_genotype' column into a list
gene_list = []
for genotype in df['full_genotype']:
    if pd.notna(genotype):  # Check if the value is not NaN
        # Remove anything inside parentheses
        cleaned_genotype = re.sub(r'\(.*?\)', '', genotype)
        
        # Split by semicolons or commas and strip whitespace
        genes = re.split(r'[;,]', cleaned_genotype)
        genes = [gene.strip().lower() for gene in genes]
                
        # Remove non-standard characters (e.g., BOM or other symbols)
        genes = [re.sub(r'[^\w-]', '', gene) for gene in genes]

        # Extend the gene list with cleaned genes
        gene_list.extend(genes)

# Remove duplicates while preserving order
unique_genes = list(dict.fromkeys(gene_list))

print(unique_genes)
print(f"Total unique genes: {len(unique_genes)}")



# %%
# From the merged 'df', I want to extract the unique NHRs that are more 'mutant sensitive', 'mutant resistant' and 'mutant gain'. 
# There should be two columns in the output csv file: 'gene' from 'full_genotype' and 'stat_sig_type' 
# The string in the 'stat_sig_type' should be either 'mutant sensitive', 'mutant resistant' or 'mutant gain' depending on which value in the 'stat_sig_type' column the gene is associated with.

# Extract unique NHRs with their stat_sig_type

# Extract unique NHRs with their stat_sig_type
def extract_unique_nhrs(df):
    # Create an empty list to store the results
    results = []

    # Iterate through each row in the dataframe
    for _, row in df.iterrows():
        # Split the 'full_genotype' column into individual NHRs using both ';' and ','
        nhrs = re.split(r'[;,]', row['full_genotype'])
        # Append each NHR with its associated 'stat_sig_type' to the results list
        for nhr in nhrs:
            # Remove brackets and their contents, and convert to lowercase
            cleaned_nhr = re.sub(r'\(.*?\)', '', nhr).strip().lower()
            results.append({'gene': cleaned_nhr, 'stat_sig_type': row['stat_sig_type']})

    # Convert the results list into a DataFrame
    unique_nhrs_df = pd.DataFrame(results).drop_duplicates()

    # Save the DataFrame to a CSV file
    unique_nhrs_df.to_csv('unique_nhrs.csv', index=False)

    return unique_nhrs_df

# Call the function and extract the unique NHRs
unique_nhrs_df = extract_unique_nhrs(df)

# Print the resulting DataFrame
print(unique_nhrs_df)
# %%
# Save the list to a csv file
output_file = '/Volumes/behavgenom$/John/data_exp_info/NHR/Nematicidial_drugs/data/Analysis/data/sequence_analysis/type_III_twoway_anova_sig_nhrs.csv'
unique_genes_df = pd.DataFrame(unique_nhrs_df, columns=['gene', 'stat_sig_type'])
unique_genes_df.to_csv(output_file, index=False)
# %%
