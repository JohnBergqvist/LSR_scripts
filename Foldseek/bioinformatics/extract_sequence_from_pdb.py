
#%%
import pandas as pd
import os
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import IUPACData

# %%
pdbs_path = '/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/1_commerical_proteins_processing/all_commercial_proteins/all_commercial_pdbs'

fasta_save_path = '/Volumes/behavgenom$/John/data_exp_info/BT/FoldSeek/commercial_proteins/4_BLAST/1_commercial_BLAST/fasta'

multi_fasta_path = os.path.join(fasta_save_path, 'compiled_sequences.fasta')

os.makedirs(fasta_save_path, exist_ok=True)
# %%
# Function to extract sequences from a PDB file
def extract_sequences_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    sequences = []
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            seq = []
            for residue in chain:
                if residue.id[0] == ' ':
                    three_letter_code = residue.resname
                    one_letter_code = IUPACData.protein_letters_3to1.get(three_letter_code.capitalize(), 'X')
                    if one_letter_code == 'X':
                        print(f"Unknown residue: {three_letter_code} in {pdb_file}")
                    seq.append(one_letter_code)
            seq_str = ''.join(seq)
            seq_record = SeqRecord(Seq(seq_str), id=f"{os.path.basename(pdb_file).split('.')[0]}_{chain_id}", description="")
            sequences.append(seq_record)
    
    return sequences

#%%
## Check the number of sequences in the multi-fasta file after run to ensure the number
## of expected sequences are present. Issues with some missing and chains missing have 
## happened before without knowing why. 

# Iterate over all PDB files in the directory
for pdb_file in os.listdir(pdbs_path):
    if pdb_file.endswith('.pdb'):
        pdb_file_path = os.path.join(pdbs_path, pdb_file)
        sequences = extract_sequences_from_pdb(pdb_file_path)
        
        # Save sequences in FASTA format
        fasta_file_path = os.path.join(fasta_save_path, f"{os.path.basename(pdb_file).split('.')[0]}.fasta")
        with open(fasta_file_path, 'w') as fasta_file:
            SeqIO.write(sequences, fasta_file, 'fasta')
            print(f"Sequences from {pdb_file} have been written to {fasta_file_path}")

# Compile all individual FASTA files into one multi-FASTA file without duplicates
seen_sequences = set()
with open(multi_fasta_path, 'w') as multi_fasta_file:
    for fasta_file in os.listdir(fasta_save_path):
        if fasta_file.endswith('.fasta'):
            fasta_file_path = os.path.join(fasta_save_path, fasta_file)
            with open(fasta_file_path, 'r') as individual_fasta_file:
                for record in SeqIO.parse(individual_fasta_file, 'fasta'):
                    if str(record.seq) not in seen_sequences:
                        SeqIO.write(record, multi_fasta_file, 'fasta')
                        seen_sequences.add(str(record.seq))
                        print(f"Added sequence {record.id} to multi-FASTA file")

print("Multi-FASTA file has been created without duplicates.")
print(f"Number of unique sequences: {len(seen_sequences)}")
# %%
