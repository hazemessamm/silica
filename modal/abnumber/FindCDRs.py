import modal
import sys
import pkg_resources
import argparse
import os
import sys
import csv
import re
import pandas as pd

app = modal.App("get_cdr_indices")

dockerfile_image = modal.Image.from_dockerfile("Dockerfile")

def clean_non_ascii(text):
    # Remove any non-ASCII characters from the text
    return re.sub(r'[^\x00-\x7F]+', '', text)

def get_chain_indices(ordered_dict):
    result = {}
    
    for chain, residues in ordered_dict.items():
        # Extract the first and last residue objects (which have .cdr_definition_position)
        first_residue = next(iter(residues.keys()))  # First key
        last_residue = next(reversed(residues.keys()))  # Last key
        
        # Access the .cdr_definition_position attribute for start and end positions
        start_index = first_residue.cdr_definition_position
        end_index = last_residue.cdr_definition_position
        
        result[chain] = (start_index, end_index)
    
    return result

@app.function(image=dockerfile_image, concurrency_limit=5)
def get_chain_length(seq: str, scheme: str) -> str:
    # Import necessary modules
    from abnumber import Chain, ChainParseError

    # Perform abnumber chain operation using the specified scheme
    try:
        chain = Chain(seq, scheme=scheme)
        return get_chain_indices(chain.regions)
    except ChainParseError as e:
        # Return an error message as a string
        return f"Error processing sequence '{seq[:10]}...' with scheme '{scheme}': {str(e)}"
    except Exception as e:
        # Return a generic error message as a string
        return f"Unexpected error for sequence '{seq[:10]}...' with scheme '{scheme}': {str(e)}"

def read_sequences_from_csv(filepath):
    sequences = []
    try:
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Assuming the sequence is in the first column and not empty
                if row and row[0].strip():
                    sequences.append(clean_non_ascii(row[0].strip()))
        if not sequences:
            raise ValueError("The CSV file is empty or contains no valid sequences.")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)
    return sequences

@app.local_entrypoint()
def main(seq: str, scheme: str):

    #check if the sequence is a file path
    if os.path.isfile(seq):
        print(f"Reading sequences from file: {seq}")
        sequences = read_sequences_from_csv(seq)
    else:
        print(f"Processing single sequence: {seq})")
        sequences = [seq]

    print(f"Processing sequences using scheme: {scheme}")
    
    results = get_chain_length.starmap([(seq, scheme) for seq in sequences], return_exceptions=True)

    result_list = []
    for seq, result in zip(sequences, results):
        result_list.append((seq, result))

    # save results to a csv file. First column is the sequence, second column is the result
    df = pd.DataFrame(result_list, columns=['Sequence', 'Indices'])
    df.to_csv('results.csv', index=False)
    
    return result_list