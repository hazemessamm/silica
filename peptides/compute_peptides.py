import pandas as pd
import numpy as np
from pathlib import Path
import peptides
from pandarallel import pandarallel
import argparse

pandarallel.initialize()


numeric_features = [
    "aliphatic_index",
    "boman",
    "charge",
    "descriptors",  # contains all descriptors so do not need physical_descriptors
    "frequencies",
    "hydrophobic_moment",
    "hydrophobicity",
    "instability_index",
    "isoelectric_point",
    "mass_shift",
    # 'membrane_position_profile',  # may be useful, but may need to be parameterized ex: T, S
    "molecular_weight",
    "mz",
    #  'structural_class',  # may be useful, is the predicted structural class ex: alpha
]

vector_features = [
    "hydrophobic_moment_profile",
    "hydrophobicity_profile",
    "linker_preference_profile",
]


def compute_peptides(seq: str, vector: bool = False):

    """
    Get descriptors computed by the peptides package for input sequence.

    Parameters:
        seq (str): The protein sequence.
        vector (bool): Whether vector descriptors should be returned. Default False.
    """
    
    features = {}
    pep = peptides.Peptide(seq)
    if vector:
        pep_features = numeric_features + vector_features
    else:
        pep_features = numeric_features
    for i in pep_features:
        if i != "descriptors" and i != "frequencies":
            features[i] = getattr(pep, i)()
        elif i == "descriptors":
            features.update(getattr(pep, i)())
        elif i == "frequencies":
            features.update({k + "_frequency": v for k, v in getattr(pep, i)().items()})
    if vector:
        features["hydrophobicity_profile"] = list(features["hydrophobicity_profile"])
        features["hydrophobic_moment_profile"] = list(
            features["hydrophobic_moment_profile"]
        )
    features["sequence"] = seq
    return features


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Compute peptides for a CSV input file expecting sequences in a column named sequences.')

    # Add an argument for the CSV input file
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file expecting sequences in a column named sequences.')

    # Add an argument for the CSV output file
    parser.add_argument('out_file', type=str, help='Path for the computed peptides features of each sequence.')

    # Parse the arguments
    args = parser.parse_args()

    # Access the CSV file argument
    csv_file_path = args.csv_file
    out_file_path = args.out_file
    print(f'Input CSV file: {csv_file_path}')

    # Load data
    df = pd.read_csv(csv_file_path)
    
    # Compute peptides. Expects sequences to be in sequence column
    peptides_features = pd.DataFrame(df["sequence"].parallel_apply(compute_peptides).tolist())
    
    # Save
    peptides_features.to_csv(f"{out_file_path}", index=False)
    print(f'Peptides saved to file: {out_file_path}')

if __name__ == '__main__':
    main()