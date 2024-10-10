# Peptides Descriptors

Compute peptide (https://peptides.readthedocs.io/en/stable/api/peptide.html#peptides.Peptide) descriptors for protein sequences.

## Dependencies
```shell
pip install -r requirements.txt
```

## Example Usage
Expects input CSV to have a sequence column.
```shell
python compute_peptides.py ../source_data/adaptyv_biolm_egfr_data.csv peptides.csv
```
