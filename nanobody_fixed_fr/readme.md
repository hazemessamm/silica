# Template-based nanobody generation

Currently there is only one protocol (`esm3_egfr_generate_nanobodies_with_fixed_FRs.ipynb` file, tested in Google Colab) present, it does the following:

1. Downloads 3 pdb files with nanobodies from rcsb.org
    - https://www.rcsb.org/structure/4krl
    - https://www.rcsb.org/structure/4KRN
    - https://www.rcsb.org/structure/4krp
2. Extracts nanobody sequences, numerates them with sadie
3. Extracts FR regions - both coordinates and aminoacid sequence - fixes them, uses as a templates
4. For each of the templates generates N full sequences (with varying length of CDRs) and corresponding full-atom structures (with previously fixed coordinates of FR regions).
5. Currently there are 3 different settings, generated data is saved with the following prefixes:
    - `nano_cdr3_`: here we generate only amino acids and their atom coordinates from CDR3
    - `nano_cdr1-3_`: here we generate data for CDR1, CDR2 and CDR3
    - `nano_coords_`: inverse folding mode: start from FR atoms coordinates and corresponding secondary structure, reconstruct the amino acid sequence, then reconstruct the missing atom positions

    In all 3 settings we vary the lengths of generated fragment.

6. Collects all results and downloads them (this part is Google Colab-specific)

## Validation & filtering

In silico drug design is extremely hard. The hardest part is not the generation method to pick, but the part where you are trying not to make chemists laugh at your attempts to create realistic molecule which might not be synthesizable, might not produce the expected fold, might not bind. 

That's why we filtered our generated data taking the following into account:

0. We used the script to generate 225 sequences.
1. We renumbered the sequences (we used sadie with Chothia numbering) and removed the ones which had unprocessed 'follows' part after renumbering. If the whole nanobody was numbered before the generation, and it didn't work well after - it might not be nanobody any more.
2. FR regions are highly conservative. It affects amino acids at certain positions, it also means that gaps are rare. We filtered out the sequences which had gaps in FR1.
We also excluded from consideration sequences with 3 aa gap in FR3: this gap was labeled as possible by our team's antibody expert.
3. We also did Cysteines check and removed all generated sequences which had this amino acid on any positions except 2 where this aminoacid appears almost constantly
4. Since we focused on CDR changes, we needed some additional way to understand if our Frankenstein nanobody is still a nanobody. To check this, we decided to compare the  embeddings for generated vs. real nanobodies.
  * We've decided that our `real nanobodies` dataset will be this one: https://research.naturalantibody.com/static/downloads/patent_sequence.tsv.gz (patent nanobody sequences)
  * We decided to use nanoBERT as a source of embeddings. First, for each of the nanobodies we extracted token embeddings. After that, we kept embeddings of the tokens which correspond to aminoacids located at +- 3 positions near the region border between CDR and FR. Using them, we computed the mean embedding, which emphasizes the edge of the conservative and variable region.
  * To understand what's going on, we computed Mahalanobis distance between the  embeddings for generated data and the distribution of patent embeddings. Thus making sure that our generated data isn't located too far from nanobody space and is not an outlier.