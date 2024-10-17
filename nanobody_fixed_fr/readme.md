# Template-based nanobody generation

Currently there is only one protocol (`esm3_egfr_generate_nanobodies_with_fixed_FRs.ipynb` file, tested in Google Colab) present, it does the following:

1. Downloads 3 pdb files with nanobodies from rcsb.org
    - https://www.rcsb.org/structure/4krl
    - https://www.rcsb.org/structure/4KRN
    - https://www.rcsb.org/structure/4krp
2. Extracts nanobody sequences, numerates them with sadie
3. Extracts FR regions - both coordinates and aminoacid sequence - fixes them, uses as a templates
4. For each of the templates generates N full sequences (with varying length of CDRs) and corresponding full-atom structures (with previously fixed coordinates of FR regions).
5. Collects all results and downloads them (this part is Google Colab-specific)


