Filtering metrics borrowed from [martinpacesa/BindCraft](https://github.com/martinpacesa/BindCraft)

```python
    # output interface score array and amino acid counts at the interface
    interface_scores = {
    'binder_score': binder_score,
    'surface_hydrophobicity': surface_hydrophobicity,
    'interface_sc': interface_sc,
    'interface_packstat': interface_packstat,
    'interface_dG': interface_dG,
    'interface_dSASA': interface_dSASA,
    'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
    'interface_fraction': interface_binder_fraction,
    'interface_hydrophobicity': interface_hydrophobicity,
    'interface_nres': interface_nres,
    'interface_interface_hbonds': interface_interface_hbonds,
    'interface_hbond_percentage': interface_hbond_percentage,
    'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
    'interface_delta_unsat_hbonds_percentage': interface_bunsch_percentage
    }
```

### USAGE
`pyrosetta_utils_modal.py` expects .pdb designs from AlphaFold2 Initial Guess stored in `diffusion-volume` as .pdb file
Additionally, AF2 metrics should be gathered in `/csv/aggregated_designs.csv` to be later merged with BindCraft metrics into `final.csv`