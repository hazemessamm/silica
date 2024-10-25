import modal
from pathlib import Path
import shutil
import time

REMOTE_DESIGNS_DIR = Path("/diffusion_output/af2")
REMOTE_DESIGNS_DIR_COPY = Path("/diffusion_output/af2_copy")

######### TEST CODE #########
LOCAL_DESIGNS_DIR = Path(__file__).parent / "diffusion-volume"
designs_mount = modal.Mount.from_local_dir(LOCAL_DESIGNS_DIR, remote_path=REMOTE_DESIGNS_DIR)

# Create the shared volume
shared_volume = modal.Volume.from_name("diffusion-volume", create_if_missing=True)
############################


def build_image():
    return (
        modal.Image.debian_slim(python_version="3.11.5")
        .pip_install(
            "biopython",
            "numpy<2.0.0",
            "pyrosetta-installer",
            "scipy",
        )
        .run_commands(
            "python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'",
        )
    )

# Clean unnecessary rosetta information from PDB
def clean_pdb(pdb_file):
    with open(pdb_file, 'r') as f_in:
        relevant_lines = [line for line in f_in if line.startswith(('ATOM', 'HETATM', 'MODEL', 'TER', 'END'))]
    with open(pdb_file, 'w') as f_out:
        f_out.writelines(relevant_lines)


bindcraft_image = build_image()
app = modal.App("PyRosettaScoreInterfaceviaBindCraft", image=bindcraft_image, mounts=[designs_mount], volumes={"/shared": shared_volume})


def combine_df_with_json(df, sequence_col='sequence'):
    import json 
    import copy
    _df = copy.deepcopy(df)
    _df['metadata'] = _df.drop(columns=[sequence_col]).apply(lambda row: json.dumps(row.to_dict()), axis=1)
    _df2 = _df[[sequence_col, 'metadata']]

    return _df2

@app.function()
def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0):
    from Bio.PDB import Selection, PDBParser
    from scipy.spatial import cKDTree

    import numpy as np

    three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0]['A'], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues

@app.function()
def score_interface(pdb_file: str, binder_chain: str = "B"):
    import pyrosetta as pr
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

    pr.init(extra_options="-holes:dalphaball -mute core.conformation")

    pose = pr.pose_from_pdb(pdb_file)

    iam = InterfaceAnalyzerMover()
    iam.set_interface("A_B")
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    interface_residues_set = hotspot_residues.remote(pdb_file, binder_chain)
    interface_residues_pdb_ids = []
    
    for pdb_res_num, aa_type in interface_residues_set.items():
        interface_AA[aa_type] += 1
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    interface_nres = len(interface_residues_pdb_ids)
    interface_residues_pdb_ids_str = ','.join(interface_residues_pdb_ids)

    hydrophobic_aa = set('ACFILMPVWY')
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100 if interface_nres != 0 else 0

    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value
    interface_interface_hbonds = interfacescore.interface_hbonds
    interface_dG = iam.get_interface_dG()
    interface_dSASA = iam.get_interface_delta_sasa()
    interface_packstat = iam.get_interface_packstat()
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100
    buns_filter = XmlObjects.static_get_filter('<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="0" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    if interface_nres != 0:
        interface_hbond_percentage = (interface_interface_hbonds / interface_nres) * 100
        interface_bunsch_percentage = (interface_delta_unsat_hbonds / interface_nres) * 100
    else:
        interface_hbond_percentage = None
        interface_bunsch_percentage = None

    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    interface_binder_fraction = (interface_dSASA / binder_sasa) * 100 if binder_sasa > 0 else 0

    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
    surface_res = layer_sel.apply(pose)

    exp_apol_count = 0
    total_count = 0 
    
    for i in range(1, len(surface_res) + 1):
        if surface_res[i] == True:
            res = pose.residue(i)
            if res.is_apolar() == True or res.name() in ['PHE', 'TRP', 'TYR']:
                exp_apol_count += 1
            total_count += 1

    surface_hydrophobicity = exp_apol_count / total_count

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

    interface_scores = {k: round(v, 2) if isinstance(v, float) else v for k, v in interface_scores.items()}

    return interface_scores, interface_AA, interface_residues_pdb_ids_str

@app.function(timeout=7200)
def process_pdb_files():
    import csv
    import os
    import io


    # Create the new directory
    os.makedirs(REMOTE_DESIGNS_DIR_COPY, exist_ok=True)
    if REMOTE_DESIGNS_DIR_COPY:
            print(f"Successfully created {REMOTE_DESIGNS_DIR_COPY}!")
    else:
        print(f"COPY ERROR")

    results = []

    # Process each PDB file in the copied directory
    for pdb_file in REMOTE_DESIGNS_DIR.glob("*.pdb"):

        print(f'Trying with {pdb_file.name}')

        # Create the destination path for the copied file
        dest_file = REMOTE_DESIGNS_DIR_COPY / pdb_file.name
        
        # Copy the file
        shutil.copy2(pdb_file, dest_file)

        if dest_file:
            print(f"Successfully copied {pdb_file} to {dest_file}!")
        else:
            print(f"COPY ERROR")


        # Clean the PDB file
        clean_pdb(str(pdb_file))

        # Create a unique identifier
        unique_id = str(pdb_file).split('/')[-1].replace('.pdb', '')
        print(f'unique_id: {unique_id}')

        # Score the interface
        interface_scores, interface_AA, interface_residues_pdb_ids_str = score_interface.remote(str(pdb_file))

        print('[DEBUG] Obtained interface scores.')

        # Prepare the result
        result = {"ID": unique_id, **interface_scores, "interface_AA": interface_AA, "interface_residues_pdb_ids_str": interface_residues_pdb_ids_str}
        results.append(result)

    # Create CSV in memory
    output = io.StringIO()
    fieldnames = ["ID"] + list(interface_scores.keys()) + ["interface_AA", "interface_residues_pdb_ids_str"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)
    
    return output.getvalue()
   

@app.local_entrypoint()
def main():
    import pandas as pd
    # Process the PDB files and save the content into interface_scores.csv
    csv_content = process_pdb_files.remote()
    output_file = "csv/interface_scores.csv"
    
    with open(output_file, "w", newline="") as f:
        f.write(csv_content)
    
    print(f"Results written to {output_file}")
    
    # Load the two CSV files into DataFrames
    df_interface_scores = pd.read_csv("csv/interface_scores.csv")
    df_af2_results = pd.read_csv("csv/aggregated_designs.csv")
    
    # Merge the two DataFrames on the 'ID' column
    merged_df = pd.merge(df_interface_scores, df_af2_results, on="ID")
    
    # Apply combine_df_with_json on the merged DataFrame
    final_df = combine_df_with_json(merged_df, sequence_col="sequence")
    
    # Save the final DataFrame to a new CSV file
    final_output_file = "csv/final.csv"
    final_df.to_csv(final_output_file, index=False)
    
    print(f"Merged and processed results saved to {final_output_file}")

if __name__ == "__main__":
    modal.runner.deploy_stub(app)