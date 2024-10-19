import modal
from huggingface_hub import HfApi

from esm.utils.structure.protein_chain import ProteinChain
from esm.sdk import client
# from esm.sdk.api import (
#     ESMProtein,
#     GenerationConfig,
#     LogitsConfig,
#     SamplingConfig
# )

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    ESMProteinTensor,
    GenerationConfig,
    LogitsConfig,
    LogitsOutput,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation
from esm.utils.constants import models
import numpy as np
import os
from pathlib import Path

import torch

# Create a Docker image from the specified Dockerfile
docker_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3", add_python="3.11")
    .apt_install("git")  # Install git if needed
    .pip_install("huggingface_hub", "esm", "nglview", "plotly", "py3Dmol")  # Install required Python packages
    .env({"FORGE_TOKEN": "7iLuRQO7n1c1DNMcifs58W"})
)

# Create or access a Modal volume
volume = modal.Volume.from_name("hackaton_volume", create_if_missing=True)
MOUNT_DIR = "/data"
local_directory = "/Users/Cellini/codes/hackaton"

pdb_id = "4KRO"  # PDB ID corresponding to EGFR
chain_id = "B"  # Chain ID corresponding to the nanobody in the PDB complex

file_tag = "nano_ir_id"  # use this to prepend to the file names
NUM_SEEDS = 10

# Basic sequence definitions
# cut off the His tag
# seq from the fasta
seq_fasta = 'QVQLQESGGGLVQPGGSLRLSCAASGRTFSSYAMGWFRQAPGKQREFVAAIRWSGGYTYYTDSVKGRFTISRDNAKTTVYLQMNSLKPEDTAVYYCAATYLSSDYSRYALPQRPLDYDYWGQGTQVTVSSLEHHHHHH'
i = 0
while seq_fasta[len(seq_fasta)-i-1] == 'H':
    i += 1
seq_clean = seq_fasta[:len(seq_fasta)-i]

timeout = 86000  # seconds - this is now set to almost 24h

# Define a Modal app using the custom container image
app = modal.App("esm3_model", image=docker_image)

@app.function()
def get_sample_protein():
    egfr_nano_chain = ProteinChain.from_rcsb(pdb_id, chain_id)
    # convert it to ESMProtein object to enable function annotation
    protein = ESMProtein.from_protein_chain(egfr_nano_chain)

    modelled_res = egfr_nano_chain.residue_index_no_insertions  # numbering starts from 1; adjust later for python-friendly numbering
    seq = egfr_nano_chain.sequence  # seq from the pdb

    CDR1 = 'GRTFSSY' # 	CDRH1
    CDR2 = 'RWSGGY' # CDRH2
    CDR3 = 'TYLSSDYSRYALPQRPLDYDY' # CDRH3
    cdr_dict = {'CDR1': {},
            'CDR2': {},
            'CDR3': {}
            }

    for cdr_id, cdr in zip(['CDR1', 'CDR2', 'CDR3'],[CDR1, CDR2, CDR3]):
        idx = seq.index(cdr)
        idx_fasta = seq_fasta.index(cdr)
        cdr_dict[cdr_id]['seq_pos'] = [_i for _i in range(idx, idx+len(cdr))]
        cdr_dict[cdr_id]['fasta_pos'] = [_i for _i in range(idx_fasta, idx_fasta+len(cdr))]
        cdr_dict[cdr_id]['res'] = cdr

    # binding sites are based on seq_fasta; posistions are python friendly, i.e. 0 based
    binding_sites = [{
        'val': 32,
        'keyword': 'CDD - cd04981',
        }, {
        'val': 49,
        'keyword': 'CDD - cd04981'
        }, {
        'val': 98,
        'keyword': 'CDD - cd04981'
        }]

    h_dimer_interface = [
        {
        'val': 38,
        'keyword': 'CDD - cd04981'
        },
        {
        'val': 42,
        'keyword': 'CDD - cd04981'
        },
        {
        'val': 46,
        'keyword': 'CDD - cd04981'
        },
        {
        'val': 94,
        'keyword': 'CDD - cd04981'
        },
        {
        'val': 119,
        'keyword': 'CDD - cd04981'
        },
        {
        'val': 120,
        'keyword': 'CDD - cd04981'
        }]

    # construct the prompts

    _sites_bind = [
        FunctionAnnotation(label="antigen_binding_site", start=binding_site['val'], end=binding_site['val']+1)
        for binding_site in binding_sites
        ]

    _sites_interface = [
        FunctionAnnotation(label="heterodimer_interface", start=interface_site['val'], end=interface_site['val']+1)
        for interface_site in h_dimer_interface
        ]

    protein.sequence = seq_clean  # This is important! update the sequence with the relevant files

    # get the corresponding structure prompt
    adjusted_structure = torch.full((len(seq_clean), 37, 3), np.nan)

    structure_pos = [idx for idx in range(len(seq_clean)) if idx+1 in set(modelled_res)]
    adjusted_structure[structure_pos] = torch.tensor(
        egfr_nano_chain.atom37_positions
    )
    protein.coordinates = adjusted_structure  # update the structure that aligns with seq_clean
    protein.function_annotations = _sites_bind + _sites_interface

    return protein

@app.function(gpu="H100", volumes={MOUNT_DIR: volume}, timeout=timeout)  # Mounting at /data  "a10g"
def run_model_and_save_pdbs():
    forge_token = os.environ["FORGE_TOKEN"]

    model = client(
        model="esm3-large-2024-03",
        url="https://forge.evolutionaryscale.ai",
        token=forge_token,
        )

    print("login was successful")
    # prepare the data
    pdb_id = "4KRO"  # PDB ID corresponding to EGFR
    chain_id = "B"  # Chain ID corresponding to the nanobody in the PDB complex
    egfr_nano_chain = ProteinChain.from_rcsb(pdb_id, chain_id)
    modelled_res = egfr_nano_chain.residue_index_no_insertions
    # convert it to ESMProtein object to enable function annotation
    egfr_nano_chain = ESMProtein.from_protein_chain(egfr_nano_chain)

    # seq from the fasta
    seq_fasta = 'QVQLQESGGGLVQPGGSLRLSCAASGRTFSSYAMGWFRQAPGKQREFVAAIRWSGGYTYYTDSVKGRFTISRDNAKTTVYLQMNSLKPEDTAVYYCAATYLSSDYSRYALPQRPLDYDYWGQGTQVTVSSLEHHHHHH'
    seq = egfr_nano_chain.sequence  # seq from the pdb

    CDR1 = 'GRTFSSY' # 	CDRH1
    CDR2 = 'RWSGGY' # CDRH2
    CDR3 = 'TYLSSDYSRYALPQRPLDYDY' # CDRH3
    cdr_dict = {'CDR1': {},
            'CDR2': {},
            'CDR3': {}
            }

    for cdr_id, cdr in zip(['CDR1', 'CDR2', 'CDR3'],[CDR1, CDR2, CDR3]):
        idx = seq.index(cdr)
        idx_fasta = seq_fasta.index(cdr)
        cdr_dict[cdr_id]['seq_pos'] = [_i for _i in range(idx, idx+len(cdr))]
        cdr_dict[cdr_id]['fasta_pos'] = [_i for _i in range(idx_fasta, idx_fasta+len(cdr))]
        cdr_dict[cdr_id]['res'] = cdr

    # binding sites are based on seq_fasta; posistions are python friendly, i.e. 0 based
    binding_sites = [{
        'val': 32,
        'keyword': 'CDD - cd04981',
        }, {
        'val': 49,
        'keyword': 'CDD - cd04981'
        }, {
        'val': 98,
        'keyword': 'CDD - cd04981'
        }]

    h_dimer_interface = [
        {
        'val': 38,
        'keyword': 'CDD - cd04981'
        },
        {
        'val': 42,
        'keyword': 'CDD - cd04981'
        },
        {
        'val': 46,
        'keyword': 'CDD - cd04981'
        },
        {
        'val': 94,
        'keyword': 'CDD - cd04981'
        },
        {
        'val': 119,
        'keyword': 'CDD - cd04981'
        },
        {
        'val': 120,
        'keyword': 'CDD - cd04981'
        }]

    # construct the prompts
    binding_motif = np.array([binding_site['val'] for binding_site in binding_sites])
    interface_motif = np.array([interface_site['val'] for interface_site in h_dimer_interface])

    get_sample_protein_call = get_sample_protein.spawn()
    protein = get_sample_protein_call.get()

    print("this worked so far...")

    # Now the sequence generation first - mask out the CDRs, plus the binding site
    _cdr_mask = []
    for _, val in cdr_dict.items():
        _cdr_mask.extend(val['seq_pos'])

    # add to this the binding and interface sites:
    all_masks = _cdr_mask + binding_motif.tolist() + interface_motif.tolist()
    res = ["_" if idx in set(all_masks) else char for idx, char in enumerate(seq_clean)]
    seq_prompt = (''.join(res))


    # put together the structure prompt
    all_structure_pos = {res-1: idx for idx, res in enumerate(modelled_res)}
    # take out the CDRs and interface structures and leave in the binding sites
    structure_mask = set(all_structure_pos.keys()) - set(_cdr_mask) - set(interface_motif)
    seq_pos_map = {idx: all_structure_pos[idx] if idx in all_structure_pos else None for idx in range(len(seq_clean))}

    structure_motif_pos = [val for key, val in seq_pos_map.items() if key in structure_mask and val != None]
    # needed to extract the atom positions
    egfr_PC = ProteinChain.from_rcsb(pdb_id, chain_id)
    structure_mofif = egfr_PC.atom37_positions[structure_motif_pos]

    # get the corresponding structure prompt
    structure_prompt = torch.full((len(seq_prompt), 37, 3), np.nan)


    structure_prompt[structure_motif_pos] = torch.tensor(
        structure_mofif
    )

    # print("Structure prompt shape: ", structure_prompt.shape)
    # print(
    #     "Indices with structure conditioning: ",
    #     torch.where(~torch.isnan(structure_prompt).any(dim=-1).all(dim=-1))[0].tolist(),
    # )


    # SEEDS
    seeds = np.random.randint(0, 10**6, size=NUM_SEEDS)

    for _nr, seed in enumerate(seeds):
        print(f"generating nr: {_nr} and seed: {seed}")

        try:

            # combine the sequence and structure prompt
            protein_prompt = ESMProtein(sequence=seq_prompt, coordinates=structure_prompt)

            # We'll have to first construct a `GenerationConfig` object that specifies the decoding parameters that we want to use
            sequence_generation_config = GenerationConfig(
                track="sequence",  # We want ESM3 to generate tokens for the sequence track
                num_steps=seq_prompt.count("_")
                // 2,  # We'll use num(mask tokens) // 2 steps to decode the sequence
                temperature=0.5,  # We'll use a temperature of 0.5 to control the randomness of the decoding process
            )

            # Now, we can use the `generate` method of the model to decode the sequence
            sequence_generation = model.generate(protein_prompt, sequence_generation_config)

            assert isinstance(sequence_generation, ESMProtein)

            seq_gen_file_name = f"{file_tag}_tag_{seed}_seq_gen.pdb"
            seq_gen_pdb_path = Path(MOUNT_DIR) / seq_gen_file_name
            sequence_generation.to_pdb(str(seq_gen_pdb_path))

            print(f"sequence generated too... {sequence_generation.sequence}")

            # now the structure generation
            # ----------------------------
            structure_prediction_config = GenerationConfig(
                track="structure",  # We want ESM3 to generate tokens for the structure track
                num_steps=len(sequence_generation) // 8,
                temperature=0.7,
            )
            structure_prediction_prompt = ESMProtein(sequence=sequence_generation.sequence)
            structure_prediction = model.generate(
                structure_prediction_prompt, structure_prediction_config
            )

            assert isinstance(structure_prediction, ESMProtein)

            struct_gen_file_name = f"{file_tag}_tag_{seed}_struct_gen.pdb"
            struct_gen_pdb_path = Path(MOUNT_DIR) / struct_gen_file_name
            structure_prediction.to_pdb(str(struct_gen_pdb_path))

            print(f"structure generated too...")


            # Convert the generated structure to a back into a ProteinChain object
            structure_prediction_chain = structure_prediction.to_protein_chain()


            pdb_str = egfr_PC.to_pdb_string()
            _cdr_idxs = [idx+1 for idx, res in enumerate(seq_prompt) if res=='_']

            diffs = 0
            for idx in range(len(sequence_generation.sequence)):
                if sequence_generation.sequence[idx] != seq_clean[idx]:
                    if idx+1 in set(_cdr_idxs):
                        print(idx, sequence_generation.sequence[idx], seq_clean[idx])
                    diffs += 1
            new_res_count = diffs

            # inv_folding
            num_steps = int(len(seq_clean) / 16)
            get_sample_protein_call = get_sample_protein.spawn()
            protein = get_sample_protein_call.get()
            protein.coordinates = structure_prediction.coordinates
            protein.sequence = None
            protein.sasa = None
            protein.function_annotations = None
            inv_folded_protein = model.generate(
                protein,
                GenerationConfig(track="sequence", schedule="cosine", num_steps=num_steps),
            )
            assert isinstance(inv_folded_protein, ESMProtein)

            inv_fold_file_name = f"{file_tag}_tag_{seed}_inv_fold.pdb"
            inv_fold_pdb_path = Path(MOUNT_DIR) / inv_fold_file_name
            protein.to_pdb(str(inv_fold_pdb_path))

            # get logits
            # protein = get_sample_protein()
            # protein.coordinates = None
            # protein.function_annotations = None
            # protein.sasa = None
            # protein_tensor = model.encode(protein)
            # logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True))
            # assert isinstance(
            #     logits_output, LogitsOutput
            # ), f"LogitsOutput was expected but got {logits_output}"
            # assert (
            #     logits_output.logits is not None and logits_output.logits.sequence is not None
            # )

            volume.commit()

        except AssertionError as e:
            print(f"{e} in round {_nr}")
            continue

@app.function(image=docker_image)
def download_files(volume, file_to_download, local_directory):
    """Download a specified file from the volume to the local directory."""

    # Ensure local directory exists
    os.makedirs(local_directory, exist_ok=True)

    # Download the specified file from the volume to the local directory
    print(f"Downloading {file_to_download} to {local_directory}...")

    modal.volume.get(volume, file_to_download, f"{local_directory}/{file_to_download}")

    print(f"Downloaded {file_to_download} successfully.")

@app.local_entrypoint()
def main():
    # Run the model and save PDBs
    run_model_and_save_pdbs.remote()
    # function_call.get()

    # local_dir = local_directory

    # vol = modal.Volume.lookup("hackaton_volume")
    # os.makedirs(local_dir, exist_ok=True)

    # # List all files in the root of the volume
    # for file_entry in vol.listdir("/"):  # Use "/" for root

    #     remote_path = file_entry.path  # Use the path attribute directly
    #     filename = os.path.basename(remote_path)  # Extract just the filename
    #     local_file_path = os.path.join(local_dir, filename)

    #     with open(local_file_path, "wb") as local_file:
    #         for chunk in vol.read_file(remote_path):  # No need for folder name here
    #             local_file.write(chunk)

if __name__ == "__main__":
    main()