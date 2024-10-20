import modal
from huggingface_hub import HfApi
import pandas as pd

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

# Create a Docker container. Here we use the official nvidia pytorch
# Note that this also takes in the Forge token. Make sure it exists, so that the routine has access to ESM3 API.
docker_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3", add_python="3.11")
    .apt_install("git")  # Install git if needed
    .pip_install("huggingface_hub", "esm", "nglview", "plotly", "py3Dmol")  # Install required Python packages
    .env({"FORGE_TOKEN": "7iLuRQO7n1c1DNMcifs58W"})
)

# Create or access a Modal volume
volume = modal.Volume.from_name("hackaton_volume", create_if_missing=True)
MOUNT_DIR = "/data"

pdb_id = "4KRO"  # PDB ID corresponding to EGFR
chain_id = "B"  # Chain ID corresponding to the nanobody in the PDB complex

file_tag = "nano_ir_id"  # use this to prepend to the file names
run_id_csv_tag = "metrics_run_v2339.csv"

# Number of seeds determines how many rounds we generate. Be mindful using the large model will take more time.
NUM_SEEDS = 10

# Basic sequence definitions - we obtained the fasta sequence from the relevant fasta in the PDB entry
# cut off the His tag
# seq from the fasta
seq_fasta = 'QVQLQESGGGLVQPGGSLRLSCAASGRTFSSYAMGWFRQAPGKQREFVAAIRWSGGYTYYTDSVKGRFTISRDNAKTTVYLQMNSLKPEDTAVYYCAATYLSSDYSRYALPQRPLDYDYWGQGTQVTVSSLEHHHHHH'
i = 0
while seq_fasta[len(seq_fasta)-i-1] == 'H':
    i += 1
seq_clean = seq_fasta[:len(seq_fasta)-i]

# this is important to set it right, otherwise the run will be interrupted with the FunctionTimeOut error
timeout = 86000  # seconds - this is now set to almost 24h

# Define a Modal app using the custom container image
app = modal.App("esm3_model", image=docker_image)

@app.function()
def get_sample_protein():

    '''
    This function is used to get a base ESMProtein object that is prepared with the relevant annotations and data
    which can be used as an initial state for downstream prompting and generation.
    '''

    egfr_nano_chain = ProteinChain.from_rcsb(pdb_id, chain_id)
    # convert it to ESMProtein object to enable function annotation
    protein = ESMProtein.from_protein_chain(egfr_nano_chain)

    modelled_res = egfr_nano_chain.residue_index_no_insertions  # numbering starts from 1; adjust later for python-friendly numbering
    seq = egfr_nano_chain.sequence  # seq from the pdb

    # CDRs are retrieved from SabDaB 'https://www.google.com/url?q=https%3A%2F%2Fopig.stats.ox.ac.uk%2Fwebapps%2Fsabdab-sabpred%2Fsabdab%2Fcdrsearch%2F%3Fpdb%3D4KRO%26CDRdef_pdb%3DChothia'
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

    # Binding sites and heterodimer interface are extracted from 'https://www.ebi.ac.uk/interpro/result/InterProScan/iprscan5-R20241015-221717-0261-42684762-p1m/'
    # Note that for the function annotation the start and end positions must be 1-indexed as per esm3 github.
    # Keywords should be compatible with the allowed function annotations, either existing InterPro tags or keyword. This requires
    # some careful mining of the relevant tokenizers
    binding_sites = [{
        'val': 33,
        'keyword': 'antigen binding',  # 'CDD - cd04981',
        }, {
        'val': 50,
        'keyword': 'antigen binding',  # 'CDD - cd04981',
        }, {
        'val': 99,
        'keyword': 'antigen binding',  # 'CDD - cd04981',
        }]

    h_dimer_interface = [{
        'val': 39,
        'keyword': 'interface',  # 'CDD - cd04981'
        },
        {
        'val': 43,
        'keyword': 'interface',  # 'CDD - cd04981'
        },
        {
        'val': 47,
        'keyword': 'interface',  # 'CDD - cd04981'
        },
        {
        'val': 95,
        'keyword': 'interface',  # 'CDD - cd04981'
        },
        {
        'val': 120,
        'keyword': 'interface',  # 'CDD - cd04981'
        },
        {
        'val': 121,
        'keyword': 'interface',  # 'CDD - cd04981'
        }]

    # construct the prompts

    _sites_bind = [
        FunctionAnnotation(label=binding_site['keyword'], start=binding_site['val'], end=binding_site['val']+1)
        for binding_site in binding_sites
        ]

    _sites_interface = [
        FunctionAnnotation(label=interface_site['keyword'], start=interface_site['val'], end=interface_site['val']+1)
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
    egfr_nano_chain = ProteinChain.from_rcsb(pdb_id, chain_id)
    modelled_res = egfr_nano_chain.residue_index_no_insertions
    # convert it to ESMProtein object to enable function annotation
    egfr_nano_chain = ESMProtein.from_protein_chain(egfr_nano_chain)
    seq = egfr_nano_chain.sequence  # seq from the pdb that will likely be different to the seq_fasta as there can be unmodelled residues.

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

    # Binding sites and interface are based on seq_fasta; posistions are python friendly, i.e. 0 based
    # This is slightly different than the function annotation that goes into the ESMProtein object.
    # The reason for this is that we use these positions to mask out the sequence, hence the 0-based positions.
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

    # SEEDS
    seeds = np.random.randint(0, 10**5, size=NUM_SEEDS)

    metrics_dict = {
            'tag': [],
            'seq_seq_gen': [],
            'seq_struct_gen': [],
            'seq_inv_fold_gen': [],
            'sequence_ptm': [],
            'sequence_plddt': [],
            'structure_ptm': [],
            'structure_plddt': [],
            'inv_fold_ptm': [],
            'inv_fold_plddt': [],
        }

    for _nr, seed in enumerate(seeds):
        print(f"generating nr: {_nr} and seed: {seed}")
        # populate the round with dummy values, as the generation steps may fail, but we still want to keep track of it
        metrics_dict['tag'].append(str(seed))
        metrics_dict['seq_seq_gen'].append(None)
        metrics_dict['seq_struct_gen'].append(None)
        metrics_dict['seq_inv_fold_gen'].append(None)
        metrics_dict['sequence_ptm'].append(None)
        metrics_dict['sequence_plddt'].append(None)
        metrics_dict['structure_ptm'].append(None)
        metrics_dict['structure_plddt'].append(None)
        metrics_dict['inv_fold_ptm'].append(None)
        metrics_dict['inv_fold_plddt'].append(None)

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
            # update the metrics
            metrics_dict['seq_seq_gen'].pop()
            metrics_dict['seq_seq_gen'].append(sequence_generation.sequence)
            metrics_dict['sequence_ptm'].pop()
            metrics_dict['sequence_ptm'].append(sequence_generation.ptm.numpy().item())
            metrics_dict['sequence_plddt'].pop()
            metrics_dict['sequence_plddt'].append(sequence_generation.plddt.numpy())

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
            metrics_dict['seq_struct_gen'].pop()
            metrics_dict['seq_struct_gen'].append(structure_prediction.sequence)
            metrics_dict['structure_ptm'].pop()
            metrics_dict['structure_ptm'].append(structure_prediction.ptm.numpy().item())
            metrics_dict['structure_plddt'].pop()
            metrics_dict['structure_plddt'].append(structure_prediction.plddt.numpy())

            struct_gen_file_name = f"{file_tag}_tag_{seed}_struct_gen.pdb"
            struct_gen_pdb_path = Path(MOUNT_DIR) / struct_gen_file_name
            structure_prediction.to_pdb(str(struct_gen_pdb_path))

            _cdr_idxs = [idx+1 for idx, res in enumerate(seq_prompt) if res=='_']

            # as an fun side - count the number of differring residues - this is not used later
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
            # update the metrics
            metrics_dict['seq_inv_fold_gen'].pop()
            metrics_dict['seq_inv_fold_gen'].append(inv_folded_protein.sequence)
            metrics_dict['inv_fold_ptm'].pop()
            metrics_dict['inv_fold_ptm'].append(inv_folded_protein.ptm.numpy().item())
            metrics_dict['inv_fold_plddt'].pop()
            metrics_dict['inv_fold_plddt'].append(inv_folded_protein.plddt.numpy())

            inv_fold_file_name = f"{file_tag}_tag_{seed}_inv_fold.pdb"
            inv_fold_pdb_path = Path(MOUNT_DIR) / inv_fold_file_name
            inv_folded_protein.to_pdb(str(inv_fold_pdb_path))

            volume.commit()

        except AssertionError as e:
            print(f"{e} in round {_nr}")
            continue

    # save the metrics
    df = pd.DataFrame(metrics_dict)
    csv_metrics_path = Path(MOUNT_DIR) / run_id_csv_tag
    df.to_csv(str(csv_metrics_path))


@app.local_entrypoint()
def main():
    # Run the model and save PDBs
    run_model_and_save_pdbs.remote()

if __name__ == "__main__":
    main()