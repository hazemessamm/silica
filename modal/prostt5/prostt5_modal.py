from curses.ascii import isupper
import json
import modal
import warnings

prostt5_image = modal.Image.micromamba(python_version="3.10").run_commands(
    "pip install transformers accelerate datasets sentencepiece biopython wget"
).apt_install("wget").run_commands("wget -q -nc https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz; tar xzf foldseek-linux-avx2.tar.gz -C /root/; export PATH=$(pwd)/foldseek/bin/:$PATH")


app = modal.App("inverse_folding_prostt5", image=prostt5_image)


def read_fasta(in_path, is_3Di):
    """
    Reads in fasta file containing a single or multiple sequences.
    Returns dictionary.
    """

    sequences = dict()
    with open(in_path, "r") as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith(">"):
                uniprot_id = (
                    line.split(" ")[0]
                    .replace(">", "")
                    .replace(".pdb", "")
                    .strip()
                )
                sequences[uniprot_id] = ""
            else:
                if is_3Di:
                    sequences[uniprot_id] += (
                        "".join(line.split()).replace("-", "").lower()
                    )  # drop gaps and cast to lower-case
                else:
                    sequences[uniprot_id] += "".join(line.split()).replace(
                        "-", ""
                    )
    return sequences




@app.function(image=prostt5_image, gpu="L4", timeout=1000)
def inverse_fold_v2(gen_kwargs, query_id, target_atom=None, compute_rmsd=True):
    import torch
    from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
    import time
    import requests
    import Bio.PDB
    from pathlib import Path
    import os

    os.mkdir(query_id)
    os.system(f"wget -q -O {query_id}/AF-{query_id}-F1-model_v4.pdb https://alphafold.ebi.ac.uk/files/AF-{query_id}-F1-model_v4.pdb")
    os.system(f"foldseek/bin/foldseek createdb {query_id}/ {query_id}/queryDB")
    os.system(f"foldseek/bin/foldseek lndb {query_id}/queryDB_h {query_id}/queryDB_ss_h")
    os.system(f"foldseek/bin/foldseek convert2fasta {query_id}/queryDB_ss {query_id}/queryDB_ss.fasta")

    os.system(f"more {query_id}/queryDB_ss.fasta")

    fasta_file = os.path.join(query_id, "queryDB_ss.fasta")
    seq_dict = read_fasta(fasta_file, is_3Di=True)

    ckpt = "Rostlab/ProstT5"

    tokenizer = T5Tokenizer.from_pretrained(
        ckpt,
        do_lower_case=False,
        legacy=True,
    )
    print("Loaded tokenizer.")
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    if torch.cuda.is_available():
        print("CUDA available.")
        model.cuda()
    model = model.eval()
    print("Loaded model.")

    generated_sequences = dict()
    for seq_idx, (fasta_id, seq) in enumerate(seq_dict.items(), 0):
        seq_len = len(seq)
        seq = (
            seq.replace("U", "X")
            .replace("Z", "X")
            .replace("O", "X")
            .replace("B", "X")
        )
        seq = " ".join(list(seq))

        max_len = seq_len + 1
        min_len = seq_len + 1

        # starting point tokens
        start_encoding = tokenizer.batch_encode_plus(
            ["<fold2AA>" + " " + seq],
            add_special_tokens=True,
            return_tensors="pt",
        )
        inputs = start_encoding.input_ids
        attention_mask = start_encoding.attention_mask
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()
        target = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=max_len,
            min_length=min_len,
            **gen_kwargs,
        )
        print(f"Generated sequence: {fasta_id}.")
        t_strings = tokenizer.batch_decode(target, skip_special_tokens=True)
        generated_sequences[f"{fasta_id}_{seq_idx}"] = t_strings

        if not compute_rmsd:
            return generated_sequences

        if target_atom is None:
            raise ValueError("Expected `target_atom` to have a value if you "
                             "want to compute RMSD.")

        for gen_seq_idx, t in enumerate(t_strings, 0):
            time.sleep(5)
            gen_seq = "".join(t.split(" "))
            gen_seq_id = fasta_id + f"__{seq_idx}" + f"__{gen_seq_idx}"

            esmfold_api_url = 'https://api.esmatlas.com/foldSequence/v1/pdb/'
            r = requests.post(esmfold_api_url, data=gen_seq, timeout=10)
            while not r.status_code == 200:
                print("Internal Server error of ESMFold API. Sleeping 6s and then trying again.")
                time.sleep(6)
                r = requests.post(esmfold_api_url, data=gen_seq, timeout=60)

            structure = r.text

            if not os.path.exists(query_id):
                os.mkdir(query_id)

            if not os.path.exists(os.path.join(query_id, "gen_seqs")):
                os.mkdir(os.path.join(query_id, "gen_seqs"))

            with open(os.path.join(query_id, "gen_seqs", f"{gen_seq_id}.pdb"), "w") as out_f:
                out_f.write(structure)
            print("Success")

    # Start the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)

    # Get the structures
    ref_structure = pdb_parser.get_structure("reference", f"{query_id}/AF-{query_id}-F1-model_v4.pdb")
    best_RMSD = None
    for idx, p in enumerate(Path(f"{query_id}/gen_seqs").rglob("*.pdb")):
        sample_structure = pdb_parser.get_structure("sample", p)
        chain_idx = p.name.split("__")[1]
        seq_idx_in_chain = p.name.split("__")[2].split(".")[0]
        fasta_id = str(p).split("__")[0].split("/")[-1]

        # Use the first model in the pdb-files for alignment
        # Change the number 0 if you want to align to another structure
        ref_model = ref_structure[0]
        sample_model = sample_structure[0]

        # Make a list of the atoms (in the structures) you wish to align.
        # In this case we use CA atoms whose index is in the specified range
        ref_atoms = []
        sample_atoms = []

        # Iterate of all chains in the model in order to find all residues
        for ref_chain in ref_model:
            # Iterate of all residues in each model in order to find proper atoms
            for ref_res in ref_chain:
                # Append CA atom to list
                ref_atoms.append(ref_res[target_atom])

        # Do the same for the sample structure
        for sample_chain in sample_model:
            for sample_res in sample_chain:
                sample_atoms.append(sample_res[target_atom])

        # Now we initiate the superimposer:
        super_imposer = Bio.PDB.Superimposer()
        super_imposer.set_atoms(ref_atoms, sample_atoms)
        super_imposer.apply(sample_model.get_atoms())
        generated_sequences[f"{fasta_id}_{chain_idx}"][int(seq_idx_in_chain)] = [generated_sequences[f"{fasta_id}_{chain_idx}"][int(seq_idx_in_chain)], super_imposer.rms]

        # Print RMSD:
        print(f'The calculated RMSD for {p} is: {super_imposer.rms}Å')
        if best_RMSD is None or best_RMSD > super_imposer.rms:
            best_RMSD = super_imposer.rms
    return generated_sequences


@app.local_entrypoint()
def main(
    query_id: str,
    target_atom: str = None,
    compute_rmsd: bool = True,
    top_p: float = 0.85,
    temperature: float = 1.0,
    top_k: int = 3,
    repetition_penalty: float = 1.2,
    num_beams: int = 1,
    num_return_sequences: int = 10,
    length_penalty: float = 1.0,
):
    if query_id.islower():
        warnings.warn("`query_id` is usually the PDB file name without the "
                      "extension which is usually in upper case. "
                      f"Received: {query_id}")
    gen_kwargs = {
        "do_sample": True,
        "top_p": float(top_p),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "repetition_penalty": float(repetition_penalty),
        "num_beams": int(num_beams),
        "num_return_sequences": int(num_return_sequences),
        "length_penalty": float(length_penalty),
    }
    output = inverse_fold_v2.remote(gen_kwargs, query_id, target_atom, compute_rmsd)
    with open(f"outputs_{query_id}.json", "w") as f:
        json.dump(output, f)
