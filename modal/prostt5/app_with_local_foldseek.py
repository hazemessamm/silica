import json
import modal
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import pathlib
import os
import shutil


prostt5_image = modal.Image.micromamba(python_version="3.10").run_commands(
    "pip install transformers accelerate datasets sentencepiece"
)


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

    # example = sequences[uniprot_id]

    # print("##########################")
    # print(f"Input is 3Di: {is_3Di}")
    # print(f"Example sequence: >{uniprot_id}\n{example}")
    # print("##########################")

    return sequences


@app.function(image=prostt5_image, timeout=1000)
def inverse_fold(gen_kwargs, seq_dict):
    ckpt = "Rostlab/ProstT5"

    tokenizer = T5Tokenizer.from_pretrained(
        ckpt,
        do_lower_case=False,
        legacy=True,
    )
    print("Loaded tokenizer.")
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    model = model.eval()
    print("Loaded model.")

    generated_sequences = dict()
    for seq_idx, (fasta_id, seq) in enumerate(seq_dict.items(), 1):
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
        target = model.generate(
            start_encoding.input_ids,
            attention_mask=start_encoding.attention_mask,
            max_length=max_len,
            min_length=min_len,
            **gen_kwargs,
        )
        print(f"Generated sequence: {fasta_id}.")
        t_strings = tokenizer.batch_decode(target, skip_special_tokens=True)
        generated_sequences[f"{fasta_id}_{seq_idx}"] = t_strings
    return generated_sequences


@app.local_entrypoint()
def main(
    pdb,
    foldseek_path="~/foldseek/bin/foldseek",
    top_p=0.85,
    temperature=1.0,
    top_k=3,
    repetition_penalty=1.2,
    num_beams=1,
    num_return_sequences=1,
    length_penalty=1.0,
):
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

    query_id = pathlib.Path(pdb).stem
    if pathlib.Path(query_id).exists():
        shutil.rmtree(query_id)
    pathlib.Path(query_id).mkdir(exist_ok=True)
    shutil.copy(pdb, query_id)
    db = os.path.join(query_id, "queryDB")
    dst_query_filename = os.path.join(query_id, "queryDB_ss_h")
    fasta_path = os.path.join(query_id, "queryDB_ss.fasta")
    src = os.path.join(query_id, "queryDB_h")
    src_filename = os.path.join(query_id, "queryDB_ss")

    os.system(f"{foldseek_path} createdb {query_id} {db}")
    os.system(f"{foldseek_path} lndb {src} {dst_query_filename}")
    os.system(f"{foldseek_path} convert2fasta {src_filename} {fasta_path}")
    fasta_file = os.path.join(query_id, "queryDB_ss.fasta")
    seq_dict = read_fasta(fasta_file, is_3Di=True)
    output = inverse_fold.remote(gen_kwargs, seq_dict)
    with open("outputs.json", "w") as f:
        json.dump(output, f)
