import modal
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding
from constants import DEFAULT_BQ_SCREEN_RESULTS_TABLE, ESM3_MODEL, ESM3_OPEN_MODEL

import os
import torch
import json
import base64
import numpy as np

# Build image and dependencies. Will be cached after build
esm3_image = (
    modal.Image.debian_slim(python_version="3.12").pip_install("torch",  "pandas", "esm", "huggingface")
    .apt_install("libopenblas-dev", "git")
    .workdir("/app"))  #.pip_install("esm@https://github.com/evolutionaryscale/esm.git")
    
model_volume = modal.Volume.from_name("esm3-model-vol", create_if_missing=True)
    
app = modal.App(name="ESM3", image=esm3_image)






def log_sum_exp(logits):
    import base64
    import numpy as np
    """Applies the log-sum-exp trick for numerical stability"""
    # Accepts a 1-dim logits vector
    a_max = np.max(logits)
    return a_max + np.log(np.sum(np.exp(logits - a_max)))

def calculate_sequence_max_log_probability(logits, input_ids):
    import base64
    import numpy as np
    """
    Computes the sequence log-probability using the provided logits and input tokens.

    logits: Tensor of shape (sequence_length, vocab_size)
    input_ids: Tensor of token ids for the input sequence (sequence_length)
    tokenizer: Tokenizer for decoding tokens
    """
    total_log_probability = 0
    # Token IDs to exclude (special tokens)
    allowed_token_ids = {5, 10, 17, 13, 23, 16,  6,  9, 21, 12,  4, 15, 20, 18, 14,  8, 11, 22, 19,  7}  # Only count AA tokens
    for index, token_id in enumerate(input_ids):
        if token_id.item() not in allowed_token_ids:
            # Skip special tokens
            continue
        token_logits = logits[index].cpu().numpy()
        selected_logit = token_logits[token_id.item()]

        # Normalize the selected logit against the logits distribution to compute the log probability
        log_probability = selected_logit - log_sum_exp(token_logits)
        total_log_probability += log_probability

    return total_log_probability





class TextDataset(Dataset):
    def __init__(self, texts, methods, metadata):
        self.texts = texts
        self.methods = methods  # List of strings (methods)
        self.metadata = metadata  # List of JSON objects (metadata)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Return text, methods, and metadata for each entry
        return self.texts[idx], self.methods[idx], self.metadata[idx]


# attach gpu if using local model
@app.cls(
    image=esm3_image,
    timeout=24*60*60,
    volumes={"/vol": model_volume},  # Mount the volume at "/vol"
    secrets=[
        modal.Secret.from_name("gcp-biolm-hackathon-bq-secret"),
        modal.Secret.from_name("hf-api-token"),
        modal.Secret.from_name("esm3-api-token")
    ],  # Include the GCP BQ secret
)
class Model:
    @modal.build()  # add another step to the image build
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download, login
        login(os.environ["HF_TOKEN"], add_to_git_credential=True)
        os.makedirs("/app/esm3", exist_ok=True)
        #snapshot_download("esm3-open", local_dir="/app/esm3")  # have to authenticate with HF set HF_TOKEN in hf-api-token secret

    @modal.enter()
    def setup(self):

        from esm.models.esm3 import ESM3
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
        from esm.sdk import client

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if ESM3_OPEN_MODEL:
            self.model = ESM3.from_pretrained("esm3-open").to(self.device)
        else:
            self.model = client(ESM3_MODEL, token=os.environ["ESM3_API_TOKEN"])
        
    def write_to_bigquery(self, results, table_id):
        # Insert data into BigQuery
        errors = self.bq_client.insert_rows_json(table_id, results)  # results is a list of dicts

        if errors:
            print(f"Encountered errors while inserting rows: {errors}")
        else:
            print(f"Inserted {len(results)} rows into {table_id}")

    @modal.method()
    def inference_logits(self, seqs, subproject,
        username, methods, metadatas, batch_size = 256, write_to_bq=False, 
        bq_table_name=DEFAULT_BQ_SCREEN_RESULTS_TABLE):


        
        results = []
        for seq, method, metadata in zip (seqs, methods, metadatas):
            import esm
            from esm.models.esm3 import ESM3
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
            protein = ESMProtein(sequence=seq)
            protein_tensor = self.model.encode(protein)
            logits_output = self.model.logits(protein_tensor, LogitsConfig(return_embeddings=True, sequence=True))
            assert isinstance(
                logits_output, LogitsOutput
            ), f"LogitsOutput was expected but got {logits_output}"
            assert (
                logits_output.logits is not None and logits_output.logits.sequence is not None
            )
       
            log_prob = calculate_sequence_max_log_probability(logits_output.logits.sequence, protein_tensor.sequence, )
            embedding = logits_output.embeddings.squeeze().mean(dim=0)  # mean pooling
            embedding_bytes = base64.b64encode(
                np.array(embedding, dtype=np.float32).tobytes()
            ).decode("utf-8")

            result = {
            "sequence": seq,
            "subproject": subproject,  # Particular hackathon subproject
            "username": username,      # Identifier for the participant submitting the sequence
            "method": method,          # Sequence generation method used (e.g., "mpnn")
            "method_metadata": metadata,  # JSON string of metadata associated with the generation method
            "sequence_log_probability": log_prob,
            "embedding_bytes": embedding_bytes,
            "embedding_metadata":ESM3_MODEL
        }
            results.append(result)

        if write_to_bq:
            # Write results to BigQuery
            self.write_to_bigquery(results, bq_table_name)
        else:
            print("write_to_bq is False; results not written to BigQuery.")

        print(results)
        return results


def screen_sequences():

    subproject = "example-subproject"  # The particular hackathon subproject
    username = "example-user"          # Identifier for the hackathon participant
    methods = ["example-oracle"]*2        # The sequence generation method used
    metadatas = [json.dumps({
        "example-param1": "example-value1",
        "example-param2": "example-value2"
    })]*2  # JSON string of metadata associated with the generation method

    write_to_bq = False  # Set to False for testing without writing to BigQuery

    with modal.enable_output():
        with app.run():
            # Instantiate the Model class
            esm3_model = Model()
            # Call the screen_sequences method
            result = esm3_model.inference_logits.remote(
                ["QLVSGPEVKKPASVKVSCKASGYIFNNYGISWVRQAPGQGLEWMGWISTDNGNTNYAQKVQGRVTMTTDTSTSTAYMELRSLRYDDTAVYYCANNWGSYFEHWGQGTLVTVSS",  # expected lp -52.24681282043457
                "QLVSGPEVKKPASVKVSCKASGYIFNNYGISWVRQAPGQGLEWMGWISTDNGNTNYAQKVQGRVTMTTDTSTSTAYMELRSLRYDDTAVYYCANNWGSYFEHWGQGTLVTVSS"],
                subproject=subproject,
                username=username,
                methods=methods,
                metadatas=metadatas,
                write_to_bq = write_to_bq
            )
            print(result)


if __name__ == "__main__":
    # example_forward_pass()
    screen_sequences()