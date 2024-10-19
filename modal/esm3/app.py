import modal
from pathlib import Path
from constants import DEFAULT_BQ_SCREEN_RESULTS_TABLE, ESM3_MODEL, ESM3_OPEN_MODEL

import os
import torch
import json
import base64
import numpy as np

# Build image and dependencies. Will be cached after build
esm3_image = (
    modal.Image.debian_slim(python_version="3.12").pip_install("torch",  "pandas", "esm", "huggingface", "google-cloud-bigquery==3.26.0")
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
        #have to authenticate with HF set HF_TOKEN in hf-api-token secret
        from huggingface_hub import snapshot_download, login
        login(os.environ["HF_TOKEN"], add_to_git_credential=True)
        os.makedirs("/app/esm3", exist_ok=True)
        
        snapshot_download("EvolutionaryScale/esm3-sm-open-v1", local_dir="/app/esm3")  # local model not tested 

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
        if ESM3_OPEN_MODEL:  ## Not working tested
            self.model = ESM3.from_pretrained("/app/esm3").to(self.device)
        else:
            self.model = client(ESM3_MODEL, token=os.environ["ESM3_API_TOKEN"])
        # Try loading BQ client
        import json
        from google.cloud import bigquery
        from google.oauth2 import service_account
        self.bq_client = None
        try:
            # Load credentials from the environment variable set by the Modal secret
            credentials_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
    
            # Create a BigQuery client
            self.bq_client = bigquery.Client(credentials=credentials, project=credentials_info["project_id"])
    
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Traceback information:")
            traceback.print_exc()  # This will print the full stack trace for debugging.
            print(f"Credentials info: {credentials_info if 'credentials_info' in locals() else 'Not loaded'}")
            print(f"BigQuery client state: {'Initialized' if self.bq_client else 'Not initialized'}")
            
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