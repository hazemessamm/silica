import modal
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding
from constants import DEFAULT_BQ_SCREEN_RESULTS_TABLE
import os
import torch
import json
import base64
import numpy as np

# Build image and dependencies. Will be cached after build
NanoBERT = (
    modal.Image.debian_slim(python_version="3.12").pip_install("transformers", "torch", "sentence-transformers", "datasets", "accelerate", "sentence-transformers",  "pandas")
    .apt_install("libopenblas-dev")
    .run_commands("mkdir -p /app/NB2_weights")
    .workdir("/app"))
    
model_volume = modal.Volume.from_name("nanobert-model-vol", create_if_missing=True)
    
app = modal.App(name="nanoBERT", image=NanoBERT)






def log_sum_exp(logits):
    import base64
    import numpy as np
    """Applies the log-sum-exp trick for numerical stability"""
    # Accepts a 1-dim logits vector
    a_max = np.max(logits)
    return a_max + np.log(np.sum(np.exp(logits - a_max)))
def mean_pooling(embeddings, attention_mask):
    # Apply the attention mask to exclude padded tokens
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
def calculate_sequence_max_log_probability(logits, input_ids, tokenizer):
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
    excluded_token_ids = {0, 1, 2, 3, 4}  # Exclude [PAD], [s], [/s], <unk>, <mask>
    for index, token_id in enumerate(input_ids):
        if token_id.item() in excluded_token_ids:
            # Skip special tokens
            continue
        token_logits = logits[index].cpu().numpy()  # Logits for the current token position
        if token_id == tokenizer.mask_token_id:
            # For masked positions, select the logit associated with the maximum prediction
            selected_logit = np.max(token_logits)
        elif token_id not in [0,2]:
            # For unmasked positions, select the logit associated with the original token
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



@app.cls(
    image=NanoBERT,
    gpu="L4",  # Use L4 GPU
    timeout=600,
    volumes={"/vol": model_volume},  # Mount the volume at "/vol"
    secrets=[
        modal.Secret.from_name("gcp-biolm-hackathon-bq-secret")
    ],  # Include the GCP BQ secret
)
class Model:
    @modal.build()  # add another step to the image build
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download

        os.makedirs("/app/nanoBERT", exist_ok=True)
        snapshot_download("tadsatlawa/nanoBERT", local_dir="/app/nanoBERT")

    @modal.enter()
    def setup(self):
        from transformers import pipeline, RobertaTokenizer, AutoModel,AutoModelForMaskedLM
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained("/app/nanoBERT", return_tensors="pt")
        self.vocab = self.tokenizer.get_vocab()
        self.model = AutoModelForMaskedLM.from_pretrained("/app/nanoBERT").to(self.device)


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
        dataset = TextDataset(seqs, methods, metadatas)


        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for batch in dataloader:
            print(len(batch))
            raw_texts, batch_methods, batch_metadata = batch  # Unpack the text, methods, and metadata
        
            tokenized_batch = self.tokenizer(
                raw_texts,
                padding="longest",  # Pad the sequences to the longest in the batch
                truncation=True,  # Truncate sequences longer than the model's limit
                return_tensors="pt"  # Return as PyTorch tensors
            ).to(self.device)
        
            input_ids = tokenized_batch['input_ids']  # Tokenized and padded inputs
            attention_mask = tokenized_batch['attention_mask'] 


        # Perform a forward pass through the model
            with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    logits = outputs.logits  # (batch_size, sequence_length, vocab_size)
                    hidden_states = outputs.hidden_states[-1]  # Get the last layer of hidden states (embeddings)

    # Step 9: Perform mean pooling on the hidden states to get sentence embeddings
            embeddings = mean_pooling(hidden_states, attention_mask)  # Get sentence embeddings via mean pooling
    
        # Step 10: Optionally move logits and embeddings back to CPU if needed for further processing
            logits = logits.cpu()
            embeddings = embeddings.cpu()
 
            # For each input in the batch, calculate the log probability for the sequence
            for batch_idx in range(input_ids.size(0)):  # Iterate over each sample in the batch
                input_sequence = input_ids[batch_idx]
                logits_for_sequence = logits[batch_idx]  # Get logits for the current sequence
                log_prob = calculate_sequence_max_log_probability(logits_for_sequence, input_sequence, self.tokenizer)
                seq_embedding = base64.b64encode(
                np.array(embeddings[batch_idx], dtype=np.float32).tobytes()
            ).decode("utf-8")

                result = {
                "sequence": raw_texts[batch_idx],
                "subproject": subproject,  # Particular hackathon subproject
                "username": username,      # Identifier for the participant submitting the sequence
                "method": batch_methods[batch_idx],          # Sequence generation method used (e.g., "mpnn")
                "method_metadata": batch_metadata[batch_idx],  # JSON string of metadata associated with the generation method
                "sequence_log_probability": log_prob,
                "embedding_bytes": seq_embedding,
                "embedding_metadata": "nanobert"
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
    import base64
    import numpy as np
    seqs = ["QLVSGPEVKKPASVKVSCKASGYIFNNYGISWVRQAPGQGLEWMGWISTDNGNTNYAQKVQGRVTMTTDTSTSTAYMELRSLRYDDTAVYYCANNWGSYFEHWGQGTLVTVSS",  # expected lp -68.98106646537781
                "QLVSGPEVKKPASVKVSCKASGYIFNNYGISWVRQAPGQGLEWMGWISTDNGNTNYAQKVQGRVTMTTDTSTSTAYSYFEHWGQGTLVTVSS"]  # expected lp -56.54032325744629
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
            model = Model()
            # Call the screen_sequences method
            result = model.inference_logits.remote(
                seqs = seqs,
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