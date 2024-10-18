# app.py

import os
import modal
from constants import DEFAULT_BQ_SCREEN_RESULTS_TABLE
import pandas as pd


app = modal.App("esm2-inference-app")

# Create or retrieve the volume, ensure it is created if missing
model_volume = modal.Volume.from_name("esm2-model-vol", create_if_missing=True)

image = modal.Image.from_registry(
    "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
).pip_install(
    "sentence-transformers==2.2.2",
    "https://github.com/facebookresearch/esm/archive/2b369911bb5b4b0dda914521b9475cad1656b2ac.zip",
    "numpy",
    "google-cloud-bigquery==3.26.0",
    "pandas"
)


@app.cls(
    image=image,
    gpu="L4",  # Use L4 GPU
    timeout=600,
    volumes={"/vol": model_volume},  # Mount the volume at "/vol"
    secrets=[
        modal.Secret.from_name("gcp-biolm-hackathon-bq-secret")
    ],  # Include the GCP BQ secret
)
class ESMModel:
    """
    ESMModel class that handles loading the ESM2 model, running inference,
    and screening sequences.
    """
    # Build-time method
    @modal.build()
    def build(self):
        # Build-time setup if needed
        pass

    # Container startup method
    @modal.enter()
    def enter(self):
        # This method is called when the container starts
        # Load the model here
        self.setup_model()

    def setup_model(self):
        # Load the ESM2 model
        import torch
        import esm

        # Path to the model cache directory in the volume
        model_cache_dir = "/vol/torch_cache"
        os.makedirs(model_cache_dir, exist_ok=True)

        # Set the Torch Hub cache directory to the volume
        torch.hub.set_dir(model_cache_dir)

        # Load the model and alphabet once when the container starts
        print("Loading ESM2 model...")
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.cuda()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

        self.aa_alphabet = list("LAGVSERTIDPQKNFYMHWC")  # Ordering of amino-acids in ESM2

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

    @modal.method()
    def forward_pass(
        self,
        sequences,
        repr_layers=[33],
        include=["mean"],
        max_sequence_len=2048,
    ):
        """
        Perform inference on a list of sequences using the ESM2 model.

        Parameters:
        - sequences (List[str]): A list of amino acid sequences to process.
        - repr_layers (List[int]): List of layer indices to extract representations from.
            Valid values are integers from 0 to 33 (since the model has 33 layers).
            Negative indices can also be used to refer to layers from the end (-1 refers to the last layer).
        - include (List[str]): List of output types to include in the results.
            Options are:
                - "mean": Mean representation of tokens (default).
                - "per_token": Per-token representations.
                - "bos": Beginning-of-sequence representation.
                - "contacts": Predicted inter-residue distances (contacts).
                - "logits": Predicted per-token logits.
                - "attentions": Self-attention weights.
        - max_sequence_len (int): Maximum sequence length. Sequences longer than this will be truncated.
        """
        import torch
        from esm.data import FastaBatchedDataset

        ## ESM2 params explained:

        # This is max number of tokens/chars per batch; May help OOM errors to lower
        toks_per_batch = 4096

        # This is maximum length for each sequence; Longer sequences are truncated to this length
        truncation_seq_length = max_sequence_len  # Default is 2048

        # extra_toks_per_seq: Used to add buffer tokens per sequence if needed, e.g., <cls> and <eos> tokens.
        # Since alphabet.{prepend_bos, append_eos} are True, we set extra_toks_per_seq = 2.
        # ESM2 paper: "We used BOS and EOS tokens to signal the beginning and end of a real
        # protein, to allow the model to separate a full-sized protein from a cropped one."
        extra_toks_per_seq = 2

        ## Process inputs
        # Determine if contacts or attentions need to be returned
        return_contacts = "contacts" in include or "attentions" in include

        n_max_layers = self.model.num_layers
        # Validate repr_layers to ensure they are within valid bounds
        if not all(-(n_max_layers + 1) <= i <= n_max_layers for i in repr_layers):
            raise ValueError(
                f"Requested representation layers are out of bounds. Ensure the "
                f"layer indices are between -{n_max_layers + 1} and {n_max_layers}."
            )
        # Convert negative layer indices to positive indices
        repr_layers = [(i + n_max_layers + 1) % (n_max_layers + 1) for i in repr_layers]

        # Create dataset and batches
        sequence_ids = [str(i) for i in range(len(sequences))]
        dataset = FastaBatchedDataset(sequence_ids, sequences)
        batches = dataset.get_batch_indices(
            toks_per_batch, extra_toks_per_seq=extra_toks_per_seq
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=self.alphabet.get_batch_converter(truncation_seq_length),
            batch_sampler=batches,
        )

        results = []
        for batch_idx, (
            sequence_labels,
            sequence_strings,
            tokenized_sequences,
        ) in enumerate(data_loader):
            # batch_idx: Index of the current batch within the data_loader iteration.
            # sequence_labels: List/Batch of labels for all sequences in the current batch.
            # sequence_strings: List/Batch of raw amino acid sequences in string format.
            # tokenized_sequences: Tensor containing tokenized sequences ready for model input.

            print(
                f"Processing batch {batch_idx + 1} of {len(batches)} "
                f"({tokenized_sequences.size(0)} sequences in this batch)"
            )

            tokenized_sequences = tokenized_sequences.cuda()
            with torch.no_grad():
                model_output = self.model(
                    tokenized_sequences,
                    repr_layers=repr_layers,
                    return_contacts=return_contacts,
                )

            ## Process 'logits'
            # Note: has 2 extra tokens <cls>, <eos>; shape: (batch_size, seq_len+2, num_tokens)
            logits = model_output.get("logits", None)
            if logits is not None:
                logits = logits.cpu()

            ## Process 'representations'
            # Note: has 2 extra tokens <cls>, <eos>; shape: (batch_size, seq_len+2, representation_dim)
            representations = {
                layer: r.cpu() for layer, r in model_output["representations"].items()
            }

            ## Process 'contacts'
            # Note: does NOT have extra tokens; shape: (batch_size, seq_len, seq_len)
            contacts = model_output.get("contacts", None)
            if contacts is not None:
                contacts = contacts.cpu()

            ## Process 'attentions'
            # Note: has extra tokens; shape: (batch_size, num_layers, num_heads, seq_len+2, seq_len+2)
            attentions = model_output.get("attentions", None)
            if attentions is not None:
                attentions = attentions.cpu()

            for i, label_ in enumerate(sequence_labels):
                result_dict = {"sequence_index": label_}

                # In case sequence was truncated, we only want to return the non-truncated part.
                # However, this should always be the full sequence length, since we use
                # validation to enforce sequence length <= max_sequence_len.
                truncate_len = min(truncation_seq_length, len(sequence_strings[i]))

                # Call clone on tensors to ensure they are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_token" in include:
                    result_dict["representations"] = {
                        # These indices remove <cls> and <eos> tokens
                        str(layer_n): t[i, 1 : truncate_len + 1].clone().tolist()
                        for layer_n, t in representations.items()
                    }
                if "mean" in include:
                    result_dict["mean_representations"] = {
                        # These indices remove <cls> and <eos> tokens
                        str(layer_n): t[i, 1 : truncate_len + 1]
                        .mean(dim=0)
                        .clone()
                        .tolist()
                        for layer_n, t in representations.items()
                    }
                if "bos" in include:
                    result_dict["bos_representations"] = {
                        # Grabs the <cls> token (which is same as <bos> token)
                        str(layer_n): t[i, 0].clone().tolist()
                        for layer_n, t in representations.items()
                    }
                if "contacts" in include and contacts is not None:
                    result_dict["contacts"] = (
                        contacts[i, :truncate_len, :truncate_len].clone().tolist()
                    )
                if "logits" in include and logits is not None:
                    # The indices below remove <cls> and <eos> tokens
                    # Returns the logits for each token
                    result_dict["logits"] = (
                        logits[i, 1 : truncate_len + 1, 4:-9].clone().tolist()
                    )
                if "attentions" in include and attentions is not None:
                    # The indices below remove <cls> and <eos> tokens
                    # Attentions shape: (batch_size, num_layers, num_heads, seq_len+2, seq_len+2)
                    # Averaging over layers and heads
                    avg_attentions = (
                        attentions[i]
                        .clone()
                        .mean(dim=1)
                        .mean(dim=1)[:, 1 : truncate_len + 1]
                    )
                    result_dict["attentions"] = avg_attentions.tolist()

                # Optionally include the sequence
                result_dict["sequence"] = sequence_strings[i]

                results.append(result_dict)

        # Optionally sort results
        results = sorted(results, key=lambda x: int(x["sequence_index"]))

        return results

    @modal.method()
    def screen_sequences(
        self,
        sequences,
        subproject,
        username,
        methods,
        method_metadatas,
        write_to_bq=False,
        bq_table_name=DEFAULT_BQ_SCREEN_RESULTS_TABLE,
    ):
        """
        Screen a list of sequences using the ESM2 model and compute sequence log-probabilities.
        Optionally write results to BigQuery.

        Parameters:
        - sequences (List[str]): A list of amino acid sequences to process.
        - subproject (str): Identifier for the hackathon subproject that the sequences are associated with.
        - username (str): Identifier for the hackathon participant who is submitting the sequences.
        - method (str): The sequence generation method used (e.g., "mpnn").
        - method_metadata (str): JSON string of metadata associated with the generation method.
            This can be an empty dict or None.
        - write_to_bq (bool): If True, write the results to BigQuery. Set to False to test the function without writing to BigQuery.
        - bq_table_name (str): BigQuery table name where results should be written. Defaults to DEFAULT_BQ_SCREEN_RESULTS_TABLE.
            Should remain the default unless there is another use-case requiring writing to a separate table.
        """
        import base64
        import numpy as np

        def log_sum_exp(logits):
            """Applies the log-sum-exp trick for numerical stability"""
            # Accepts a 1-dim logits vector
            a_max = np.max(logits)
            return a_max + np.log(np.sum(np.exp(logits - a_max)))

        def calculate_sequence_max_log_probability(logits_list, sequence_tokens):
            """
            Computes the sequence log-probability using the provided logits and sequence tokens.
            """

            total_log_probability = 0

            for index, logits in enumerate(logits_list):
                if sequence_tokens[index] == "<mask>":
                    # For masked positions, select the logit associated with the maximum prediction
                    selected_logit = np.max(logits)
                else:
                    # For unmasked positions, select the logit associated with the wildtype residue
                    wildtype_residue = sequence_tokens[index]
                    wildtype_index = self.aa_alphabet.index(wildtype_residue)
                    selected_logit = logits[wildtype_index]

                # Normalize the selected logit against the logits distribution to compute the log probability
                log_probability = selected_logit - log_sum_exp(logits)
                total_log_probability += log_probability

            return total_log_probability

        # Run the forward pass
        esm2_result_list = self.forward_pass.remote(
            sequences=sequences,
            repr_layers=[33],
            include=["mean", "logits"],
            max_sequence_len=2048
        )

        # Compute sequence log-probability and prepare results
        results = []
        for sequence, esm2_result, method, method_metadata in zip(sequences, esm2_result_list, methods, method_metadatas):

            # Get the per-token logits
            logits_list = esm2_result["logits"]  # Should be list of per-token logits

            # Compute the log-probability
            sequence_tokens = list(sequence)
            sequence_lp = calculate_sequence_max_log_probability(logits_list, sequence_tokens)

            # Compute the base64 encoded embedding
            embedding_f32_l33 = esm2_result["mean_representations"]["33"]
            embedding_f32_l33_bytes = base64.b64encode(
                np.array(embedding_f32_l33, dtype=np.float32).tobytes()
            ).decode("utf-8")

            # Prepare the result dictionary
            result = {
                "sequence": sequence,
                "subproject": subproject,  # Particular hackathon subproject
                "username": username,      # Identifier for the participant submitting the sequence
                "method": method,          # Sequence generation method used (e.g., "mpnn")
                "method_metadata": method_metadata,  # JSON string of metadata associated with the generation method
                "sequence_log_probability": sequence_lp,
                "embedding_bytes": embedding_f32_l33_bytes,
                "embedding_metadata": "esm2-650m-f32-l33"
            }
            results.append(result)

        if write_to_bq:
            # Write results to BigQuery
            self.write_to_bigquery(results, bq_table_name)
        else:
            print("write_to_bq is False; results not written to BigQuery.")

        return results

    def write_to_bigquery(self, results, table_id):
        # Insert data into BigQuery
        errors = self.bq_client.insert_rows_json(table_id, results)  # results is a list of dicts

        if errors:
            print(f"Encountered errors while inserting rows: {errors}")
        else:
            print(f"Inserted {len(results)} rows into {table_id}")


def example_forward_pass():
    """
    Example usage of the forward_pass() method.
    """
    # Example sequences
    sequences = [
        "KVKLEESGGGLVQTGGSLRLTCAASGRTSRSYGMGWFRKAPGKEREFVSGISWRGDSTGYADSVKGRFTISRDNAKNTVDLQMNSLKPEDTAIYYCAAAAGSAWYGTLYEYDYWGQGTQVTVSS",
        "MSLLQDVLEFNKKFVEEKKYELYETSKFPDQKMVILSCMDTRLVELLPHALNLRNGDVKIVQNAGALVSHPFGSIMRSILVAVYELQADEVCVIGHHDCGAGKLQAEPFLEKVRAQGISDEVINTIEYSMDLKQWLTGFDSVEETVQHSVETIRNHPLFLKDTPVHGLVIDPNTGKLDVVVNGYEAIENN",
        # ...
    ]

    batch_size = 8  # Adjust based on GPU RAM and sequence lengths
    batches = [
        sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)
    ]

    # Set parameters
    repr_layers = [33]  # Valid layer indices are from 0 to 33 for ESM2-650m
    include = ["mean"]  # Options for 'include' parameter
    max_sequence_len = 2048  # Adjust as needed

    # Prepare arguments for starmap
    args_list = [(batch, repr_layers, include, max_sequence_len) for batch in batches]

    with app.run():
        # Instantiate the ESMModel class
        model = ESMModel()

        # Process batches in parallel using starmap
        results = []
        for embeddings in model.forward_pass.starmap(args_list):
            results.extend(embeddings)

        # Use the results (e.g., save to a file)
        print(f"Processed {len(results)} sequences.")
        # For demonstration, print the results
        for res in results:
            print(res)


def example_screen_sequences():
    """
    Example usage of the screen_sequences() method.
    """
    import json

    sequences = [
        "KVKLEESGGGLVQTGGSLRLTCAASGRTSRSYGMGWFRKAPGKEREFVSGISWRGDSTGYADSVKGRFTISRDNAKNTVDLQMNSLKPEDTAIYYCAAAAGSAWYGTLYEYDYWGQGTQVTVSS",
        # Expected log-probability: -84.14707884
        "MSLLQDVLEFNKKFVEEKKYELYETSKFPDQKMVILSCMDTRLVELLPHALNLRNGDVKIVQNAGALVSHPFGSIMRSILVAVYELQADEVCVIGHHDCGAGKLQAEPFLEKVRAQGISDEVINTIEYSMDLKQWLTGFDSVEETVQHSVETIRNHPLFLKDTPVHGLVIDPNTGKLDVVVNGYEAIENN",
        # Expected log-probability: -119.64532
        # Add more sequences as needed
    ]

    
    subproject = "example-subproject"  # The particular hackathon subproject
    username = "example-user"          # Identifier for the hackathon participant
    method = ["example-oracle"]*2          # The sequence generation method used
    method_metadata = [json.dumps({
        "example-param1": "example-value1",
        "example-param2": "example-value2"
    })]*2  # JSON string of metadata associated with the generation method

    write_to_bq = False  # Set to False for testing without writing to BigQuery
    bq_table_name = DEFAULT_BQ_SCREEN_RESULTS_TABLE  # BigQuery table name

    with app.run():
        # Instantiate the ESMModel class
        model = ESMModel()

        # Call the screen_sequences method
        result = model.screen_sequences.remote(
            sequences,
            subproject,
            username,
            methods,
            method_metadatas,
            write_to_bq,
            bq_table_name,
        )
        print(result)


if __name__ == "__main__":
    # example_forward_pass()
    example_screen_sequences()
