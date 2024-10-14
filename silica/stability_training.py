import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from transformers import AutoModel, AutoTokenizer
import huggingface_hub
from peft import LoraConfig, get_peft_model
from torch import nn
from scipy import stats
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import EvalPrediction, Trainer, TrainingArguments

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class SingleMutationDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return {
            "wt_seq": self.df["wt_seq"][idx],
            "mut_seq": self.df["mutant_seq"][idx],
            "position": self.df["position"][idx],
            "label": self.df["ddG_ML"][idx],
        }


class SingleMutationPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, wt_embedding, mut_embedding, positions):
        embed_shape = wt_embedding.shape[-1]
        positions = positions.view(-1, 1).unsqueeze(2).repeat(1, 1, embed_shape)
        wt_residues = torch.gather(wt_embedding, 1, positions).squeeze(1)
        mut_residues = torch.gather(mut_embedding, 1, positions).squeeze(1)
        return wt_residues + mut_residues



class StabilityPrediction(nn.Module):
    def __init__(self, backbone, embed_dim=1536):
        super().__init__()
        self.backbone = backbone
        self.pooler = SingleMutationPooler()
        self.regressor = nn.Linear(embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.regressor.weight, -0.01, 0.01)

    def compute_loss(self, logits, labels):
        if labels is None:
            return
        return F.mse_loss(logits, labels)

    def forward(self, wt_input_ids, mut_input_ids, positions, labels=None):
        wt_embeddings = self.backbone(sequence_tokens=wt_input_ids).embeddings
        mut_embeddings = self.backbone(sequence_tokens=mut_input_ids).embeddings
        aggregated_embeddings = self.pooler(wt_embeddings, mut_embeddings, positions)
        logits = self.regressor(aggregated_embeddings)
        loss = self.compute_loss(logits, labels)
        return {
            "loss": loss,
            "logits": logits,
        }


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        wt_seqs = []
        mut_seqs = []
        positions = []
        labels = []
        for b in batch:
            wt_seqs.append(b["wt_seq"])
            mut_seqs.append(b["mut_seq"])
            positions.append(b["position"])
            labels.append(b["label"])

        wt_input_ids = self.tokenizer(wt_seqs, return_tensors="pt")["input_ids"]
        mut_input_ids = self.tokenizer(mut_seqs, return_tensors="pt")["input_ids"]
        positions = torch.tensor(positions, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        return {
            "wt_input_ids": wt_input_ids,
            "mut_input_ids": mut_input_ids,
            "positions": positions,
            "labels": labels,
        }
    
def compute_metrics(p: EvalPrediction):
    logits = p.predictions
    labels = p.label_ids
    spearman_correlation = stats.spearmanr(logits, labels).correlation
    mae = metrics.mean_absolute_error(logits, labels)
    mse = metrics.mean_squared_error(logits, labels)
    return {
        "spearman_correlation": spearman_correlation,
        "mae": mae,
        "mse": mse,
    }


def main():
    tokenizer = EsmSequenceTokenizer()
    model = ESM3_sm_open_v0("cuda")

    train = pd.read_csv("k50_1_2_processed_single.tsv", sep="\t")

    valid_misfits = ['1UFM', 'v2R31S_R32S_2N5D', '2LHR', '2M8E', '2L6Q']
    validation = train[train['WT_name'].isin(valid_misfits)]
    validation = validation.reset_index()

    lora_config = LoraConfig(r=16, lora_alpha=32, bias="none", use_dora=False, target_modules=["layernorm_qkv.1"])
    peft_backbone = get_peft_model(model, lora_config)

    downstream_model = StabilityPrediction(peft_backbone).cuda()

    training_args = TrainingArguments(
        output_dir="stability_weights",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=1000,
        learning_rate=1e-04,
        weight_decay=0.0,
        logging_dir="stability_logs",
        logging_steps=10,
        logging_strategy="steps",
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
    #     eval_steps=1,
    #     save_steps=1,
        gradient_accumulation_steps=16,
        fp16=False,
        fp16_opt_level="02",
        run_name="stability_experiment",
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="spearman_correlation",
        greater_is_better=True,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    collator = Collator(tokenizer)
    train_ds = SingleMutationDataset(train)
    eval_ds = SingleMutationDataset(validation)
    trainer = Trainer(
        model=downstream_model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()


if __name__ == "__main__":
    main()
