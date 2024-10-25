import random
import numpy as np
import torch
import torch.nn.functional as F
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from peft import LoraConfig, get_peft_model
from torch import nn
from scipy import stats
from sklearn import metrics
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import EvalPrediction, Trainer, TrainingArguments

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class SingleMutationDatasetV2(Dataset):
    def __init__(self, hf_ds, swap=True):
        self.hf_ds = hf_ds
        self.swap = swap

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        wt_seq = self.hf_ds["wt_seq"][idx]
        mut_seq = self.hf_ds["mut_seq"][idx]
        pos = self.hf_ds["pos"][idx]
        label = self.hf_ds["ddg"][idx]

        if self.swap and random.random() > 0.50:
            wt_seq, mut_seq = mut_seq, wt_seq
            label = -label

        return {
            "wt_seq": wt_seq,
            "mut_seq": mut_seq,
            "position": pos,
            "label": label,
        }


class SingleMutationPooler(nn.Module):
    def __init__(self, embed_dim=1536):
        super().__init__()
        self.wt_weight = nn.Parameter(
            torch.ones((1, embed_dim)), requires_grad=True
        )
        self.mut_weight = nn.Parameter(
            -1 * torch.ones((1, embed_dim)), requires_grad=True
        )
        self.norm = nn.LayerNorm(embed_dim, bias=False)

    def forward(self, wt_embedding, mut_embedding, positions):
        embed_shape = wt_embedding.shape[-1]
        positions = (
            positions.view(-1, 1).unsqueeze(2).repeat(1, 1, embed_shape) + 1
        )
        wt_residues = torch.gather(wt_embedding, 1, positions).squeeze(1)
        mut_residues = torch.gather(mut_embedding, 1, positions).squeeze(1)
        wt_residues = wt_residues * self.wt_weight
        mut_residues = mut_residues * self.mut_weight
        return self.norm(wt_residues + mut_residues)


class StabilityPrediction(nn.Module):
    def __init__(self, backbone, embed_dim=1536):
        super().__init__()
        self.backbone = backbone
        self.pooler = SingleMutationPooler()
        self.regressor = nn.Linear(embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.regressor.weight, -0.01, 0.01)
        nn.init.zeros_(self.regressor.bias)

    def compute_loss(self, logits, labels):
        if labels is None:
            return
        return F.mse_loss(logits, labels)

    def forward(self, wt_input_ids, mut_input_ids, positions, labels=None):
        wt_embeddings = self.backbone(sequence_tokens=wt_input_ids).embeddings
        mut_embeddings = self.backbone(
            sequence_tokens=mut_input_ids
        ).embeddings
        aggregated_embeddings = self.pooler(
            wt_embeddings, mut_embeddings, positions
        )
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

        wt_input_ids = self.tokenizer(wt_seqs, return_tensors="pt")[
            "input_ids"
        ]
        mut_input_ids = self.tokenizer(mut_seqs, return_tensors="pt")[
            "input_ids"
        ]
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
    spearman = stats.spearmanr(logits, labels).statistic
    mae = metrics.mean_absolute_error(labels, logits)
    mse = metrics.mean_squared_error(labels, logits)
    pearson = stats.pearsonr(logits, labels).statistic[0]
    rmse = metrics.root_mean_squared_error(labels, logits)
    return {
        "spearman": spearman,
        "mae": mae,
        "mse": mse,
        "pearson": pearson,
        "rmse": rmse,
    }


def main():
    tokenizer = EsmSequenceTokenizer()
    model = ESM3_sm_open_v0("cuda")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        bias="none",
        use_dora=True,
        target_modules=["layernorm_qkv.1", "ffn.1", "ffn.3"],
    )
    peft_backbone = get_peft_model(model, lora_config)

    downstream_model = StabilityPrediction(peft_backbone).cuda()

    training_args = TrainingArguments(
        output_dir="stability_weights_dora_norm_swap_v2",
        num_train_epochs=20,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=400,
        learning_rate=1e-04,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        logging_dir="stability_dora_norm_logs_swap_v2",
        logging_steps=10,
        logging_strategy="steps",
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        gradient_accumulation_steps=16,
        fp16=False,
        fp16_opt_level="02",
        run_name="stability_dora_norm_swap_v2_experiment",
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_validation_spearman",
        greater_is_better=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="wandb",
        save_total_limit=1,
    )

    ds = load_dataset(
        "hazemessam/ddg",
        data_files={
            "train": "S2648.csv",
            "validation": "ssym.csv",
            "test": "ssym_r.csv",
        },
    )

    collator = Collator(tokenizer)
    train_ds = SingleMutationDatasetV2(ds["train"], swap=True)
    eval_ds = SingleMutationDatasetV2(ds["validation"], swap=False)
    test_ds = SingleMutationDatasetV2(ds["test"], swap=False)

    test_datasets = load_dataset(
        "hazemessam/ddg",
        data_files={
            "train": "myoglobin.csv",
            "validation": "myoglobin_r.csv",
            "test": "p53.csv",
        },
    )

    myoglobin_ds = SingleMutationDatasetV2(test_datasets["train"], swap=False)
    myoglobin_r_ds = SingleMutationDatasetV2(
        test_datasets["validation"], swap=False
    )
    p53_ds = SingleMutationDatasetV2(test_datasets["test"], swap=False)

    trainer = Trainer(
        model=downstream_model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset={
            "validation": eval_ds,
            "test": test_ds,
            "myoglobin": myoglobin_ds,
            "myoglobin_r": myoglobin_r_ds,
            "p53": p53_ds,
        },
    )

    trainer.train()


if __name__ == "__main__":
    main()
