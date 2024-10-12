import torch
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


def compute_sequence_log_probs(
    sequence,
    model,
    tokenizer,
    mask_special_tokens=True,
    mask_cls_eos=True,
    device="cuda",
):
    model.eval()
    input_ids = tokenizer(
        sequence,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"]
    input_ids = input_ids.to(device=device)

    with torch.no_grad():
        logits = model(input_ids).sequence_logits

    if mask_special_tokens:
        special_tokens = [3, 29, 31, 30, 32, 1]
        if mask_cls_eos:
            special_tokens.extend([0, 2])
        logits[:, :, special_tokens] = -torch.inf

    log_probs = torch.log_softmax(logits, dim=-1)
    gathered_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1))
    if mask_cls_eos:
        gathered_log_probs = gathered_log_probs[:, 1:-1, :]
    return gathered_log_probs.sum() / input_ids.shape[-1]


def compute_sequence_embeddings(
    sequence, model, tokenizer, slice_cls=False, slice_eos=False, device="cuda"
):
    model.eval()
    input_ids = tokenizer(
        sequence,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"]
    input_ids = input_ids.to(device=device)
    with torch.no_grad():
        embeddings = model(input_ids).embeddings
    if slice_cls:
        embeddings = embeddings[:, 1:, :]
    if slice_eos:
        embeddings = embeddings[:, :-1, :]
    return embeddings


class ESM3:
    def __init__(
        self,
        checkpoint=None,
        mask_special_tokens=True,
        mask_cls_eos=True,
        device="cuda",
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.mask_special_tokens = mask_special_tokens
        self.mask_cls_eos = mask_cls_eos
        self.tokenizer = EsmSequenceTokenizer()
        self.model = ESM3_sm_open_v0(device)
        self._special_tokens = [3, 29, 31, 30, 32, 1]
        if mask_cls_eos:
            self._special_tokens.extend([0, 2])

    def compute_sequence_log_probs(self, sequence):
        self.model.eval()
        input_ids = self.tokenizer(
            sequence,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"]

        with torch.no_grad():
            logits = self.model(input_ids).sequence_logits

        if self.mask_special_tokens:
            logits[:, :, self._special_tokens] = -torch.inf
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1))
        if self.mask_cls_eos:
            gathered_log_probs = gathered_log_probs[:, 1:-1, :]
        return gathered_log_probs.sum() / input_ids.shape[-1]

    def compute_sequence_embeddings(
        self,
        sequence,
        slice_cls=False,
        slice_eos=False,
    ):
        self.model.eval()
        input_ids = self.tokenizer(
            sequence,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"]
        with torch.no_grad():
            embeddings = self.model(input_ids).embeddings
        if slice_cls:
            embeddings = embeddings[:, 1:, :]
        if slice_eos:
            embeddings = embeddings[:, :-1, :]
        return embeddings
