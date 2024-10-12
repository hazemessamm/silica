import torch
from transformers import AutoModel
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer


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
        sequence, add_special_tokens=True, return_tensors="pt"
    )["input_ids"]
    input_ids = input_ids.to(device=device)

    with torch.no_grad():
        logits = model(input_ids=input_ids)["logits"]

    if mask_special_tokens:
        special_tokens = [1, 3, 29, 30, 31, 32]
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
        embeddings = model(input_ids=input_ids)
    embeddings = embeddings["last_hidden_state"]
    if slice_cls:
        embeddings = embeddings[:, 1:, :]
    if slice_eos:
        embeddings = embeddings[:, :-1, :]
    return embeddings


class ESM2:
    def __init__(
        self,
        checkpoint,
        mask_special_tokens=True,
        mask_cls_eos=True,
        device="cuda",
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.mask_special_tokens = mask_special_tokens
        self.mask_cls_eos = mask_cls_eos
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self._model_for_logits = None
        self._model_for_embeddings = None
        self._special_tokens = [1, 3, 29, 30, 31, 32]
        if mask_cls_eos:
            self._special_tokens.extend([0, 2])

    def compute_sequence_log_probs(self, sequence):
        if self._model_for_logits is None:
            self._model_for_logits = AutoModelForMaskedLM.from_pretrained(
                self.checkpoint
            )
            self._model_for_logits.to(device=self.device)
            self._model_for_logits.eval()

        output = compute_sequence_log_probs(
            sequence,
            self._model_for_logits,
            self.tokenizer,
            self.mask_special_tokens,
            self.mask_cls_eos,
            self.device,
        )
        return output

    def compute_sequence_embeddings(
        self, sequence, slice_cls=False, slice_eos=False
    ):
        if self._model_for_embeddings is None:
            self._model_for_embeddings = AutoModel.from_pretrained(
                self.checkpoint,
            )
            self._model_for_embeddings.to(device=self.device)
            self._model_for_embeddings.eval()

        embeddings = compute_sequence_embeddings(
            sequence,
            self._model_for_embeddings,
            self.tokenizer,
            slice_cls,
            slice_eos,
            self.device,
        )
        return embeddings
