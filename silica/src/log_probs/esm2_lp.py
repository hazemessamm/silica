import torch
from transformers import AutoModel
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer


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

        input_ids = self.tokenizer(
            sequence, add_special_tokens=True, return_tensors="pt"
        )["input_ids"]
        input_ids = input_ids.to(device=self.device)

        with torch.no_grad():
            logits = self._model_for_logits(input_ids=input_ids)["logits"]

        if self.mask_special_tokens:
            logits[:, :, self._special_tokens] = -torch.inf
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1))
        if self.mask_cls_eos:
            gathered_log_probs = gathered_log_probs[:, 1:-1, :]
        return gathered_log_probs.sum() / input_ids.shape[-1]

    def compute_sequence_embeddings(
        self, sequence, slice_cls=False, slice_eos=False
    ):
        if self._model_for_embeddings is None:
            self._model_for_embeddings = AutoModel.from_pretrained(
                self.checkpoint,
            )
            self._model_for_embeddings.to(device=self.device)
            self._model_for_embeddings.eval()

        input_ids = self.tokenizer(
            sequence,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"]
        input_ids = input_ids.to(device=self.device)
        with torch.no_grad():
            embeddings = self._model_for_embeddings(input_ids=input_ids)
        embeddings = embeddings["last_hidden_state"]
        if slice_cls:
            embeddings = embeddings[:, 1:, :]
        if slice_eos:
            embeddings = embeddings[:, :-1, :]
        return embeddings
