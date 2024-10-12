# Examples

## ESM2 Example

```python
from silica import esm2_lp
ckpt = "facebook/esm2_t33_650M_UR50D"
esm2 = esm2_lp.ESM2(ckpt, device="cuda")
sequence = "MQMV"
lp = esm2.compute_sequence_log_probs(sequence)
embeddings = esm2.compute_sequence_embeddings(sequence, slice_cls=True, slice_eos=True)
print(lp) # tensor(-0.8124)

# If you already have the model and do not
# want to use this level of abstraction (i.e. ESM2 class),
# you can use the following functions directly:

model = # your ESM2 model (should be loaded from huggingface).
tokenizer = # your ESM2 tokenizer (should be loaded from huggingface).
lp = esm2_lp.compute_sequence_log_probs(
    sequence,
    model,
    tokenizer,
    mask_special_tokens=True,
    mask_cls_eos=True,
    device="cuda",
)

embeddings = esm2_lp.compute_sequence_embeddings(
    sequence,
    model,
    tokenizer,
    slice_cls=False,
    slice_eos=False,
    device="cuda",
)
```

## ESM3 Example

```python
from silica import esm3_lp
esm3 = esm3_lp.ESM3(device="cuda")
sequence = "MQMV"
lp = esm3.compute_sequence_log_probs(sequence)
embeddings = esm3.compute_sequence_embeddings(sequence, slice_cls=True, slice_eos=True)
print(lp) # tensor(-1.7008)


# If you already have the model and do not
# want to use this level of abstraction (i.e. ESM3 class),
# you can use the following functions directly:

model = # your ESM3 model
tokenizer = # your ESM3 tokenizer
lp = esm3_lp.compute_sequence_log_probs(
    sequence,
    model,
    tokenizer,
    mask_special_tokens=True,
    mask_cls_eos=True,
    device="cuda",
)

embeddings = esm3_lp.compute_sequence_embeddings(
    sequence,
    model,
    tokenizer,
    slice_cls=False,
    slice_eos=False,
    device="cuda",
)


```
