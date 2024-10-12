# Examples

## ESM2 Example

```python
ckpt = "facebook/esm2_t33_650M_UR50D"
esm2 = ESM2(ckpt, device="cuda")
sequence = "MQMV"
lp = esm2.compute_sequence_log_probs(sequence)
embeddings = esm2.compute_sequence_embeddings(sequence, slice_cls=True, slice_eos=True)
print(lp) # tensor(-0.8124)
```

## ESM3 Example
```python
esm3 = ESM3(device="cuda")
sequence = "MQMV"
lp = esm3.compute_sequence_log_probs(sequence)
embeddings = esm3.compute_sequence_embeddings(sequence, slice_cls=True, slice_eos=True)
print(lp) # tensor(-1.7008)
```
