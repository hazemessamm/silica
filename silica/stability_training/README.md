# Stability Training

### `stability_training.py` is a script for training ESM-3 using DoRA for predicting ddG with single mutation

### Notes

* When the mutated position is at position 3, for example, the ESM-3 tokenizer automatically adds `BOS` and `EOS` tokens. However, you should keep your original position unchanged, as the `SingleMutationPooler` class handles the necessary adjustments.

* If you want to test your model with swapping mutated sequence and wildtype sequence then you can use `SingleMutationDatasetV2` and when initializing an instance set `swap = True`
