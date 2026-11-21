# Weights

The recipe serves the Qwen team's own FP8 checkpoint, unchanged:
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) at revision
`7fb48d708ca6cd8eeed8f4db0433be18af96984f` (144 files, 185,563,775,685 bytes, 131 safetensors shards). There is no
NVIDIA FP8 repository for this model. No weights are included here or uploaded anywhere by this recipe.

```bash
hf download Qwen/Qwen3.8-Flash-Next-FP8 --revision 7fb48d708ca6cd8eeed8f4db0433be18af96984f --local-dir flash-fp8
scripts/verify-weights.sh flash-fp8
```

- [`config/weights/fp8.sha256`](../config/weights/fp8.sha256): the sha256 of each of the 133 LFS files (the shards,
  `model.safetensors.index.json` and `tokenizer.json`), the hashes Hugging Face lists for that revision.
- [`config/weights/fp8.blob`](../config/weights/fp8.blob): the git blob sha1 of each of the 11 small files
  (`config.json`, the chat template, the tokenizer configs, `LICENSE`, `README.md`).
- The measured copy was checked file by file against those hashes when it was downloaded, and its sizes match the list.

What the engine reads: Qwen's FP8 release converts only the routed experts (E4M3 weights with BF16 128x128
`weight_scale_inv`), which run on the block-FP8 expert kernels; the shared-expert, attention and dense projections stay
BF16. At `--tp 2` each rank loads half of the routed experts and its share of the dense projections: 71.74 GiB a rank.

The checkpoint is under the Qwen Community License 1.0 (its `LICENSE` file). If you redistribute it, follow that
license.
