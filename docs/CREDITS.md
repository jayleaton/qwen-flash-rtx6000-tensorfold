# Credits

- **[Qwen team](https://huggingface.co/Qwen): [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)**,
  the model and its FP8 release, at revision `7fb48d708ca6cd8eeed8f4db0433be18af96984f`. Qwen Community License 1.0.
- **[Ash Hart (ashhart) / TensorFold](https://github.com/ashhart/TensorFold)** and the TensorFold contributors: the
  engine and server. The Flash Next CUDA family, its DeltaNet, attention and expert kernels, the TP2 rank engine, the
  drafting and verify machinery, the prompt cache, the benchmarks and the kernel-contract harness are upstream work.
  Apache License 2.0; code written before 0.6.0 keeps its MIT notice. Patch 0022 is ashhart's.
- **[Scott Biggart (swbiggart)](https://github.com/swbiggart)**: the TP2 work in patches 0001-0026, from upstream PRs
  #2971 (control-link polling), #2972 (rank load wait), #2978 (blocking collectives on one host), #2980 and #2982
  (the decode-round graph with the dense split).
- **[MiaAI-Lab](https://github.com/MiaAI-Lab)**: the shared expert on a side stream (patches 0035-0036) follows patch
  0016 of MiaAI-Lab's
  [Qwen3.8-Flash-Next-RTX-PRO-6000-TernsorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-RTX-PRO-6000-TernsorFold)
  recipe (commit 57d815c2c), MIT License, Copyright (c) 2026 MiaAI-Lab. The series adds that notice to the engine's
  `THIRD_PARTY_NOTICES.md`; the MIT text is in its `LICENSES/Mia-FlashNext-MIT.txt`.
- **Upstream PR #3413**: `TF_FLASHNEXT_READOUT_FUSED` (patch 0034).
- [NCCL](https://github.com/NVIDIA/nccl) (BSD-3-Clause): the collectives the ranks run, installed by the user.
- [EvalPlus](https://github.com/evalplus/evalplus) (HumanEval+) and the
  [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla) (BFCL): the quality checks.
- [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8 (Apache License 2.0): the grammar engine behind `libtfgrammar`,
  which `scripts/build-engine.sh` builds for required and named tool calls and `response_format`.
