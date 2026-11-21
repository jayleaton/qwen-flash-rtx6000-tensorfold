# Quality

## Soak

Stack E, the profile in `config/cuda/`, 35.1 minutes of mixed-length traffic: 30 passes over 16 prompts (8K-32K
prompts among short code ones, 250 ms apart) at 1, 8 and 16 streams, greedy and sampled (temperature 0.7); every third
pass also checks each concurrent reply against its alone run and its `"draft": false` reply.

| check | result |
| --- | --- |
| requests | 1,500, 0 failed, 0 unmeasured |
| equality checks (alone == concurrent, alone == `"draft": false`) | 1,460, 0 unequal |
| NCCL errors, hangs, timeouts in either rank's log | none |
| GPU memory, idle floor minute 5 to 35 | flat at 75,055 / 72,333 MiB |
| rank host memory (RSS) | flat, 2.9 / 3.7 GiB |
| device peak under load | 83.7 GiB a rank |

## FP8 against the 4-bit packs

FP8 at TP2 on two RTX PRO 6000s against the NVIDIA 4-bit packs at TP1 on one card, with the same evaluation kit,
thinking budgets and scorer. The FP8 scores come from an earlier binary of this line (before the FP8 expert kernel's
K-split, which changes FP8's sum order and so its hashes); stack E's rows are pending.

| pack | cards | HumanEval+ (thinking 1024) | BFCL held-out 450 (thinking 512) | top-1 / KL against FP8 | text NLL |
| --- | --- | --- | --- | --- | --- |
| FP8 (this recipe) | 2x 6000, TP2 | **155/164 = 94.5** | 333/450 = 74.0 | reference | 2.144 |
| NVFP4 (`nvidia/...-NVFP4@fc694b54`) | 1x 6000, TP1 | 154/164 = 93.9 | 328/450 = 72.9 | 86.4% / 0.104 | 2.169 |
| FP4-Local (`nvidia/...-FP4-Local@b9ce18a6`) | 1x 6000, TP1 | 152/164 = 92.7 | **341/450 = 75.8** | 85.0% / 0.125 | **2.143** |

- The HumanEval+ and BFCL spreads (3 problems, 18 rows) are within run noise; only fidelity separates the packs.
- Fidelity: TensorFold's `flashnext-fidelity-rank` (FP8, TP2) and `flashnext-fidelity-single` (4-bit, TP1), 32 fixed
  histories x 65 scored rows (2,080 rows), `TF_FLASHNEXT_CHUNK=32`. The 4-bit rows match a DGX Spark's to every printed
  digit, so the hardware does not move them.

## OpenCode

| pack | run | result |
| --- | --- | --- |
| FP8, stack E | 30-minute OpenCode session, thinking 4096 | pending |
| FP8 / NVFP4 / FP4-Local | pass counts on the same task set | pending |
