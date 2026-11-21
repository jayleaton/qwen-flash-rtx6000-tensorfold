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
thinking budgets and scorer, greedy. "FP8, earlier binary" ran before the FP8 expert kernel's K-split (patch 0033),
which changes FP8's sum order and so its tokens; stack E is the profile in `config/cuda/`. The HumanEval+ and BFCL runs
use `--prompt-cache-gib 0` and 8 requests at a 33,024-token window, the same for every FP8 row.

| pack | cards | HumanEval+ (thinking 1024) | BFCL held-out 450 (thinking 512) | top-1 / KL against FP8 | text NLL |
| --- | --- | --- | --- | --- | --- |
| **FP8, stack E (this recipe)** | 2x 6000, TP2 | 153/164 = 93.3 | 336/450 = 74.7 | | |
| FP8, earlier binary | 2x 6000, TP2 | **155/164 = 94.5** | 333/450 = 74.0 | reference | 2.144 |
| NVFP4 (`nvidia/...-NVFP4@fc694b54`) | 1x 6000, TP1 | 154/164 = 93.9 | 328/450 = 72.9 | 86.4% / 0.104 | 2.169 |
| FP4-Local (`nvidia/...-FP4-Local@b9ce18a6`) | 1x 6000, TP1 | 152/164 = 92.7 | **341/450 = 75.8** | 85.0% / 0.125 | **2.143** |

- The HumanEval+ and BFCL spreads (3 problems, 18 rows across the packs; 2 problems and 3 rows between stack E and the
  earlier FP8 binary) are within run noise; only fidelity separates the packs.
- Stack E's BFCL by category (earlier FP8 binary in brackets): irrelevance 74/100 (77), live irrelevance 73/100 (73),
  live multiple 77/100 (76), live simple 76/100 (77), parallel multiple 36/50 (30). Outcomes: 336 right, 14 wrong call,
  43 names only, 53 called, 4 no call (333, 20, 43, 50, 4).
- Stack E's fidelity and NLL columns were not re-measured.
- Fidelity: TensorFold's `flashnext-fidelity-rank` (FP8, TP2) and `flashnext-fidelity-single` (4-bit, TP1), 32 fixed
  histories x 65 scored rows (2,080 rows), `TF_FLASHNEXT_CHUNK=32`. The 4-bit rows match a DGX Spark's to every printed
  digit, so the hardware does not move them.

## OpenCode

OpenCode core waves: 2 waves of 5 workflows, thinking 4096, the same tasks and grading for every pack.

| pack | window x streams | result | wave walls | tool calls |
| --- | --- | --- | --- | --- |
| FP8, stack E | 262,144 x 5 (`TF_FLASHNEXT_KV_HOST=overflow`) | **9/10** (python 6/6, games 3/4) | 358 s + 548 s | 190, 0 errors |
| FP8, stack E | 131,072 x 5 (on the cards) | **10/10** (python 6/6, games 4/4) | 437 s + 1,033 s | 215, 0 errors |
| FP4-Local | 262,144 | 10/10 | 795 s | |
| NVFP4 | 262,144 | 9/10 | 1,083 s | |

- The 262,144 rows compare directly: the 4-bit packs ran at that window too, on other hosts, so wall times compare only
  roughly.
- At 256K the one miss is a snake game: `snake.html` parses, but the play check finds the snake never
  moves. That is the model's output, not a serving fault; the same workflow passes in its other waves.
- Neither stack E run hit the output limit, and neither rank logged an error.
