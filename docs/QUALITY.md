# Quality

## Exactness of patches 0077-0093 (2026-11-26)

On the host and build in [PERFORMANCE](PERFORMANCE.md#patches-0077-0093-2026-11-26):
- **The arms and cold 128K:** every cell of every arm hashes equal to R, R equals the recipe on all 8 keys, and every
  128K reply is `9a3c5d245425`.
- **OpenCode replay** (62 turns, greedy, 320 tokens a turn): the token ids of every `--pp 2` arm, with or without the
  cache and the round switches, and of `tp2-fast` with the cache, equal the 0076 `pp2-prefill`'s and the ids pinned from
  an earlier run, on every turn. There were 0 `NotAtMark` warnings.
- **Mixed traffic:** every decoder reply on `tp2-fast`, the 0076 `pp2-prefill` and the new `pp2-prefill` equals its
  solo reply, and every solo reply equals `tp2-fast`'s.
- **Soak of `tp2-overlap` with the prompt cache, 15 minutes:** STABLE.
  - 4 probes and 64 load rounds.
  - Probe hashes constant and equal to the recipe's (`153389030f94`, `147305861125`, `ddadf33e7860`).
  - 0 failed requests, no rank exit, no engine errors, code x1 mean 378.2 with 0.0% drift.
  - The new `pp2-prefill` was not soaked: the run's budget guard dropped it. The same levers with the cache were
    STABLE in a 10-minute soak on an earlier build, before 0086 and with a since-dropped switch on as well.

**`tp2-overlap` before the fix:**
- **Agent:** one OpenCode turn (of 24, one workflow at a time) parted from the reference at token 276.
- **Mixed:** one decoder's lone reply parted at token 127 in 5 of 10 runs.
- **The cause:** after serving a window's early n-gram rows, rank 1 kept the old chain's rolled history. When the next
  window started after the same two tokens, it read that window's first rows with the stale history, so the ranks ran
  the window on different rows.
  - Only lone streams use the early rows, and only rank 1 at `--tp 2` reads them this way.
  - `--pp 2` is not affected, and neither is `tp2-fast`, which does not set `TF_FLASHNEXT_PLE_EARLY`.
- **The fix, 0091:** `serve` restarts the rolled history.
- **Its host test:** serves a window, extends one after the same history and compares the rows with the n-gram
  table's own read. It fails on the tree before 0091 and passes after it.

**The GPU check** (2x RTX PRO 6000 Workstation Edition, 600 W, NODE, peer-to-peer 52.6 / 52.3 GB/s). Each arm's
replies were compared with the run above, pinned:

| arm | build | prediction | result |
| --- | --- | --- | --- |
| `tp2-fast` + early rows, x16 | before 0091 | the lone decoder's reply parts at token 127, the other 7 equal | met |
| `tp2-fast` + the overlap (no early rows), x14 | before 0091 | all 8 lone replies equal | met |
| `tp2-overlap`, x14 | 0093 | all 8 lone replies and the 32K-prompt scenario's replies equal | met |
| `tp2-overlap` + cache, x14 | 0093 | the 24 OpenCode turns equal the pinned ids | met: the turn at 276 now hashes `44e419eaaaa3`, as on `tp2-fast` |

- **What it shows:**
  - the early rows alone reproduce the fault before the fix;
  - the overlap alone is exact;
  - the fix ends both failures.
- **Quality suite:** not rerun on 0093. Every profile's tokens equal RC-tp2's, so its 213/228 stands for them.

## Soaks of `tp2-fast` and `pp2-prefill` (2026-11-24)

Each profile for 30 minutes on the Workstation Edition host in [PERFORMANCE](PERFORMANCE.md):
- every 5 minutes, the fixed probes: greedy code and chat (token hashes against the recipe's) and one cold 32K prompt;
- in between, load: code and chat at 1, 4 and 16 streams.

| check | `tp2-fast` | `pp2-prefill` |
| --- | --- | --- |
| verdict | STABLE | STABLE |
| probes, load rounds | 7, 143 | 7, 146 |
| probe hashes (code, chat, cold 32K) | constant `153389030f94`, `147305861125`, `ddadf33e7860` | the same |
| failed requests, rank exits, engine errors | 0, no, none | 0, no, none |
| code x1 mean (min-max), drift last third against first | 375.1 (355.8-378.1), -0.1% | 304.2 (299.7-305.7), +0.0% |
| code x16 mean, drift | 1,215.1, -0.1% | 1,283.9, +0.2% |

## Quality suite and kernel checks (2026-11-24)

- **The 228-item greedy suite:** RC-tp2 scores 213/228 (gsm8k 95/100, mmlu 90/100, instructions 20/20, long context 8/8)
  on both hosts measured. Every profile's probe tokens equal RC-tp2's and stack E's.
- **Not shipped:** `TF_FLASHNEXT_OUTPROJ_KSPLIT`, which changes the tokens, scores 214/228.
- **On the GPU, `tf-cuda-test route-rows`:** the spread router logits and the fused top-k and plan are byte-equal to the
  kernels they replace.
- **The kernel contract:** it passes gates 1-4 on `tf_route_logits_spread`, `fn_hcn_down_act`, `fn_hcn_up1` and the
  head's `row_best` merge.

## Soak

Stack E (`PROFILE=stack-e`), 35.1 minutes of mixed-length traffic: 30 passes over 16 prompts (8K-32K
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
which changes FP8's sum order and so its tokens; stack E is `PROFILE=stack-e`. The HumanEval+ and BFCL runs
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

## OpenCode tool calls on the profiles (2026-11-24)

The same core wave runs on the Workstation Edition host:
- 5 workflows in parallel: tetris, snake, a 3-turn todo app, numtools and bigcalc;
- a 30-minute turn cap;
- greedy (temperature 0) unless noted.

| profile | window x streams | tool calls | tool errors | `oldString` not found | `edit` without `path` | other `edit` mistakes | malformed | history compactions | graded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RC-tp2 | 49,152 x 16 | 176 | 6 (3.4%) | 4 | 1 | 1 | 0 | 9 | 5/5 |
| `tp2-fast` | 49,152 x 16 | 492 | 29 (5.9%) | 26 | 0 | 3 | 1 | 26 | 5/5 |
| `pp2-prefill` | 49,152 x 16 | 262 | 19 (7.3%) | 16 | 3 | 0 | 0 | 18 | 5/5 |
| `pp2-prefill`, sampled | 49,152 x 16 | 252 | 8 (3.2%) | 7 | 1 | 0 | 0 | 16 | 4/5 |
| **`agent`** | 139,264 x 4 | 134 | 6 (4.5%) | **1** | 2 | 3 | 0 | **0** | **5/5** |

- **Same tokens on every profile.**
  - 10 captured OpenCode requests sent again at temperature 0: the first of each workflow and one long one from each, up
    to 140 KB of history.
  - They give the same tokens, token for token, on RC-tp2, `tp2-fast` (alone and under load), `pp2-prefill` and RC-tp2
    with `--prompt-cache-gib 0`.
  - The runs above differ only because each workspace path enters the prompt.
- **`oldString` not found: the model's text, not the parser.**
  - Each failed edit was replayed against the file as its session left it.
  - None matched. One differed only in whitespace, where the model moved a mid-line `hud: hud` onto its own line. The
    rest quote older or paraphrased text.
  - In the 49,152 runs, 6.1% of calls after a compaction failed, against 1.9% before. At 139,264 there was none.
- **`edit` without `path`:** the model omitted the key. The raw stream holds no `path` parameter, and each reply ended
  with `finish_reason=tool_calls` and nothing malformed kept.
- **No loops:** no run repeated an identical call. The long numtools session re-checked its values after each
  compaction, finished, and its tests pass.
- **Malformed:** the one malformed call is a `write` cut by the 24,576-token output limit.
- **Earlier recipes, the same harness at a 262,144 window (4-bit packs on one card):**
  - FP4-Local: 1/165 tool errors.
  - NVFP4: 7/178.
  - MLX4: 18/206.
  - The 49,152 runs sit inside that range, and `agent` at its low end.
