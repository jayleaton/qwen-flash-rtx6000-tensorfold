# Performance

Stack E's sections below: two RTX PRO 6000 Blackwell Server Edition cards (97,887 MiB each, 600 W limit, PCIe Gen5 x16,
both on NUMA node 0,
peer-to-peer read and write OK, 38.9 GiB/s each way), 2x Xeon 6520P, driver 620.70.06, NCCL 2.32.6, FP8 checkpoint
`Qwen/Qwen3.8-Flash-Next-FP8@7fb48d70`, `--tp 2`, ranks on 127.0.0.1, each under
`numactl --cpunodebind=0 --preferred=0`.

## Patches 0077-0093 (2026-11-26)

- **Host:** a rented host with 2x RTX PRO 6000 Blackwell Workstation Edition (600 W, one NUMA node, topology NODE,
  peer-to-peer 53.8 / 53.5 GB/s).
  - AMD Threadripper PRO 7965WX (24 cores), governor `schedutil`, driver 615.71.09, NCCL 2.32.6.
  - Both ranks under `numactl --cpunodebind=0`, on 127.0.0.1.
- **Build:** the series to `0090` (tree of `fp8/combined-improvements` a87951e4a), before rank 1's early-row fix (0091).
- **Cells:** as in the section below, with `--prompt-cache-gib 0`, 2 cells an arm in one A-B-B-A sitting
  (R C CO PF PK PK PF CO C R), then C at 16 streams once.
- **Arms:**
  - R is the 0076 recipe's `--tp 2` (stack E + RC-tp2);
  - C is `tp2-fast` and CO is `tp2-overlap`, both at 14 streams;
  - C16 is `tp2-fast` as shipped, at 16 streams;
  - PF is the 0076 `pp2-prefill` and PK is PF + the three round switches, the new `pp2-prefill`.
  - C is served at 14 streams like CO, so the x16 rows compare like with like. Its clients still run at 1 and 16.

| tok/s | cold 8K | cold 32K | code x1 | chat x1 | code x16 | chat x16 |
| --- | --- | --- | --- | --- | --- | --- |
| R (x16) | 6,513 | 6,614 | 351.4 | 241.5 | 1,087.1 | 984.7 |
| C = `tp2-fast` (x14) | 6,529 | 6,634 | 364.8 | 256.1 | 1,012.2 | 804.0 |
| C16 = `tp2-fast` (x16, as shipped) | 6,529 | 6,623 | 365.1 | 257.1 | 1,155.8 | 1,072.5 |
| **CO = `tp2-overlap`** (x14) | **9,026** (+38.3%) | **9,343** (+40.8%) | **375.7** (+3.0%) | **263.0** (+2.7%) | 1,018.6 (+0.6%) | 804.8 (+0.1%) |
| PF = 0076 `pp2-prefill` | 8,146 | 12,595 | 300.1 | 207.4 | 1,198.7 | 1,182.2 |
| **PK = `pp2-prefill`** | 8,142 | 12,596 | **310.6** (+3.5%) | **213.0** (+2.7%) | **1,212.6** (+1.2%) | **1,198.6** (+1.4%) |

- **The table:**
  - code and chat x1 are `bench_openai`'s one-stream greedy rows;
  - CO's percentages are against C, and PK's against PF;
  - every arm's hashes equal R's, and R's equal the recipe's.
- **Time to first token:** CO cuts C's from 1.25 to 0.91 s at 8K and from 4.94 to 3.51 s at 32K.
- **Cold 128K,** at `--context 139264 --parallel 4`, A-B-B-A, 2 cells an arm, `--prompt-cache-gib 0`:

  | arm | TTFT s | tok/s |
  | --- | --- | --- |
  | R | 20.58 | 6,368 |
  | C | 20.54 | 6,381 |
  | **CO** | **14.89** | **8,801** (+37.9%) |
  | PF | 9.51 | 13,777 |
  | PK | 9.49 | 13,805 (+0.2%) |

  Every reply's hash is the recipe's `9a3c5d245425`.
- **The prompt cache under `--pp 2`,** replaying the captured OpenCode sessions (24 turns one workflow at a time, 38
  turns five workflows at once, greedy, 320 tokens a turn) at 49,152 x 16:

  | arm | x1 tok/s | x5 tok/s | prompt tokens cached x1 / x5 |
  | --- | --- | --- | --- |
  | PF, no cache | 106.1 | 104.3 | 0% / 0% |
  | PF + cache | 141.1 | 209.2 | 74% / 82% |
  | **PK + cache** (`pp2-prefill`) | **144.2** | **209.7** | 74% / 82% |
  | `tp2-fast` + cache | 140.9 | 213.0 | 74% / 82% |
  | `tp2-overlap` + cache (x14) | 152.7 | 225.7 | 74% / 82% |

  The tok/s are tokens over wall time. With the cache, `--pp 2` caches as much as `--tp 2` and doubles its x5
  throughput.
- **Mixed traffic** (decoders while 8K and 32K cold prompts arrive, no cache, one cell an arm):

  | scenario, arm | long prompts' TTFT s | round gap p50 while one prefills, ms | aggregate tok/s |
  | --- | --- | --- | --- |
  | M1 (4 decoders; 8K then 32K), `tp2-fast` | 1.33, 5.38 | 496 | 239.7 |
  | M1, `tp2-overlap` | 1.05, 4.17 | 381 | 270.4 |
  | M1, 0076 `pp2-prefill` | 1.13, 4.53 | 420 | 228.2 |
  | M1, `pp2-prefill` | 1.14, 4.60 | 419 | 229.8 |
  | M2 (8 decoders; a burst of 8K, 8K, 32K, 32K), `tp2-fast` | 1.33, 2.70, 8.67, 13.63 | 505 | 234.4 |
  | M2, `tp2-overlap` | 1.06, 2.17, 6.90, 10.81 | 397 | 270.2 |
  | M3 (1 decoder; a 32K prompt 3 s in), `tp2-fast` | 5.20 | 485 | 109.6 |
  | M3, `tp2-overlap` | 4.08 | 390 | 128.2 |

  Decoders still keep only 2-8% of their speed while a long prompt prefills, on every profile.
- **After the fix:** `tp2-overlap`'s numbers above come from the build before rank 1's early-row fix (0091). The fix
  changes no reads or launches, so its speed is expected to be the same, but it was not remeasured.
  The fix's GPU check is in [QUALITY](QUALITY.md#exactness-of-patches-0077-0093-2026-11-26).

## The profiles (2026-11-24, patches to 0076)

- **Host:** a rented host with 2x RTX PRO 6000 Blackwell Workstation Edition (600 W, one NUMA node, topology NODE,
  peer-to-peer 53.8 / 53.5 GB/s, 1 GiB `cudaMemcpyPeer`).
  - AMD Threadripper 9960X (24 cores), governor `schedutil`, driver 595.99.02, NCCL 2.32.6.
  - Both ranks under `numactl --cpunodebind=0`, on 127.0.0.1.
- **Cells:** each cell boots its own server.
  - `bench_openai` (greedy code and chat, 256 tokens), cold 8K and 32K prompts (`bench_cache --require-cold`), then
    `bench_concurrent --alone --serial` at 1, 4 and 16 streams.
  - Run with `--prompt-cache-gib 0`, so every prompt is cold.
  - The cold 128K prompt runs on its own boot at `--context 139264 --parallel 4`.
- **Order:** the `tp2` pair ran as R C C R and the `pp2` pair as P PF PF P. Stack E (one boot of the 0046 build at
  49,152 x 16, plus one at 139,264 x 4 for 128K) ran in the same sitting.

| tok/s | cold 8K | cold 32K | cold 128K | code x1 | code x4 | code x16 | chat x1 | chat x4 | chat x16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stack E (0046 build) | 5,349 | 5,397 | 5,264 | 357.5 | 847.8 | 1,143.7 | 246.0 | 598.3 | 1,024.1 |
| RC-tp2 = stack E + `OWNER_SLOTS` + `GDN_SPLIT` | 7,122 | 7,150 | 6,940 | 358.8 | 851.0 | 1,138.3 | 245.8 | 597.3 | 1,021.2 |
| **`tp2-fast`** = RC-tp2 + the seven decode fixes | 7,118 | 7,128 | 6,938 | **374.2** | **915.5** | 1,214.1 | **259.2** | **644.4** | 1,120.1 |
| RC-pp2 = stack E env + `--pp 2` + `GDN_SPLIT` | 9,700 | 14,868 | 16,036 | 306.4 | 853.9 | 1,211.0 | 207.7 | 596.5 | 1,207.7 |
| **`pp2-prefill`** = RC-pp2 + the four `--pp 2` fixes | **9,736** | **14,807** | **15,964** | 304.5 | 867.2 | **1,293.4** | 209.5 | 604.0 | **1,258.8** |

The A-B-B-A rounds, `tp2-fast` against RC-tp2 (code / chat):

| cell | R | `tp2-fast` | R | `tp2-fast` vs R |
| --- | --- | --- | --- | --- |
| x1 | 359.4 / 246.4 | 374.3 / 260.0, 374.1 / 258.5 | 358.1 / 245.1 | +4.3% / +5.5% |
| x4 | 855.3 / 600.6 | 913.4 / 644.9, 917.6 / 643.9 | 846.8 / 594.0 | +7.6% / +7.9% |
| x16 | 1,140.1 / 1,021.7 | 1,213.8 / 1,123.3, 1,214.4 / 1,116.9 | 1,136.6 / 1,020.7 | +6.7% / +9.7% |
| cold 32K | 7,169 | 7,129, 7,128 | 7,130 | -0.3% |

- **`tp2-fast` against RC-tp2:** every decode sign held in both rounds.
- **`pp2-prefill` against RC-pp2:**
  - code and chat x16 gain +6.8% and +4.2%, and x4 gains +1.6% and +1.3%; every sign held;
  - one stream moves −0.6% and +0.9%, so there is no sign;
  - cold 8K / 32K / 128K move +0.4%, −0.4% and −0.4%.
- **Each `tp2` fix alone,** one cell each on a slower host (Server Edition, `powersave` governor, a code x1 base of
  281-293):
  - each switch was within that host's single-cell noise; together they are the +4.3% above;
  - the batched rollback replay and spinning waits are host-side, so they may read larger on a slow host.
- **Not shipped:** `TF_FLASHNEXT_OUTPROJ_KSPLIT` (0076, off) on top of `tp2-fast` gives code x1 398.3 (+11.0% over
  RC-tp2), chat 238.9 (−2.8%), x16 +15.8% / +6.2%. It changes the tokens (54% of a code reply matches) and is not in a
  profile.

## Method

- Every cell: `--context 49152 --parallel 16`, 256-token replies, TensorFold's `bench_openai` (one stream, code and chat
  prompts, greedy and sampled at temperature 0.7) and `bench_concurrent --alone --serial` at 1, 4, 8 and 16 streams,
  then cold 8K and 32K prompts (`bench_cache --require-cold`). Code and chat are separate prompt sets; "aggregate" is
  the total tokens per second of all streams.
- **A-B-B-A:** a change is measured as base, change, change, base in one sitting, and a column is the mean of its two
  runs. Repeats of one arm agree within about 0.5%.
- **Exactness gates on every arm:** the token hashes equal the base's (greedy code `153389030f94`, greedy chat
  `147305861125`, the sampled sets equal); every concurrent reply equals its alone run; every alone run equals its
  `"draft": false` reply.

## Stack E against E0 and stack D (A-B-B-A, 2026-11-21)

| tok/s | stack D | E0 (D + split rows) | stack E without the split `fn_row_best` | **stack E** | E vs E0 |
| --- | --- | --- | --- | --- | --- |
| 1 stream code / chat | 296.7 / 204.5 | 297.8 / 205.5 | 314.4 / 216.2 | **319.4 / 219.0** | +7.3% / +6.6% |
| 1 stream sampled code / chat | 236.2 / 169.9 | 236 / 169 | 248.9 / 177.1 | 251.1 / 178.0 | +6.4% / +5.3% |
| code x4 / x8 / x16 | | 671 / 894 / 846 | 729 / 981 / 926 | **750 / 1,016 / 966** | +12% / +14% / +14% |
| chat x4 / x8 / x16 | | 487 / 659 / 781 | 527 / 720 / 857 | 541 / 746 / 896 | +11% / +13% / +15% |
| cold prefill 8K / 32K | 4,394 / 4,424 | 4,487 / 4,550 | 4,411 / 4,438 | 4,409 / 4,440 | -1.7% / -2.4% |

Stack D at 8 and 16 streams measured 845-855 and 791-803 tok/s (greedy code) on the same binary line. Stack E's split
`fn_row_best` alone (stack E against the column before it, same env): +1.6% at one stream, +3.6% / +4.3% at 8 / 16
streams, level prefill.

## From stack D to stack E (greedy code)

| arm | 1 stream | x8 / x16 | note |
| --- | --- | --- | --- |
| stack D | 296-297.5 | 845-855 / 791-803 | base |
| E0 = D + `TF_FLASHNEXT_SPLIT_ROWS=16` (A-B-B-A) | 297.8 | 894 / 846 | +4.4% / +4.7% at width |
| E1 = E0 + `TF_FLASHNEXT_WINDOW_GRAPH=0` (A-B-B-A) | 297.4 | 901 / 854 | +0.8% / +1.0% |
| E1 + `TF_QMMF_HALF_BELOW=400` | 298.4 | 901 / 847 | +0.4% at 1 stream |
| E1 + `TF_FLASHNEXT_ATTN_LANES=1` | 296.9 | 910.5 / 866.3 | one run against E1 controls at 898-901 / 850-855; confirmed on stack E below |
| E0 + `TF_FLASHNEXT_PEER_SLOTS=1` (A-B-B-A) | 312.4 / 313.8 | 958 / 903, 962 / 904 | +5% / +7% / +7%; prefill -2.5% |
| stack E without the split `fn_row_best` | 314.2 / 314.6 | 982 / 926, 981 / 926 | full env |
| **stack E** | **319.2 / 319.5** | **1,015 / 966, 1,017 / 967** | split `fn_row_best`: +1.6% / +3.6% / +4.3% |
| stack E without `TF_FLASHNEXT_ATTN_LANES` (A-B-B-A on the stack E binary) | 318.8 / 319.0 | 1,003 / 950, 1,003 / 952 | `TF_FLASHNEXT_ATTN_LANES=1`: +1.3% / +1.6% at x8 / x16 (1,016 / 966 against 1,003 / 951), chat x16 +1.6%, level at 1 stream and in prefill; hashes equal in all four arms |

Stack D itself (env `NCCL_P2P_LEVEL=SYS TF_FLASHNEXT_CHUNK_SOLO=8192 TF_FLASHNEXT_DRAFTS_GREEDY=12
TF_FLASHNEXT_DRAFT_HEAD=affine4 TF_FLASHNEXT_SHARED_SIDE=1 TF_FLASHNEXT_SHARED_SIDE_SPLIT=1`) measured 296.7 / 204.5
tok/s
at one stream (greedy code / chat), sampled 236.2 / 169.9, cold prefill 4,394 / 4,424 at 8K / 32K, and 4,330 tok/s on
a cold 128K prompt (TTFT 30.3 s). Its levers, each from A-B-B-A rows:

- `NCCL_P2P_LEVEL=SYS`: +3.3% / +3.0% at one stream, +4.3% prefill (NCCL over PCIe peer-to-peer instead of host memory).
- `TF_FLASHNEXT_CHUNK_SOLO=8192`: about +6% prefill at 8-32K, +1-2% at 128K.
- `TF_FLASHNEXT_DRAFTS_GREEDY=12`: about +5% greedy code.
- `TF_FLASHNEXT_DRAFT_HEAD=affine4`: +5.6% / +7.6% at one stream greedy, +4.5% / +6.2% sampled.
- `TF_FLASHNEXT_SHARED_SIDE_SPLIT=1`: +5.9% / +6.6% at one stream, +3-5% at 4-16 streams.

## Tried and not adopted

Each from at least one A-B round on this pair, hashes equal:

- `TF_FLASHNEXT_GREEDY_WIDE` 8/16 (-7% at x8), deeper chains at width (-7%), depth 5 everywhere (-8% at x8), depth 4
  everywhere (-23% at one stream, +1.8% at x16).
- `TF_FLASHNEXT_DRAFTS_GREEDY` 14/16 (-3%), `TF_FLASHNEXT_CONFIDENCE` -0.3 (sampled +2%, greedy code -3.4%) and -0.5
  (chat -2.7%), `TF_FLASHNEXT_SHARED_DRAFTS=0` (code -5%, chat +3.5-5%).
- `TF_FLASHNEXT_DECODE_GRAPH=1` (level; it also refuses `TF_FLASHNEXT_PEER_SLOTS`), `TF_FLASHNEXT_WARM_AHEAD=1` and
  `TF_FLASHNEXT_TABLE_WARM=1` (level), `TF_FLASHNEXT_SPLIT_SHARED=0` (level, prefill -1.8%),
  `TF_FLASHNEXT_SPLIT_DENSE=0` (-3%, and 5 fewer streams at 48K), `TF_FLASHNEXT_READOUT_FUSED=1` (+0.9%), 4 NCCL
  channels (level).

## Where a round goes

At one stream on E0 a round is about 20 ms: one verify pass (BF16 projections 6.6 ms, routed experts 3.6 ms, NCCL
3.7 ms on rank 0) plus about 4.3 draft steps of about 568 us on rank 0. At 16 streams, profiled shares on rank 0 are
BF16 projections 43%, NCCL 17-18%, FP8 experts 16-17%, hyper-connections 6%, router logits 4.4%. Stack E's
`TF_FLASHNEXT_PEER_SLOTS` replaces the MoE exchange's NCCL collective in passes of up to 128 rows, and the split
`fn_row_best` (32 blocks a row, byte-equal to the 1024-thread kernel) shortens each draft step.
