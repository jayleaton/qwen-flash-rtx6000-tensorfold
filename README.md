Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# Qwen3.8 Flash Next FP8 on 2x RTX PRO 6000 with TensorFold

A recipe for serving [Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) at full FP8
precision across two RTX PRO 6000 Blackwell cards in one host (tensor parallel, `--tp 2`) with
[TensorFold](https://github.com/ashhart/TensorFold). It ships an exact upstream TensorFold pin, a 93-patch series
(the TP2 collectives and decode-graph work, discrete-card host memory, block-FP8 experts at TP2, the levers that make
up **stack E**, the `--pp 2` pipeline, the prompt-exchange levers, the decode fixes, the `--tp 2` prompt overlap, the
round switches and the `--pp 2` prompt cache), six measured serve profiles for `--tp 2` and `--pp 2`, the weights'
exact hashes, and the numbers behind them.

It is for anyone with two Blackwell RTX PRO 6000 cards (96 GB each) who wants the unquantized FP8 checkpoint rather
than a 4-bit pack: FP8 is the fidelity reference the 4-bit packs are measured against. One card does not hold the FP8
weights (about 72 GiB a rank at TP2).

Qwen's next Flash checkpoint (Flash 4) is not covered yet: the engine fixes Flash Next's geometry at compile time, so
a new checkpoint needs its own measured pass.

## Credits

- **Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) by the
  [Qwen team](https://huggingface.co/Qwen)**, Qwen Community License 1.0. Download it under the terms on its model
  card; no weights are included here.
- **Engine: [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart (ashhart) and the TensorFold
  contributors**, Apache License 2.0 (code written before 0.6.0 keeps its MIT notice). The Flash Next CUDA family, its
  kernels, the TP2 rank engine and the server are upstream work. The patches here extend it; most are not upstream yet.
- **TP2 collectives, decode graph, control link and rank load waits: Scott Biggart (swbiggart)**, upstream PRs
  #2971, #2972, #2978, #2980 and #2982, carried as patches 0001-0026.
- **[MiaAI-Lab](https://github.com/MiaAI-Lab)**: the shared expert on a side stream comes from MiaAI-Lab's RTX PRO 6000
  Flash Next recipe (MIT), here split across both ranks; the tree's `THIRD_PARTY_NOTICES.md` and
  `LICENSES/Mia-FlashNext-MIT.txt` carry the notice.

More in [CREDITS](docs/CREDITS.md); license terms in [LICENSING](docs/LICENSING.md) and [NOTICE](NOTICE).

## Profiles

`scripts/serve.sh` serves one of six profiles (`PROFILE=`, files in [`config/cuda/profiles`](config/cuda/profiles)).
Every profile starts from stack E's levers ([`config/cuda/serve.env`](config/cuda/serve.env)) and keeps its tokens:
greedy replies are identical across all six.

| `PROFILE=` | launch | levers on top of stack E | pick it for |
| --- | --- | --- | --- |
| **`tp2-fast`** (default) | `--tp 2`, 49,152 x 16 | owner-slot prompt exchange, split DeltaNet prompt chain, and the seven exact decode fixes | one stream and general serving: the fastest single stream, +32% prefill over stack E |
| `pp2-prefill` | `--pp 2`, 49,152 x 16 | split DeltaNet prompt chain, the four decode fixes that act under `--pp 2`, the three round switches and the prompt cache | long prompts and wide batches: about 2x `tp2-fast`'s cold prefill at 32K and 128K, the best x16 |
| `agent` | `--pp 2`, 139,264 x 4 | as `pp2-prefill` | coding agents (OpenCode): a window that never compacts their history |
| `agent-tp2` | `--tp 2`, 139,264 x 4 | as `tp2-fast` | coding agents that want the faster single stream and the prompt cache |
| `tp2-overlap` (opt-in) | `--tp 2`, 49,152 x 14 | `tp2-fast` + the prompt overlap and the early n-gram rows | about 40% faster `--tp 2` prefill and +3% one stream, at 14 streams |
| `stack-e` | `--tp 2`, 49,152 x 16 | none | the previous release's profile, unchanged |

**New in patches 0077-0093** ([PERFORMANCE](docs/PERFORMANCE.md#patches-0077-0093-2026-11-26)):
- **`pp2-prefill` and `agent`:**
  - The `--pp 2` prompt cache is now on at the default budget, and it caches as much as `--tp 2` does. On the captured
    OpenCode sessions, five workflows at once run at 209.7 tok/s against 104.3 without the cache.
  - The round switches add +3.5% to one stream.
- **`tp2-fast`, `agent-tp2` and `stack-e`** launch exactly as before.
- **`tp2-overlap` is separate, not the default.** It holds about 2.15 GiB more a rank, so it serves 14 streams. Its
  speed was measured before rank 1's early-row fix (0091) and is inferred, not remeasured, after it. A GPU check on
  0093 confirms its tokens ([QUALITY](docs/QUALITY.md#exactness-of-patches-0077-0093-2026-11-26)).

**Why `tp2-fast` is the default now:** it beats stack E in every cell measured below, prefill and decode, at 1, 4 and
16 streams, with token-identical replies and a stable 30-minute soak. `stack-e` stays one variable away.

## Numbers

**2026-11-26, the series to 0090** (2x RTX PRO 6000 Workstation Edition at 600 W, Threadripper PRO 7965WX, driver
615.71.09). These are A-B-B-A means. `tp2-fast` and `tp2-overlap` are both served at 14 streams here, and 128K runs on
its own boot at 139,264 x 4.

| tok/s | cold 8K | cold 32K | cold 128K | code x1 | chat x1 | code x16 | chat x16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tp2-fast` (x14) | 6,529 | 6,634 | 6,381 | 364.8 | 256.1 | 1,012.2 | 804.0 |
| **`tp2-overlap`** (x14) | **9,026** | **9,343** | **8,801** | **375.7** | **263.0** | 1,018.6 | 804.8 |
| 0076 `pp2-prefill` | 8,146 | 12,595 | 13,777 | 300.1 | 207.4 | 1,198.7 | 1,182.2 |
| **`pp2-prefill`** | 8,142 | 12,596 | 13,805 | **310.6** | **213.0** | **1,212.6** | **1,198.6** |

- **Exact:** every cell's tokens equal R's (the 0076 `--tp 2` recipe), and R's equal the recipe's. Every 128K reply is
  `9a3c5d24`.
- **Agents and mixed traffic:** every OpenCode turn equals the pinned ids on every `--pp 2` arm and on `tp2-fast`, and
  every mixed-traffic decoder reply equals its solo reply.
- **`tp2-overlap` before 0091** was not exact under agent and mixed traffic: one turn in 24 and one lone reply split.
  0091 fixes it, confirmed on GPUs ([QUALITY](docs/QUALITY.md#exactness-of-patches-0077-0093-2026-11-26)).
- **Soak:** 15 minutes of `tp2-overlap` with the cache: STABLE.
- **Host:** compare rows within one table only. The table below is from another host.

**2026-11-24, the series to 0076,** measured on a rented host with 2x RTX PRO 6000 Blackwell **Workstation Edition**
(600 W, one NUMA node, peer-to-peer 53.8 GB/s each way), Threadripper 9960X, driver 595.99.02, ranks pinned to the GPUs'
node ([PERFORMANCE](docs/PERFORMANCE.md)).
- **Runs:** 256-token greedy replies, cold prompts with `--prompt-cache-gib 0`.
- **Averaging:** the `tp2` and `pp2` rows are means of an A-B-B-A pair. Stack E is one run, on the 0046 build.
- **128K:** each 128K prompt runs on its own boot at 139,264 x 4, since a 49,152-token window does not hold it.

| tok/s | cold 8K | cold 32K | cold 128K | code x1 | code x4 | code x16 | chat x1 | chat x4 | chat x16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stack-e` | 5,349 | 5,397 | 5,264 | 357.5 | 848 | 1,144 | 246.0 | 598 | 1,024 |
| RC-tp2 (owner slots + split chain, no decode fixes) | 7,122 | 7,150 | 6,940 | 358.8 | 851 | 1,138 | 245.8 | 597 | 1,021 |
| **`tp2-fast`** | 7,118 | 7,128 | 6,938 | **374.2** | **916** | 1,214 | **259.2** | **644** | 1,120 |
| RC-pp2 (`--pp 2` + split chain, no decode fixes) | 9,700 | 14,868 | 16,036 | 306.4 | 854 | 1,211 | 207.7 | 597 | 1,208 |
| **`pp2-prefill`** | **9,736** | **14,807** | **15,964** | 304.5 | 867 | **1,293** | 209.5 | 604 | **1,259** |

- **The decode fixes:**
  - `tp2-fast` against RC-tp2 gains +4.3% / +7.6% / +6.7% on code at 1 / 4 / 16 streams and +5.5% / +7.9% / +9.7% on
    chat. Prefill moves −0.1% to −0.3%.
  - `pp2-prefill` against RC-pp2 gains +6.8% (code) and +4.2% (chat) at 16 streams and is level at one stream. Prefill
    moves −0.4% to +0.4%.
  - Every sign held in both rounds.
- **Exact:** every cell's greedy tokens equal the previous release's: code `153389030f94`, chat `147305861125`, cold 8K
  `e1dfcdac`, 32K `ddadf33e`, 128K `9a3c5d24`. Every concurrent reply equals its alone run.
- **Soak:** 30 minutes each of `tp2-fast` and `pp2-prefill`. Probes every 5 minutes (greedy hashes and a cold 32K
  prompt), and load at 1, 4 and 16 streams between them.
  - `tp2-fast`: STABLE, 143 load rounds, code x1 mean 375.1, drift ≤0.2%.
  - `pp2-prefill`: STABLE, 146 load rounds, code x1 mean 304.2, drift ≤0.5%.
  - Both: hashes constant, 0 failed requests, no rank exit.
- **Quality:** RC-tp2 scores 213/228 on the quality suite (gsm8k 95/100, mmlu 90/100, instructions 20/20, long context
  8/8). The profiles' tokens equal its tokens, and stack E's. The kernels the fixes add pass TensorFold's kernel
  contract on the GPU, and the fused router's logits and plan are byte-equal to the kernels they replace
  ([QUALITY](docs/QUALITY.md)).
- **OpenCode:** replaying 10 captured agent requests at temperature 0 gives the same tokens on every profile, with or
  without the prompt cache.
  - At 49,152 tokens OpenCode compacts its history 9-26 times a pass, and most failed `edit` calls follow a compaction.
    Tool errors run 3.2-7.3%.
  - At 139,264 tokens (`agent`) it never compacts: 4.5% tool errors, 1 `oldString` mismatch, every workflow passes.
  - No parser fault and no loops ([QUALITY](docs/QUALITY.md#opencode)).
- **Host:** the previous release measured stack E at 319.4 code x1 on a Server Edition host. The same binary runs 357.5
  on this one, so compare rows within one table only.

## Requirements

- Two RTX PRO 6000 Blackwell cards (sm_120) in one host, with PCIe peer-to-peer between them
  (`nvidia-smi topo -p2p rw` shows OK). Measured on the Workstation Edition (this release) and the Server Edition
  (stack E's release), both at their 600 W limit, Gen5 x16, both cards on one NUMA node.
- NVIDIA driver with CUDA 13 support (measured: 595.99.02 and 620.70.06).
- CUDA 13 toolkit for `nvcc` (measured: nvcc 13.0.88) and Zig 0.17.0 (the engine's `.zig-version`).
- Git and network access to GitHub for the pin and xgrammar v0.2.8 (or pass local checkouts; see the scripts).
- NCCL 2.30 or later (measured: 2.32.6 from the `nvidia-nccl-cu13` wheel); set `TF_NCCL_LIB` to its `libnccl.so.2`
  when it is not on the loader path.
- About 173 GiB of disk for the weights.

## Quick start

**1. Get the recipe and the weights**

```bash
git clone https://github.com/jayleaton/qwen-flash-rtx6000-tensorfold.git
cd qwen-flash-rtx6000-tensorfold
hf download Qwen/Qwen3.8-Flash-Next-FP8 --revision 7fb48d708ca6cd8eeed8f4db0433be18af96984f --local-dir flash-fp8
scripts/verify-weights.sh flash-fp8                 # sha256 of every shard against config/weights
```

**2. Get the engine: prebuilt (fastest) or built from source**

```bash
# prebuilt: the 0076 binary from the release (glibc 2.28 or later); it predates 0077-0093, see below
T=tensorfold-native-fa72a8b22f5f-sm120-x86_64
U=https://github.com/jayleaton/qwen-flash-rtx6000-tensorfold/releases/download/rc-20261124
curl -fLO "$U/$T-linux-x86_64.tar.gz" -fLO "$U/$T-linux-x86_64.tar.gz.sha256"
sha256sum -c "$T-linux-x86_64.tar.gz.sha256" && tar -xzf "$T-linux-x86_64.tar.gz" && mv "$T" engine

# or build it: pin + patches 0001-0093, tree checked, sm_120 server + libtfgrammar
scripts/build-engine.sh engine
```

**3. Serve: pick tp2 or pp2**

```bash
# tp2 (default, tp2-fast): the fastest single stream
NUMA_NODE=0 scripts/serve.sh engine flash-fp8

# pp2 (pp2-prefill): 2x the prefill on long prompts, the best 16-stream throughput
PROFILE=pp2-prefill NUMA_NODE=0 scripts/serve.sh engine flash-fp8

# coding agents (OpenCode and the like): 139,264-token window x 4 streams
PROFILE=agent NUMA_NODE=0 scripts/serve.sh engine flash-fp8        # pp2
PROFILE=agent-tp2 NUMA_NODE=0 scripts/serve.sh engine flash-fp8    # tp2

# opt-in: tp2 with the prompt overlap, 14 streams (needs the 0093 build)
PROFILE=tp2-overlap NUMA_NODE=0 scripts/serve.sh engine flash-fp8
```

Rank 1 starts on GPU 1, then rank 0 on GPU 0 serves port 8090. Set `NUMA_NODE` to the node your GPUs sit on
(`nvidia-smi topo -m`), or leave it out on a single-node host. Then:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"flash","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":2048}'
```

- **What `serve.sh` reads:** the profile's `NAME.args` (window, streams, `--tp 2` or `--pp 2`), and exports its
  `NAME.env`, then [`config/cuda/serve.env`](config/cuda/serve.env).
  - Both files hold one commented lever a line.
  - A variable already set in your environment wins.
- **Adjust it with:** `GPUS=0,1`, `PORT`, `NAME`, `NUMA_NODE` and `LOG_DIR`. Extra arguments pass through to both ranks.
- **Grammar library:** it refuses to start without `engine/zig-out/native/lib/libtfgrammar.so`, which `tool_choice`
  required or named and `response_format` need. `NO_GRAMMAR=1` starts anyway.
- **By hand:** `env $(grep -hv '^#' config/cuda/profiles/tp2-fast.env config/cuda/serve.env)` before each rank's
  `tensorfold-native serve flash-fp8 --name flash $(cat config/cuda/profiles/tp2-fast.args) --rank N --master 127.0.0.1
  --port P`. Start rank 1 on GPU 1 with port 8091, then rank 0 on GPU 0 with port 8090.

**Prebuilt:** the release carries `tensorfold-native-fa72a8b22f5f-sm120-x86_64-linux-x86_64.tar.gz`, sha256
`a04c0568cc9bbadd52eea9b1f63da3512439af9c5226fcf033171dfa6f834a35`. It holds the server and libtfgrammar that the series
to 0076 builds, byte-identical to that measured binary, with TensorFold's and xgrammar's license notices.
- **No prebuilt for 0093 yet.** The 0076 server does not know the switches in 0077-0093. It ignores them, so with it:
  - `pp2-prefill` and `agent` run as before, with no prompt cache and no round switches;
  - `tp2-overlap` runs as `tp2-fast` at 14 streams.
- **For the new profiles,** build with `scripts/build-engine.sh` until a 0093 prebuilt is released.

Each rank digests every checkpoint byte on first load (cached afterwards) to check both load the same weights.
`TENSORFOLD_CHECKPOINT_SHA256`, set to the same 64 hex digits on both ranks, replaces that digest. To check the patch
series without building, run `scripts/fetch-engine.sh cuda DIR`.

## Versions

| `config/` | upstream pin | patches | engine tree | verified |
| --- | --- | --- | --- | --- |
| `cuda` | `3661a4281` (TensorFold `main`, 2026-11-20) | `cuda` `0001`-`0093` | `4c9476f45d0d` | 2026-11-26 |

- **Measured trees:**
  - the 2026-11-26 numbers ran on the series to `0090`;
  - the 0093 tree's tokens were checked on GPUs against them;
  - the 2026-11-24 numbers ran on `fa72a8b22f5f`, the series to `0076`.
- **Earlier tree:** stack E's tree `6df6ffa158e4` is the series to `0046`.
- **The series and what each part adds:** [ENGINE](docs/ENGINE.md).
- **Changes by release:** [CHANGELOG](CHANGELOG.md).

## Known issues

- **`pp2-prefill` and `agent` lose 15% at one stream:**
  - against `tp2-fast`: 304.5 vs 374.2 code, 209.5 vs 259.2 chat;
  - against stack E: −14.8%.
  - The decode fixes that apply under `--pp 2` do not close it. Use `tp2-fast` when one stream matters most.
- **`--pp 2`'s prompt cache is measured at 49,152 x 16 only.** `agent` now starts with it at 139,264 x 4, and that
  combination has not been booted yet. `--prompt-cache-gib 0` serves as before.
- **`tp2-overlap` is opt-in at 14 streams.** Its speed after 0091 is inferred, not remeasured. Whether the overlap
  becomes the default is open.
- **Coding agents at 49,152 tokens compact their history,** and most failed `edit` calls follow a compaction. Use
  `agent` or `agent-tp2` for them.
- **Not shipped:**
  - `TF_FLASHNEXT_OUTPROJ_KSPLIT` is in the series, off by default. It changes the tokens: +11% code at one stream, 54%
    of a code reply matching, quality 214/228.
  - The other levers in 0047-0076 (`TF_FLASHNEXT_HIDDEN_SUM`, window graphs, `TF_FLASHNEXT_PP_SPEC`) are measured and
    not adopted.
- **Lower power limits are not measured yet.** A 300 W limit could not be set on the test host, and the one capped
  host found had half the peer-to-peer bandwidth, so its numbers were not kept.
- **Mixed lengths stall decode,** and `TF_FLASHNEXT_KEEP_LATE=1` is refused on FP8, as in stack E.

More in [KNOWN-ISSUES](docs/KNOWN-ISSUES.md).

## Documentation

| | |
| --- | --- |
| [PERFORMANCE](docs/PERFORMANCE.md) | the A-B-B-A method, the profiles' full matrix, every arm from stack D to stack E, each lever's share |
| [QUALITY](docs/QUALITY.md) | soaks, quality suite, HumanEval+, BFCL, fidelity against the 4-bit packs, OpenCode and its tool calls |
| [CAPACITY](docs/CAPACITY.md) | streams at 64K, 128K and 256K, host overflow, the fp8 KV cache (measured, not in this series) |
| [WEIGHTS](docs/WEIGHTS.md) | the FP8 checkpoint, its revision and hashes |
| [ENGINE](docs/ENGINE.md) | the pin, the series and where each patch comes from |
| [KNOWN-ISSUES](docs/KNOWN-ISSUES.md) | what is not solved |
| [CREDITS](docs/CREDITS.md) / [LICENSING](docs/LICENSING.md) | who built what, and under which terms |

## Licensing

This recipe's scripts, patches and docs: **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). The
TensorFold source the patches modify keeps TensorFold's license and notices, MiaAI-Lab's adapted code keeps its MIT
notice, and the model keeps the Qwen Community License: [`docs/LICENSING.md`](docs/LICENSING.md).
