Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# Qwen3.8 Flash Next FP8 on 2x RTX PRO 6000 with TensorFold

A recipe for serving [Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) at full FP8
precision across two RTX PRO 6000 Blackwell cards in one host (tensor parallel, `--tp 2`) with
[TensorFold](https://github.com/ashhart/TensorFold). It ships an exact upstream TensorFold pin, a 46-patch series
(the TP2 collectives and decode-graph work, discrete-card host memory, block-FP8 experts at TP2, and the levers that
make up **stack E**), the measured serve profile, the weights' exact hashes, and the numbers behind them.

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

## Numbers

Stack E, the profile in `config/cuda/`, on two RTX PRO 6000 Blackwell Server Edition (97,887 MiB each, PCIe Gen5
x16, peer-to-peer read and write) in one host: `--context 49152 --parallel 16`, 256-token replies. Each figure is the
mean of the two stack E runs of an A-B-B-A pair ([PERFORMANCE](docs/PERFORMANCE.md)).

| tok/s | code | chat |
| --- | --- | --- |
| 1 stream, greedy | **319.4** | **219.0** |
| 1 stream, sampled (temperature 0.7) | 251.1 | 178.0 |
| 4 / 8 / 16 streams, greedy (aggregate) | **750 / 1,016 / 966** | 541 / 746 / 896 |
| cold prefill, 8K / 32K prompt | 4,409 / 4,440 | |

+7.3% / +6.6% at one stream (greedy code / chat) and +11-15% at 4-16 streams over the previous build, token-identical;
cold prefill is 1.7-2.4% slower ([Known issues](#known-issues)). History in [PERFORMANCE](docs/PERFORMANCE.md).

- **Exact:** greedy replies are token-identical to the previous builds', every concurrent reply equals its alone run,
  and every drafted reply equals its `"draft": false` reply.
- **Soak:** 35 minutes of mixed-length traffic (8K-32K prompts among short ones) at 1, 8 and 16 streams, greedy and
  sampled: 1,500 requests, 0 failed; 1,460 equality checks, 0 unequal; no NCCL error; memory flat
  ([QUALITY](docs/QUALITY.md#soak)).
- **Quality:** stack E scores HumanEval+ 93.3 and BFCL held-out 74.7, within run noise of the earlier FP8 binary's
  94.5 / 74.0 by the same scorer and settings, and of NVFP4's 93.9 / 72.9 and FP4-Local's 92.7 / 75.8; FP8 is the
  fidelity reference (4-bit packs: 85-86% top-1, KL 0.10-0.13). OpenCode on stack E: 10/10 at a 128K window and 9/10
  at 256K with 0 tool errors, against FP4-Local 10/10 and NVFP4 9/10 at 256K ([QUALITY](docs/QUALITY.md)).
- **Capacity:** stack E serves 16 streams at 49,152 tokens on the cards. For longer windows,
  `TF_FLASHNEXT_KV_HOST=overflow` (in this engine) keeps the K/V rows that do not fit in pinned host memory: 27 streams
  at 64K, 17 at 128K and 10 at 256K, measured on an earlier build of this line. Stack E ran OpenCode at 256K x 5
  streams with it (9/10, 0 tool errors). The opt-in fp8 KV cache, measured on a later branch and not in this series,
  fits 13 streams at 128K and 7 at 256K against bf16 KV's 9 / 4 ([CAPACITY](docs/CAPACITY.md)).

## Requirements

- Two RTX PRO 6000 Blackwell cards (sm_120) in one host, with PCIe peer-to-peer between them
  (`nvidia-smi topo -p2p rw` shows OK). Measured on the Server Edition at its 600 W limit, Gen5 x16, both cards on one NUMA node.
- NVIDIA driver with CUDA 13 support (measured: 620.70.06, CUDA driver API 13.5).
- CUDA 13 toolkit for `nvcc` (measured: 13.0.2, nvcc 13.0.88) and Zig 0.17.0 (the engine's `.zig-version`).
- Git and network access to GitHub for the pin and xgrammar v0.2.8 (or pass local checkouts; see the scripts).
- NCCL 2.30 or later (measured: 2.32.6 from the `nvidia-nccl-cu13` wheel); set `TF_NCCL_LIB` to its `libnccl.so.2`
  when it is not on the loader path.
- About 173 GiB of disk for the weights. Each rank's host memory stayed under 4.1 GiB in the soak.

## Quick start

```bash
git clone https://github.com/jayleaton/qwen-flash-rtx6000-tensorfold.git
cd qwen-flash-rtx6000-tensorfold
hf download Qwen/Qwen3.8-Flash-Next-FP8 --revision 7fb48d708ca6cd8eeed8f4db0433be18af96984f --local-dir flash-fp8
scripts/verify-weights.sh flash-fp8                 # sha256 of every shard against config/weights
scripts/build-engine.sh engine                      # pin + patches 0001-0046, tree checked, sm_120 server + libtfgrammar
scripts/serve.sh engine flash-fp8                   # stack E: rank 1 on GPU 1, rank 0 on GPU 0 serves port 8090
```

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"flash","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":2048}'
```

`scripts/serve.sh` reads [`config/cuda/serve.args`](config/cuda/serve.args) (`--context 49152 --parallel 16`) and
exports [`config/cuda/serve.env`](config/cuda/serve.env), one commented lever a line; a variable already set in your
environment wins. `GPUS=0,1`, `PORT`, `NAME`, `NUMA_NODE` and `LOG_DIR` adjust it, and extra arguments pass through
to both ranks. It refuses to start without `engine/zig-out/native/lib/libtfgrammar.so`, which `tool_choice` required or
named and `response_format` need (`NO_GRAMMAR=1` starts anyway). The equivalent by hand:

```bash
env $(grep -v '^#' config/cuda/serve.env) CUDA_VISIBLE_DEVICES=1 engine/zig-out/native/bin/tensorfold-native serve flash-fp8 \
  --name flash --context 49152 --parallel 16 --tp 2 --rank 1 --master 127.0.0.1 --port 8091 &
env $(grep -v '^#' config/cuda/serve.env) CUDA_VISIBLE_DEVICES=0 engine/zig-out/native/bin/tensorfold-native serve flash-fp8 \
  --name flash --context 49152 --parallel 16 --tp 2 --rank 0 --master 127.0.0.1 --port 8090
```

Each rank digests every checkpoint byte on first load (cached afterwards) to check both load the same weights;
`TENSORFOLD_CHECKPOINT_SHA256`, set to the same 64 hex digits on both ranks, replaces that digest. To check the patch
series without building, run `scripts/fetch-engine.sh cuda DIR`.

## Versions

| `config/` | upstream pin | patches | engine tree | verified |
| --- | --- | --- | --- | --- |
| `cuda` | `3661a4281` (TensorFold `main`, 2026-11-20) | `cuda` `0001`-`0046` | `6df6ffa158e4` | 2026-11-21 |

The patched tree is byte-identical to the tree the numbers were measured on. The series and what each part adds:
[ENGINE](docs/ENGINE.md).

## Known issues

- **Cold prefill is 2.4% slower** with `TF_FLASHNEXT_PEER_SLOTS=1`; unset it for prefill-heavy work.
- **`TF_FLASHNEXT_PEER_SLOTS` needs one host with peer-to-peer**; it is refused across hosts and with
  `TF_FLASHNEXT_DECODE_GRAPH=1`.
- **Mixed lengths stall decode:** long prompts (8K-32K) arriving among short ones bring 8 streams to 127.0 tok/s end
  to end (p95 round gap 785 ms). Measured, no env fix; open: smaller chunks (`TF_FLASHNEXT_CHUNK=1024`) smooth the p95
  but cost 12% (112.2 against 127.0 tok/s at 8 streams).
- **`TF_FLASHNEXT_KEEP_LATE=1` is refused** at boot on FP8.

More in [KNOWN-ISSUES](docs/KNOWN-ISSUES.md).

## Documentation

| | |
| --- | --- |
| [PERFORMANCE](docs/PERFORMANCE.md) | the A-B-B-A method, every arm from stack D to stack E, each lever's share |
| [QUALITY](docs/QUALITY.md) | soak, HumanEval+, BFCL, fidelity against the 4-bit packs, OpenCode |
| [CAPACITY](docs/CAPACITY.md) | streams at 64K, 128K and 256K, host overflow, the fp8 KV cache (measured, not in this series) |
| [WEIGHTS](docs/WEIGHTS.md) | the FP8 checkpoint, its revision and hashes |
| [ENGINE](docs/ENGINE.md) | the pin, the series and where each patch comes from |
| [KNOWN-ISSUES](docs/KNOWN-ISSUES.md) | what is not solved |
| [CREDITS](docs/CREDITS.md) / [LICENSING](docs/LICENSING.md) | who built what, and under which terms |

## Licensing

This recipe's scripts, patches and docs: **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). The
TensorFold source the patches modify keeps TensorFold's license and notices, MiaAI-Lab's adapted code keeps its MIT
notice, and the model keeps the Qwen Community License: [`docs/LICENSING.md`](docs/LICENSING.md).
