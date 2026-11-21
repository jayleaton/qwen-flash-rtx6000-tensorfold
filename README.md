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

Two RTX PRO 6000 Blackwell Server Edition (97,887 MiB each, PCIe Gen5 x16, peer-to-peer read and write), one host,
`--context 49152 --parallel 16`, 256-token replies, greedy unless marked. Each column is the mean of an A-B-B-A pair
of runs; every arm returned the same token hashes ([PERFORMANCE](docs/PERFORMANCE.md)).

| tok/s | stack D | E0 (D + split rows) | **stack E** | E vs E0 |
| --- | --- | --- | --- | --- |
| 1 stream, code / chat | 296.7 / 204.5 | 297.8 / 205.5 | **319.4 / 219.0** | +7.3% / +6.6% |
| 1 stream sampled, code / chat | 236.2 / 169.9 | 236 / 169 | 251.1 / 178.0 | +6.4% / +5.3% |
| code, 4 / 8 / 16 streams (aggregate) | | 671 / 894 / 846 | **750 / 1,016 / 966** | +12% / +14% / +14% |
| chat, 4 / 8 / 16 streams (aggregate) | | 487 / 659 / 781 | 541 / 746 / 896 | +11% / +13% / +15% |
| cold prefill 8K / 32K | 4,394 / 4,424 | 4,487 / 4,550 | 4,409 / 4,440 | -1.7% / -2.4% |

Stack D at 8 and 16 streams measured 845-855 and 791-803 tok/s (greedy code) on the same binary line.

- **Exact:** every arm gives E0's token hashes, every concurrent reply equals its alone run, and every drafted reply
  equals its `"draft": false` reply.
- **Soak:** 35 minutes of mixed-length traffic (8K-32K prompts among short ones) at 1, 8 and 16 streams, greedy and
  sampled: 1,500 requests, 0 failed; 1,460 equality checks, 0 unequal; no NCCL error; memory flat
  ([QUALITY](docs/QUALITY.md#soak)).
- **Quality:** FP8 HumanEval+ 94.5 and BFCL held-out 74.0, against 93.9 / 72.9 (NVFP4) and 92.7 / 75.8 (FP4-Local),
  within run noise; FP8 is the fidelity reference (4-bit packs: 85-86% top-1, KL 0.10-0.13). OpenCode is pending
  ([QUALITY](docs/QUALITY.md)).
- **Capacity:** 16 streams at 49,152 tokens (stack E); 13 at 64K and 7 at 128K on the cards (stack D)
  ([CAPACITY](docs/CAPACITY.md)).

## Requirements

- Two RTX PRO 6000 Blackwell cards (sm_120) in one host, with PCIe peer-to-peer between them
  (`nvidia-smi topo -p2p rw` shows OK). Measured on the Server Edition at its 600 W limit, Gen5 x16, both cards on one NUMA node.
- NVIDIA driver with CUDA 13 support (measured: 620.70.06, CUDA driver API 13.5).
- CUDA 13 toolkit for `nvcc` (measured: 13.0.2, nvcc 13.0.88) and Zig 0.17.0 (the engine's `.zig-version`).
- NCCL 2.30 or later (measured: 2.32.6 from the `nvidia-nccl-cu13` wheel); set `TF_NCCL_LIB` to its `libnccl.so.2`
  when it is not on the loader path.
- About 173 GiB of disk for the weights. Each rank's host memory stayed under 4.1 GiB in the soak.

## Quick start

```bash
git clone https://github.com/jayleaton/qwen-flash-rtx6000-tensorfold.git
cd qwen-flash-rtx6000-tensorfold
hf download Qwen/Qwen3.8-Flash-Next-FP8 --revision 7fb48d708ca6cd8eeed8f4db0433be18af96984f --local-dir flash-fp8
scripts/verify-weights.sh flash-fp8                 # sha256 of every shard against config/weights
scripts/build-engine.sh engine                      # pin + patches/cuda 0001-0046, tree checked, sm_120 server
scripts/serve.sh engine flash-fp8                   # stack E: rank 1 on GPU 1, rank 0 on GPU 0 serves port 8090
```

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"flash","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":2048}'
```

`scripts/serve.sh` reads [`config/cuda/serve.args`](config/cuda/serve.args) (`--context 49152 --parallel 16`) and
exports [`config/cuda/serve.env`](config/cuda/serve.env), one commented lever a line; a variable already set in your
environment wins. `GPUS=0,1`, `PORT`, `NAME`, `NUMA_NODE` and `LOG_DIR` adjust it, and extra arguments pass through
to both ranks. The equivalent by hand:

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
- **Mixed lengths stall decode:** long prompts (8K-32K) arriving among short ones drop 8 streams to about 125 tok/s
  end to end (p95 round gap 790 ms). A fix is being measured.
- **`TF_FLASHNEXT_KEEP_LATE=1` is refused** at boot on FP8.

More in [KNOWN-ISSUES](docs/KNOWN-ISSUES.md).

## Documentation

| | |
| --- | --- |
| [PERFORMANCE](docs/PERFORMANCE.md) | the A-B-B-A method, every arm from stack D to stack E, each lever's share |
| [QUALITY](docs/QUALITY.md) | soak, HumanEval+, BFCL, fidelity against the 4-bit packs, OpenCode (pending) |
| [CAPACITY](docs/CAPACITY.md) | streams at 64K, 128K and 256K, host overflow, the fp8 KV cache (measured, not in this series) |
| [WEIGHTS](docs/WEIGHTS.md) | the FP8 checkpoint, its revision and hashes |
| [ENGINE](docs/ENGINE.md) | the pin, the series and where each patch comes from |
| [KNOWN-ISSUES](docs/KNOWN-ISSUES.md) | what is not solved |
| [CREDITS](docs/CREDITS.md) / [LICENSING](docs/LICENSING.md) | who built what, and under which terms |

## Licensing

This recipe's scripts, patches and docs: **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). The
TensorFold source the patches modify keeps TensorFold's license and notices, MiaAI-Lab's adapted code keeps its MIT
notice, and the model keeps the Qwen Community License: [`docs/LICENSING.md`](docs/LICENSING.md).
