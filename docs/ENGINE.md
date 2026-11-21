# Engine

`config/cuda/` pins upstream TensorFold `main` at `3661a4281425a71eb5f2a56922f0f2a33e5f4f17` (2026-11-20).
`scripts/fetch-engine.sh cuda DIR` fetches it, applies `patches/cuda/0001`-`0046` with `git am` (fixed committer and
dates, the patches' own authors) and refuses any result whose tree is not `6df6ffa158e4bc157dd73e784a5d2c8ae2d2272e`,
the tree the numbers were measured on. `scripts/build-engine.sh DIR` then runs `zig build -Dgpu=sm_120`, one stripped
server with every model family and the kernel images for sm_120, and `zig build grammar` from xgrammar v0.2.8, which
puts `libtfgrammar.so` in `zig-out/native/lib`, where the server loads it from. The grammar step takes under a minute
and needs no GPU; upstream's `zig build test-grammar-tools` passes with it, walking the tool-call grammars through it.

## The series

| patches | author | what |
| --- | --- | --- |
| 0001-0021 | Scott Biggart | TP2 collectives on a blocking split of the nonblocking NCCL communicator when both ranks share a host, and the opt-in `--tp 2` decode-round graph with the dense split (upstream PRs #2978, #2980, #2982) |
| 0022 | ashhart | Flash Next command help on main's CUDA usage module |
| 0023-0026 | Scott Biggart | control-link waits poll briefly before they block (#2971); a rank waits for its peer's load before the first collective (#2972) |
| 0027-0030 | Jay Leaton | discrete cards keep prompt states in pinned host memory, reuse freed pinned buffers, and trim them on both ranks together (`TF_FLASHNEXT_KEPT`) |
| 0031-0032 | Jay Leaton | block-FP8 routed experts load a rank's half and serve at `--tp 2` |
| 0033 | Jay Leaton | the FP8 expert kernel cuts a pair's sum into four K ranges, one a decode warp |
| 0034 | Jay Leaton | `TF_FLASHNEXT_READOUT_FUSED`, from upstream PR #3413 (off in this profile) |
| 0035-0037 | Jay Leaton | the shared expert on the side stream on rank 0 too, and `TF_FLASHNEXT_SHARED_SIDE_SPLIT` (after MiaAI-Lab's patch 0016) |
| 0038-0043 | Jay Leaton | `TF_FLASHNEXT_KV_HOST` (K/V rows on mapped pinned host pages), `TF_FLASHNEXT_SPLIT_ROWS`, keep frames a whole window does not need, `TF_FLASHNEXT_SPLIT_SHARED` |
| 0044 | Jay Leaton | `TF_FLASHNEXT_PEER_SLOTS`: the MoE exchange as owner-slot writes into the peer GPU's mapped buffer |
| 0045 | Jay Leaton | `fn_row_best` split over 32 blocks a row for the MTP rows (byte-equal to the 1024-thread kernel) |
| 0046 | Jay Leaton | the merge resolutions where these branches met (rank wire, the test tool's usage table, recipe notes) |

The measured binary came from merges of these branches; the series is the same work replayed in order onto the pin,
and its last patch carries what the merges resolved, so the final tree is byte-identical. Patches before 0046 are not
each built or measured on their own.

## Stack E's levers

| env | patch | default | in stack E |
| --- | --- | --- | --- |
| `NCCL_P2P_LEVEL=SYS` | NCCL | shared host memory | peer-to-peer over PCIe |
| `TF_FLASHNEXT_CHUNK_SOLO` | upstream | the chunk | 8192 |
| `TF_FLASHNEXT_DRAFTS_GREEDY` | upstream | | 12 |
| `TF_FLASHNEXT_DRAFT_HEAD` | upstream | | `affine4` |
| `TF_FLASHNEXT_SHARED_SIDE` | upstream (0035 adds rank 0 at `--tp 2`) | off | on |
| `TF_FLASHNEXT_SHARED_SIDE_SPLIT` | 0036 | off | on |
| `TF_FLASHNEXT_SPLIT_ROWS` | 0041 | 64 | 16 |
| `TF_FLASHNEXT_WINDOW_GRAPH` | upstream | on | off |
| `TF_QMMF_HALF_BELOW` | upstream | 96 | 400 |
| `TF_FLASHNEXT_PEER_SLOTS` | 0044 | off | on |
| `TF_FLASHNEXT_ATTN_LANES` | upstream | off | on |

Every lever keeps the tokens: with each one on or off, the greedy hashes and the sampled sets are the same.

## Checks run on the measured tree

- TensorFold's kernel contract on sm_120: green, with the split `fn_row_best` byte-equal on every contract shape and
  the new kernels' register and launch-bound records matching their builds.
- `zig fmt --check`, `zig build test-cuda-host` and `zig build test-host` pass; the tree's `pr_check` and lean checks
  add no findings against the base.
- On a CPU-only build host (no GPU): `zig build -Dgpu=sm_120` builds the server (449 s), `zig build grammar` from
  xgrammar v0.2.8 builds `libtfgrammar.so` beside it (41 s), and `zig build test-grammar-tools` passes with that library.
  `scripts/serve.sh` refuses to start when the library is missing (checked with a stand-in server), since
  `tool_choice: "required"` would otherwise fail at request time.
