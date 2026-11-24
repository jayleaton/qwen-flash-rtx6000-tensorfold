# Engine

`config/cuda/` pins upstream TensorFold `main` at `3661a4281425a71eb5f2a56922f0f2a33e5f4f17` (2026-11-20).
`scripts/fetch-engine.sh cuda DIR` fetches it, applies `patches/cuda/0001`-`0076` with `git am` (fixed committer and
dates, the patches' own authors) and refuses any result whose tree is not `fa72a8b22f5fd1fc68e2567006744d1d907774a9`,
the tree the profiles were measured on. The series to `0046` gives stack E's tree,
`6df6ffa158e4bc157dd73e784a5d2c8ae2d2272e`. `scripts/build-engine.sh DIR` then runs `zig build -Dgpu=sm_120`, one
stripped server with every model family and the kernel images for sm_120, and `zig build grammar` from xgrammar
v0.2.8, which puts `libtfgrammar.so` in `zig-out/native/lib`, where the server loads it from. The grammar step takes
under a minute and needs no GPU; upstream's `zig build test-grammar-tools` passes with it, walking the tool-call
grammars through it.

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
| 0047-0048 | Jay Leaton | MoE slots summed in owner order; `TF_FLASHNEXT_OWNER_SLOTS`, the `--tp 2` prompt exchange of one hidden-size row a layer |
| 0049 | Jay Leaton | `TF_FLASHNEXT_GDN_SPLIT`: the DeltaNet prompt chain split over (value head, slice) where the SMs allow |
| 0050-0054 | Jay Leaton | the `--tp 2` prompt exchange sends only each rank's half rows; the prefill lanes behind load-time switches; `TF_FLASHNEXT_HIDDEN_SUM` (not exact, off) |
| 0055-0063 | Jay Leaton | `--pp 2`: the two ranks as pipeline stages over the rank link, prompt chunks overlapped across the stages, stage windows as graphs |
| 0064-0068 | Jay Leaton | verify and stage window graphs, draft chains up to 24, `TF_FLASHNEXT_WINDOW_PAD`, `TF_FLASHNEXT_PP_SPEC` (all measured, none adopted) |
| 0069 | Jay Leaton | `TF_FLASHNEXT_ROLL_BATCH`: a rollback's DeltaNet replay in three launches; `TF_FLASHNEXT_SPIN_WAIT`: rank waits without sleeps |
| 0070 | Jay Leaton | `TF_FLASHNEXT_SHARD_HEAD`'s greedy verify exchanges each half's (best, index) rows, 16 B a row, not the scores |
| 0071 | Jay Leaton | `TF_FLASHNEXT_SHARED_REPLICATE`: each rank also runs the peer's shared-expert half, so the shared gather goes |
| 0072 | Jay Leaton | `TF_FLASHNEXT_GATHER_SLOTS`: the out projection's gathers through the peer's mapped buffer |
| 0073 | Jay Leaton | `TF_FLASHNEXT_ROUTE_FUSED`: the router in two launches instead of four |
| 0074 | Jay Leaton | `TF_FLASHNEXT_READOUT_DECODE`: a decode pass's hyper-connection read-out in two launches instead of five |
| 0075 | Jay Leaton | `TF_FLASHNEXT_EP_LOG`: each rank's distinct experts a layer, for diagnosis (syncs; off) |
| 0076 | Jay Leaton | `TF_FLASHNEXT_OUTPROJ_KSPLIT`: out projections cut by input columns (not exact; off) |

Stack E's binary came from merges of these branches; 0001-0046 replay that work in order onto the pin, and 0046
carries what the merges resolved, so its tree is byte-identical. 0047-0076 are the measured binary's own linear
history on top of that tree. Patches are not each built or measured on their own, only 0046 and 0076.

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

## The profiles' levers (on top of stack E)

| env | patch | `tp2-fast` / `agent-tp2` | `pp2-prefill` / `agent` | under `--pp 2` |
| --- | --- | --- | --- | --- |
| `TF_FLASHNEXT_OWNER_SLOTS=1` | 0047 | on | | a `--tp 2` exchange |
| `TF_FLASHNEXT_GDN_SPLIT=1` | 0049 | on | on | acts |
| `TF_FLASHNEXT_ROLL_BATCH=1`, `TF_FLASHNEXT_SPIN_WAIT=1` | 0069 | on | on | acts |
| `TF_FLASHNEXT_SHARD_HEAD=1` | 0070 | on | | does nothing: the head lives on the last stage |
| `TF_FLASHNEXT_SHARED_REPLICATE=1` | 0071 | on | | does nothing: a stage runs the whole shared expert |
| `TF_FLASHNEXT_GATHER_SLOTS=1` | 0072 | on | | does nothing: no gathers inside a layer |
| `TF_FLASHNEXT_ROUTE_FUSED=1` | 0073 | on | on | acts |
| `TF_FLASHNEXT_READOUT_DECODE=1` | 0074 | on | on | acts |

Each of these keeps the greedy tokens (every cell of every profile hashed equal), and each is off by default in the
engine. With all of them off, the 0076 tree serves stack E's launches.

## Checks run on the measured tree

For 0076 (`fa72a8b22f5f`):
- `zig fmt --check`, the tree's `pr_check` (0 new findings against the base, lean unchanged), `zig build test-cuda-host`
  and the golden tests: 0 new failures against the base; the env table holds every new switch.
- On the GPU: `tf-cuda-test route-rows` (spread logits and the fused top-k and plan byte-equal to the kernels they
  replace) and the kernel contract on the new read-out, router and head-merge kernels (gates 1-4 pass).
- `scripts/serve.sh` with a stand-in server for every profile: each rank's env and launch equal the measured cells'.

- The server rebuilt from this series (`zig build -Dgpu=sm_120 -Dversion=0.0.0 -Dtarget=x86_64-linux-gnu.2.28`, nvcc
  13.0.88) is byte-identical to the measured binary, sha256
  `f39b0f158a3d9734ab4005e54db65a0fc4a4f768b4608ff68cb6ea83c093b0a5`.
  `zig build grammar` from xgrammar v0.2.8 builds `libtfgrammar.so`, and `zig build test-grammar-tools` passes with it.
  Both ship in the release tarball (sha256 `a04c0568cc9b…`).

For 0046 (`6df6ffa158e4`):

- TensorFold's kernel contract on sm_120: green, with the split `fn_row_best` byte-equal on every contract shape and
  the new kernels' register and launch-bound records matching their builds.
- `zig fmt --check`, `zig build test-cuda-host` and `zig build test-host` pass; the tree's `pr_check` and lean checks
  add no findings against the base.
- On a CPU-only build host (no GPU): `zig build -Dgpu=sm_120` builds the server (449 s), `zig build grammar` from
  xgrammar v0.2.8 builds `libtfgrammar.so` beside it (41 s), and `zig build test-grammar-tools` passes with that
  library.
  `scripts/serve.sh` refuses to start when the library is missing (checked with a stand-in server), since
  `tool_choice: "required"` would otherwise fail at request time.
