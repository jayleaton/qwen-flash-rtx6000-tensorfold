# Known issues and limits

## Cold prefill is 2.4% slower with `TF_FLASHNEXT_PEER_SLOTS=1`

Stack E prefills 4,409 / 4,440 tok/s at 8K / 32K against E0's 4,487 / 4,550 (-1.7% / -2.4%). Decode and verify gain
5-7%, so the profile keeps it on. For prefill-heavy work (long documents, few streams), run with
`TF_FLASHNEXT_PEER_SLOTS=0`; the tokens are the same either way.

## `TF_FLASHNEXT_PEER_SLOTS` needs one host with peer-to-peer

Each rank maps the other's receive buffer through CUDA IPC, so both ranks must run on one host with peer access
between the cards. The server refuses it at boot when the ranks sit on two hosts (`PeerSlotsNeedOneHost`) and with
`TF_FLASHNEXT_DECODE_GRAPH=1` (`PeerSlotsWithDecodeGraph`). Passes over 128 rows (prompt chunks) keep the NCCL
collectives.

## Mixed lengths stall decode

When long prompts (8K-32K tokens) arrive among short requests, their prefill chunks hold up the decoding streams. On
E0 with 16 such prompts arriving 400 ms apart, 8 streams gave 124.7 tok/s end to end (525.7 steady), with a p50 round
gap of 51 ms and a p95 of 790 ms; 16 streams gave 157.2 (628.5), p95 811 ms. Uniform traffic is not affected.

**Measured, no env fix; open.** Smaller prompt chunks (`TF_FLASHNEXT_CHUNK=1024`) smooth the p95 round gap but cost
throughput: on E0 at 8 streams 109.2 tok/s end to end against 124.7, and on stack E 112.2 against 127.0 at 8 streams
and 143.5 against 162.4 at 16 (-12%), uniform traffic level. No setting in this engine fixes it.

## `TF_FLASHNEXT_KEEP_LATE=1` is refused on FP8

The server stops at boot with `FlashNextKeepLateUnsupported` on the FP8 checkpoint. Leave it unset.

## Other limits

- The engine's recipe notes ask for `--prompt-cache-gib 0` with drafts at `--tp 2`; the shipped profile keeps the
  default 8 GiB prompt cache, and every number and the soak here ran that way.
- `--idle-unload` is refused with `--tp`.
- Images through the Flash Next vision path were not measured at TP2.
- Only the RTX PRO 6000 Blackwell (sm_120) was measured. Other Blackwell cards with enough memory for 72 GiB a rank
  and peer-to-peer may work but are untested.
