# Known issues and limits

## `--pp 2` loses 15% at one stream

`pp2-prefill` and `agent` decode one greedy stream at 304.5 / 209.5 tok/s (code / chat), against `tp2-fast`'s 374.2 /
259.2 and stack E's 357.5 / 246.0. The decode fixes that act under `--pp 2` (batched rollback replay, spinning waits,
the fused router and read-out) leave one stream level and gain 4-7% at 16 streams. A `--pp 2` round runs the two stages
one after the other, so one stream gets one card's speed, and the draft chain adds to the last stage.
`TF_FLASHNEXT_SHARD_HEAD`, `TF_FLASHNEXT_SHARED_REPLICATE` and `TF_FLASHNEXT_GATHER_SLOTS` do nothing under `--pp 2`:
the head lives on the last stage, a stage runs the whole shared expert, and no gathers run inside a layer. The three
round switches in 0084 (now in `pp2-prefill` and `agent`) add 3.5% at one stream (310.6 against 365.1 on `tp2-fast` on
the 2026-11-26 host), so the gap is still about 15%. **Open.** Use `tp2-fast` when one stream matters most.

## `--pp 2`'s prompt cache: measured at 49,152 x 16 only

Since 0085-0086, `--pp 2` keeps prompt states like `--tp 2`.
- **Where it was measured:** at 49,152 x 16 it caches 74% / 82% of the OpenCode replay's prompt tokens, with every
  token equal ([PERFORMANCE](PERFORMANCE.md#patches-0077-0093-2026-11-26)).
- **Where it was not:** the `agent` profile (`--pp 2` at 139,264 x 4) now starts with the default cache budget too. That
  combination has not been booted or replayed. Its levers were measured at 139,264 x 4 only without the cache.
- **To serve as before:** `--prompt-cache-gib 0` turns the cache off.

## `tp2-overlap` is opt-in, at 14 streams

The prompt overlap holds about 2.15 GiB more a rank, so `tp2-overlap` serves 14 streams at a 49,152-token window, not
16. Its prefill is about 40% faster than `tp2-fast`'s, and one stream about 3% faster.

**Its numbers come from the build before 0091.** Before 0091, a lone stream's greedy reply could change at `--tp 2`
with `TF_FLASHNEXT_PLE_EARLY=1` ([QUALITY](QUALITY.md#exactness-of-patches-0077-0093-2026-11-26)). 0091 fixes it,
and a GPU check on 0093 confirms the tokens. The fix changes no reads, so the speed is expected to hold, but it was not
remeasured.

Whether the overlap becomes `tp2-fast`'s default is open.

## Coding agents compact their history at a 49,152-token window

OpenCode sees the server's window and compacts its history when a session nears it: 9-26 times a 5-workflow pass at
49,152 tokens. After a compaction the model quotes file text it no longer has, and most failed `edit` calls (`oldString`
not found) follow one. Use `PROFILE=agent` (or `agent-tp2`), 139,264 tokens x 4 streams: no compactions, 1 mismatch in
134 calls ([QUALITY](QUALITY.md#opencode-tool-calls-on-the-profiles-2026-11-24)).

## Lower power limits are not measured

Every number here ran at the cards' 600 W limit. `nvidia-smi -pl 300` needs host privileges the test container did not
have. A host whose cards are capped at 325 W was tried, but its peer-to-peer link ran at 27 GB/s (PHB topology)
against 53.8 GB/s on the measured host, which skews the TP2 rows, so its numbers were not kept. A capped host with
full peer-to-peer bandwidth is the next measurement.

## Switches in the series that no profile uses

- `TF_FLASHNEXT_OUTPROJ_KSPLIT` (0076) cuts the out projections by input columns and adds the two partials once.
  The sums differ from whole dot products, so the tokens change: +11% code at one stream, 54% of a code reply matches
  RC-tp2's, quality 214/228 against 213/228. Off.
- `TF_FLASHNEXT_HIDDEN_SUM` (not exact), `TF_FLASHNEXT_TP_WINDOW_GRAPH` (-10% at one stream),
  `TF_FLASHNEXT_PP_WINDOW_GRAPH`, `TF_FLASHNEXT_PP_SPEC` (-11%), `TF_FLASHNEXT_WINDOW_PAD` and draft chains over 12:
  measured and not adopted. `TF_FLASHNEXT_EP_LOG` syncs every window; use it for counts only.

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
stack E with 16 such prompts arriving 400 ms apart, 8 streams gave 127.0 tok/s end to end, with a p50 round gap of
44 ms and a p95 of 785 ms; 16 streams gave 162.4, p95 806 ms. Most of the low end-to-end figure is the long prompts'
own prefill (a 32K prompt takes 7.4 s), not the stall. Uniform traffic is not affected.

**Measured, no env fix; open.** Smaller prompt chunks (`TF_FLASHNEXT_CHUNK=1024`) even out the round gaps (p95 331 ms
at 8 streams) but raise the p50 to 315 ms and cost throughput: 112.2 against 127.0 tok/s at 8 streams and 143.5 against
162.4 at 16 (-12%), and the slowest first token at 8 streams goes from 11.3 to 14.3 s. Uniform traffic is level. No
setting in this engine fixes it. On the series to 0090 `tp2-overlap` shortens the round gaps while a prompt prefills by
about 23% (381 against 496 ms p50) and its prompts finish sooner, but decoders still keep only 2-8% of their speed then
([PERFORMANCE](PERFORMANCE.md#patches-0077-0093-2026-11-26)).

## `TF_FLASHNEXT_KEEP_LATE=1` is refused on FP8

The server stops at boot with `FlashNextKeepLateUnsupported` on the FP8 checkpoint. Leave it unset.

## Other limits

- The engine's recipe notes ask for `--prompt-cache-gib 0` with drafts at `--tp 2`; the profiles keep the
  default 8 GiB prompt cache (`--pp 2` too since 0085-0086). Stack E's numbers and soak ran with it. The profiles' cells
  and soaks ran with `--prompt-cache-gib 0`, so every prompt is cold; their OpenCode runs used the default.
- `--idle-unload` is refused with `--tp`.
- Images through the Flash Next vision path were not measured at TP2.
- Only the RTX PRO 6000 Blackwell (sm_120) was measured. Other Blackwell cards with enough memory for 72 GiB a rank
  and peer-to-peer may work but are untested.
