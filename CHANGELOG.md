# Changelog

## rc-20261124 (not released)

- **Profiles.** `scripts/serve.sh` takes `PROFILE=` and reads `config/cuda/profiles/NAME.{args,env}` on top of stack E's
  `config/cuda/serve.env`; `config/cuda/serve.args` is gone, and the window, streams and `--tp 2` or `--pp 2` live in
  each profile's args.
  - **`tp2-fast`** (the new default) is `--tp 2` with the owner-slot prompt exchange, the split DeltaNet prompt chain
    and seven exact decode fixes. Against stack E: code x1 +4.7%, x16 +6.2%, chat x16 +9.4%, cold 32K +32%.
  - **`pp2-prefill`** is `--pp 2` with the split chain and the four fixes that act there: cold 32K 14,807 and 128K
    15,964 tok/s, code x16 1,293. It loses 15% at one stream.
  - **`agent`** and **`agent-tp2`** are those two at 139,264 tokens x 4 streams, for coding agents.
  - **`stack-e`** is the previous profile, unchanged.
- **Engine.** Patches 0047-0076 on the same pin: `TF_FLASHNEXT_OWNER_SLOTS`, `TF_FLASHNEXT_GDN_SPLIT`, `--pp 2`, and the
  decode fixes, each off by default.
  - The decode fixes: `TF_FLASHNEXT_ROLL_BATCH`, `TF_FLASHNEXT_SPIN_WAIT`, `TF_FLASHNEXT_SHARD_HEAD`'s best-row
    exchange, `TF_FLASHNEXT_SHARED_REPLICATE`, `TF_FLASHNEXT_GATHER_SLOTS`, `TF_FLASHNEXT_ROUTE_FUSED` and
    `TF_FLASHNEXT_READOUT_DECODE`.
  - Also in the series and off: the switches measured and not adopted, `TF_FLASHNEXT_EP_LOG`, and the not-exact
    `TF_FLASHNEXT_OUTPROJ_KSPLIT`.
  - The tree is `fa72a8b22f5f`.
- **Measured** on 2x RTX PRO 6000 Workstation Edition at 600 W:
  - every cell token-identical to stack E;
  - 30-minute soaks of `tp2-fast` and `pp2-prefill`, both STABLE;
  - quality suite 213/228;
  - the new kernels' contract and byte checks pass;
  - OpenCode tool-call stats per profile ([QUALITY](docs/QUALITY.md)).
- **Known issues:** `--pp 2` at one stream, `--pp 2` without a prompt cache, agents compacting at 49,152 tokens, and
  power limits below 600 W not measured ([KNOWN-ISSUES](docs/KNOWN-ISSUES.md)).

## 2026-11-21: stack E

- Patches 0001-0046 on TensorFold `3661a4281`, the stack E profile, HumanEval+ 93.3, BFCL held-out 74.7, OpenCode
  10/10 at 128K and 9/10 at 256K, a 35-minute mixed-length soak.
