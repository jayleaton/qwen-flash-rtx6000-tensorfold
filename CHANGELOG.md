# Changelog

## Unreleased: patches 0077-0093

- **Engine:** patches 0077-0093 on the same pin, tree `4c9476f45d0d`. They are `fp8/combined-improvements` 1b3ef220d
  linearized, and every new switch is off by default:
  - the `--tp 2` prompt overlap (`TF_FLASHNEXT_PROMPT_OVERLAP`, `TF_FLASHNEXT_OWNER_NOCLEAR`,
    `TF_FLASHNEXT_OVERLAP_OUTPROJ`);
  - the round switches (`TF_FLASHNEXT_PLE_EARLY`, `TF_FLASHNEXT_PP_HEAD_SPLIT`, `TF_FLASHNEXT_ROUND_TURN`);
  - the `--pp 2` prompt cache, which keeps a state at a chunk's mark with the MTP head's deferred rows (no
    `NotAtMark`);
  - 0091, the fix for rank 1's early n-gram rows at `--tp 2`, with a host test.
- **Profiles:**
  - **`pp2-prefill` and `agent`:**
    - They take the three round switches, and the prompt cache comes on at the default budget.
    - At 49,152 x 16: one stream +3.5%; the OpenCode replay caches 74% / 82% of prompt tokens; five workflows at once
      run 209.7 tok/s against 104.3.
    - `--prompt-cache-gib 0` serves as before.
  - **`tp2-overlap`** is a new opt-in profile: `tp2-fast` + the overlap + the early n-gram rows, at 49,152 x 14.
    - Prefill is about 40% faster (cold 8K / 32K / 128K +38% / +41% / +38%) and one stream +3%.
    - Its speed comes from the series to 0090. After 0091 it is inferred, not remeasured.
  - **`tp2-fast`, `agent-tp2` and `stack-e`** launch exactly as on 0076.
- **Measured** on 2x RTX PRO 6000 Workstation Edition at 600 W, on the series to 0090:
  - every arm token-identical to the 0076 recipe, and every 128K reply the recipe's;
  - the OpenCode replay and the mixed-traffic decoders token-equal on every `--pp 2` arm and on `tp2-fast`;
  - a 15-minute soak of `tp2-overlap` with the cache: STABLE.
  - `tp2-overlap` split one OpenCode turn and one lone reply before 0091. A GPU check on 0093 reproduced the fault
    without the fix and found every reply equal with it ([QUALITY](docs/QUALITY.md)).
- **Prebuilt:**
  - No release for this tree yet; the README's prebuilt is still the 0076 server.
  - That server ignores the new switches, so the new profile behaviour needs `scripts/build-engine.sh`.

## 2026-11-25: rc-20261124

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
