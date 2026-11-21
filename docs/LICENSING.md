# Licensing

| part | license | notes |
| --- | --- | --- |
| this recipe's scripts, configs and docs | Apache-2.0 | [`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE) |
| `patches/` | Apache-2.0 | modifications of TensorFold source; the TensorFold code they modify and quote as diff context stays under TensorFold's license, and the code adapted from MiaAI-Lab keeps its MIT notice |
| TensorFold | Apache-2.0 from 0.6.0; code written before 0.6.0 keeps its MIT notice | fetched from upstream at build time, not vendored; its `LICENSE`, `NOTICE`, `LICENSES/` and `THIRD_PARTY_NOTICES.md` are in every patched checkout |
| MiaAI-Lab recipe code (the shared expert's side stream) | MIT, Copyright (c) 2026 MiaAI-Lab | notice in the patched tree's `THIRD_PARTY_NOTICES.md`, text in `LICENSES/Mia-FlashNext-MIT.txt` |
| Qwen3.8-Flash-Next FP8 weights | Qwen Community License 1.0, per the checkpoint's `LICENSE` | not included; download them from the Qwen team under those terms |

Apache-2.0 asks that a redistribution keeps the license and any NOTICE, marks changed files, and keeps copyright and
attribution notices; MIT asks that its copyright and permission notice travel with the code. This repository keeps
TensorFold's NOTICE text in its own [`NOTICE`](../NOTICE). Each patch is a `git format-patch` record of a commit with
its author, date and message, which states the change. Applying the series yields a tree with TensorFold's license
files unchanged and MiaAI-Lab's notice added.
