# Capacity

The server checks its CUDA budget at boot and refuses a `--parallel` over it, naming the largest that fits. With the
FP8 weights at TP2 each rank holds 71.74 GiB of model; stack E's banner at `--context 49152` reads 16 streams at
1.14 GiB each, of 18.8 GiB left after a 3.0 GiB reserve.

## Streams per window (FP8, TP2)

| window | on the cards | `TF_FLASHNEXT_KV_HOST=overflow` |
| --- | --- | --- |
| 49,152 | **16** (stack E, the shipped profile) | |
| 64K | **13** (1.43 GiB a stream; 14 needs 19.98 GiB of an 18.86 GiB budget) | **27** (0.68 GiB a stream) |
| 128K | **7** (2.56 GiB a stream; 8 needs 20.44 GiB) | **17** (1.06 GiB a stream) |
| 256K | not run | **10** (1.81 GiB a stream) |

- The 64K-256K rows were measured on stack D's binary and env; stack E's per-stream size at those windows was not
  re-measured.
- Every served count ran its full code and chat load with 0 failures. Host RAM was not the limit at any window.
- `TF_FLASHNEXT_KV_HOST=overflow` puts K/V rows that do not fit on the card in mapped pinned host pages: 2.1x the
  streams at 64K and 2.4x at 128K. Its speed with stack E has not been measured.

To serve a longer window, edit `config/cuda/serve.args`, for example to `--context 131072 --parallel 7`. Stack E's speed
numbers are for the shipped 49,152-token profile.

## The fp8 KV cache (`--kv-cache fp8`): not in this series

Flash Next's fp8 KV cache (e4m3 rows with a power-of-two scale, adapted from MiaAI-Lab's patches 0003 and 0005) was
measured on a later branch of this line; this engine does not include it. If it lands, it stays opt-in and off by
default, for these reasons:

| | bf16 KV | fp8 KV | change |
| --- | --- | --- | --- |
| FP8 TP2, 128K prefill (A-B-B-A) | 3,765 / 3,769 | 3,885 / 3,788 | +1.8% |
| FP4-Local TP1, 128K prefill (A-B-B-A) | 6,651 / 6,647 | 7,060 / 7,071 | +6.3% |
| FP8 TP2 fidelity, fp8 against bf16 KV (2,080 rows) | reference | top-1 90.9%, KL 0.038 | |
| FP4-Local TP1 fidelity (DGX Spark, 2,080 rows) | reference | top-1 92.6%, KL 0.0236 | |

- **Recommended only for FP4-Local at TP1**, where its KL of 0.0236 sits under the 0.025 spread between valid FP8 sum
  orders and it gives +6.3% at 128K prefill and 1.83x the slots (6 to 11 at 128K on a DGX Spark).
- Capacity with the fp8 KV cache, FP8 at TP2 (its branch, no stack env, `--prompt-cache-gib 0`): bf16 KV serves 9
  streams at 128K and 4 at 256K; the fp8 KV arm is pending.
- **Not recommended with this recipe (FP8 at TP2):** KL 0.038 for +1.8% prefill. For scale, 4-bit weights against FP8
  give KL 0.10-0.13.
- The 128K prefill rows ran without the stack env on a different runner, so their absolutes sit below stack D's
  4,330 tok/s.
