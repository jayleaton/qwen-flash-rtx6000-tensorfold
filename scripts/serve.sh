#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next FP8 on two GPUs of one host with one of the measured profiles (config/cuda/profiles).
#
#   scripts/serve.sh ENGINE_DIR MODEL_DIR [extra tensorfold-native serve flags]
#
# Starts rank 1 in the background, then rank 0 in the foreground; rank 0 serves HTTP on PORT (default 8090), rank 1
# takes PORT+1 for its own port. When either the script is stopped or rank 0 exits, both ranks are stopped.
#   GPUS=0,1        the CUDA devices of rank 0 and rank 1
#   PROFILE=NAME    config/cuda/profiles/NAME.{args,env} (default tp2-fast): tp2-fast, pp2-prefill, agent, agent-tp2,
#                   stack-e; NAME.args holds the window, streams and --tp 2 or --pp 2, NAME.env the levers on top
#   NAME=flash      the served model name
#   NUMA_NODE=N     run both ranks under numactl --cpunodebind=N --preferred=N (the node the GPUs sit on)
#   LOG_DIR=DIR     where rank 1 logs (default ENGINE_DIR); rank 0 logs to the terminal
#   NO_GRAMMAR=1    start without libtfgrammar (tool_choice required or named and response_format then fail)
# NAME.env, then config/cuda/serve.env (stack E), are exported (KEY=VALUE lines); a variable already set wins.
set -euo pipefail
engine=${1:?usage: scripts/serve.sh ENGINE_DIR MODEL_DIR [flags]}
model=${2:?usage: scripts/serve.sh ENGINE_DIR MODEL_DIR [flags]}
shift 2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=$root/config/cuda
bin=$engine/zig-out/native/bin/tensorfold-native
[[ -x "$bin" ]] || { echo "no $bin: run scripts/build-engine.sh first" >&2; exit 2; }
[[ -f "$model/config.json" ]] || { echo "$model has no config.json" >&2; exit 2; }
if [[ ! -f "$engine/zig-out/native/lib/libtfgrammar.so" && "${NO_GRAMMAR:-0}" != 1 ]]; then
    echo "no $engine/zig-out/native/lib/libtfgrammar.so: tool_choice required or named and response_format would fail;" \
        "rebuild with scripts/build-engine.sh (without XGRAMMAR=none), or set NO_GRAMMAR=1 to start without it" >&2
    exit 2
fi
name=${PROFILE:-tp2-fast}
[[ "$name" =~ ^[a-z0-9-]+$ && -f "$config/profiles/$name.args" && -f "$config/profiles/$name.env" ]] ||
    { echo "no profile $name in $config/profiles" >&2; exit 2; }
read -r -a profile < "$config/profiles/$name.args"
for env in "$config/profiles/$name.env" "$config/serve.env"; do
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || { echo "bad line in $env: $key" >&2; exit 2; }
        [[ -n "${!key+x}" ]] || export "$key=$value"
    done < "$env"
done
IFS=, read -r gpu0 gpu1 <<< "${GPUS:-0,1}"
[[ "$gpu0" =~ ^[0-9]+$ && "$gpu1" =~ ^[0-9]+$ && "$gpu0" != "$gpu1" ]] || { echo "GPUS must name two devices, like 0,1" >&2; exit 2; }
port=${PORT:-8090}
wrap=()
[[ -n "${NUMA_NODE:-}" ]] && wrap=(numactl --cpunodebind="$NUMA_NODE" --preferred="$NUMA_NODE")
common=(serve "$model" --name "${NAME:-flash}" --no-update-check "${profile[@]}" "$@" --master 127.0.0.1)
log=${LOG_DIR:-$engine}/rank1.log
CUDA_VISIBLE_DEVICES=$gpu1 ${wrap[@]+"${wrap[@]}"} "$bin" "${common[@]}" --rank 1 --port $((port + 1)) > "$log" 2>&1 &
rank1=$!
rank0=
stop_ranks() {
    local rc=$?
    kill $rank0 "$rank1" 2>/dev/null || true
    wait $rank0 "$rank1" 2>/dev/null || true
    exit "$rc"
}
trap stop_ranks EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "profile $name (${profile[*]}); rank 1 on GPU $gpu1 (pid $rank1, log $log)" >&2
echo "rank 0 on GPU $gpu0 serves http://127.0.0.1:$port" >&2
CUDA_VISIBLE_DEVICES=$gpu0 ${wrap[@]+"${wrap[@]}"} "$bin" "${common[@]}" --rank 0 --port "$port" &
rank0=$!
wait "$rank0"
