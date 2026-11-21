#!/usr/bin/env bash
# Check a downloaded Qwen3.8-Flash-Next-FP8 tree against the revision the numbers were measured on.
#
#   scripts/verify-weights.sh MODEL_DIR
#
# config/weights/fp8.sha256 holds the sha256 of each LFS file (the 131 shards, the index and tokenizer.json);
# config/weights/fp8.blob holds the git blob sha1 of each small file. Hashing 173 GiB takes a few minutes.
set -euo pipefail
model=${1:?usage: scripts/verify-weights.sh MODEL_DIR}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ -d "$model" ]] || { echo "no directory $model" >&2; exit 2; }
fail=0
(cd "$model" && sha256sum --quiet -c "$root/config/weights/fp8.sha256") || fail=1
while read -r want path; do
    if [[ ! -f "$model/$path" ]]; then
        echo "$path: missing"; fail=1; continue
    fi
    got=$(git hash-object --no-filters "$model/$path")
    [[ "$got" == "$want" ]] || { echo "$path: blob $got, want $want"; fail=1; }
done < "$root/config/weights/fp8.blob"
((fail == 0)) || { echo "verify-weights: $model differs from Qwen/Qwen3.8-Flash-Next-FP8@$(cat "$root/config/weights/fp8.rev")" >&2; exit 1; }
echo "verify-weights: $model is Qwen/Qwen3.8-Flash-Next-FP8@$(cat "$root/config/weights/fp8.rev") ($(wc -l < "$root/config/weights/fp8.sha256") LFS files, $(wc -l < "$root/config/weights/fp8.blob") small files)"
