#!/usr/bin/env bash
# Fetch and patch the engine (scripts/fetch-engine.sh), then build one stripped tensorfold-native for the RTX PRO 6000.
#
#   scripts/build-engine.sh DEST [SOURCE]
#
# NVCC (default /usr/local/cuda/bin/nvcc) builds the kernel images; GPU (default sm_120, Blackwell RTX) picks the
# architecture. The server lands in DEST/zig-out/native/bin/tensorfold-native.
# XGRAMMAR=DIR also builds libtfgrammar (tool_choice required or named, response_format) from an xgrammar v0.2.8
# checkout; XGRAMMAR=clone clones it into DEST/.xgrammar. Unset, those requests get HTTP 400. The measured server was
# built without it.
set -euo pipefail
dest=${1:?usage: scripts/build-engine.sh DEST [SOURCE]}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
"$root/scripts/fetch-engine.sh" cuda "$@"
want=$(cat "$dest/.zig-version")
have=$(zig version)
[[ "$have" == "$want" ]] || { echo "zig $have found; the engine needs zig $want (.zig-version)" >&2; exit 2; }
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
[[ -x "$nvcc" ]] || { echo "no nvcc at $nvcc: set NVCC" >&2; exit 2; }
cd "$dest"
zig build -Dgpu="${GPU:-sm_120}" -Dnvcc="$nvcc"
[[ -x zig-out/native/bin/tensorfold-native ]] || { echo "the build wrote no zig-out/native/bin/tensorfold-native" >&2; exit 2; }
xgrammar_commit=97787376faee5ed8466cfad57c99855e4ce2f6aa # v0.2.8
xg=${XGRAMMAR:-}
if [[ "$xg" == clone ]]; then
    xg=$PWD/.xgrammar
    if [[ ! -d "$xg/.git" ]]; then
        git clone -q --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar "$xg.part"
        mv "$xg.part" "$xg"
    fi
    grep -qx '/.xgrammar' .git/info/exclude 2>/dev/null || echo '/.xgrammar' >> .git/info/exclude
fi
if [[ -n "$xg" ]]; then
    got=$(git -C "$xg" rev-parse HEAD)
    [[ "$got" == "$xgrammar_commit" ]] || { echo "$xg is xgrammar $got, not v0.2.8 ($xgrammar_commit)" >&2; exit 2; }
    sm=${GPU:-sm_120}
    zig build grammar -Dsm="${sm#sm_}" -Dxgrammar="$(cd "$xg" && pwd)"
    [[ -f zig-out/native/lib/libtfgrammar.so ]] || { echo "zig build grammar wrote no zig-out/native/lib/libtfgrammar.so" >&2; exit 2; }
fi
echo "built $dest/zig-out/native/bin/tensorfold-native"
