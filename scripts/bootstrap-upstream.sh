#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/upstream/llama.cpp}"
URL="https://github.com/ggml-org/llama.cpp.git"

if [[ -e "$DEST" ]]; then
  echo "Refusing to overwrite existing path: $DEST" >&2
  echo "Move it aside or pass a different destination." >&2
  exit 2
fi

mkdir -p "$(dirname "$DEST")"
echo "Cloning upstream llama.cpp into: $DEST"
git clone --depth 1 "$URL" "$DEST"

echo
echo "Upstream source is present. No QNN graph offload has been installed by this script."
echo "Next: build/test the standalone QNN probe, then implement and verify a GGML backend"
echo "against this upstream checkout. Do not enable HTP for model inference until graph"
echo "execution and CPU-reference parity tests pass."
