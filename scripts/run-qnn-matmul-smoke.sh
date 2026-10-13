#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${QNN_MATMUL_BIN:-$ROOT/build-qnn-probe/qnn-matmul-smoke}"
LIBDIR="${1:-}"

if [[ -z "$LIBDIR" ]]; then
  echo "Usage: bash scripts/run-qnn-matmul-smoke.sh /path/to/qnn/libs" >&2
  exit 64
fi
if [[ ! -x "$BIN" ]]; then
  echo "Matmul smoke binary not found: $BIN" >&2
  echo "Build it first with:" >&2
  echo "  cmake -S npu -B build-qnn-probe -DQNN_INC=/path/to/QNN/include" >&2
  echo "  cmake --build build-qnn-probe -j1" >&2
  exit 2
fi
if [[ ! -f "$LIBDIR/libQnnHtp.so" ]]; then
  echo "Missing $LIBDIR/libQnnHtp.so" >&2
  exit 3
fi

export LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$LIBDIR;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"

echo "QNN_LIBDIR=$LIBDIR"
echo "ADSP_LIBRARY_PATH=$ADSP_LIBRARY_PATH"
exec "$BIN" "$LIBDIR"
