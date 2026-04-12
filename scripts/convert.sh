#!/usr/bin/env bash
set -euo pipefail

# Convert a HuggingFace model to TurboQuant format.
# Usage: ./scripts/convert.sh <model-name-or-path> [bits] [residual-bits]

MODEL="${1:?Usage: convert.sh <model> [bits] [residual-bits]}"
BITS="${2:-4}"
RESIDUAL="${3:-4}"
OUTPUT="${MODEL}-tq${BITS}"

echo "Converting ${MODEL} to TQ${BITS}+${RESIDUAL}..."

tq-convert \
  --model "${MODEL}" \
  --bits "${BITS}" \
  --residual-bits "${RESIDUAL}" \
  --output "${OUTPUT}"

echo "Validating..."
python scripts/validate.py --model "${OUTPUT}"

echo "Done: ${OUTPUT}"
