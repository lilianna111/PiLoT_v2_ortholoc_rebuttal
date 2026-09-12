#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

matchers=(
  "eloftr"
  "loftr"
  "minima(RoMa)"
  "RoMa"
)

for matcher in "${matchers[@]}"; do
  echo "==== matcher: ${matcher} ===="
  ORTHOLOC_MATCHER="${matcher}" bash "${SCRIPT_DIR}/run_feicuiwan.sh" "$@"
done
