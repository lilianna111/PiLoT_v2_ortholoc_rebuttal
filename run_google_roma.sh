#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORTHOLOC_MATCHER='RoMa'

exec bash "${SCRIPT_DIR}/run_feicuiwan.sh" "$@"
