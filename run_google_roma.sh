#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORTHOLOC_MATCHER='RoMa'
export ORTHOLOC_PNP_PRIOR="${ORTHOLOC_PNP_PRIOR:-gravity}"
export ORTHOLOC_GRAVITY_PRIOR_DIR="${ORTHOLOC_GRAVITY_PRIOR_DIR:-/media/amax/AE0E2AFD0E2ABE69/datasets/angle}"
export ORTHOLOC_OUTPUT_ROOT="${ORTHOLOC_OUTPUT_ROOT:-/media/amax/PS2000/rebuttal/ortholoc}"

# Read image roll_deg pitch_deg from angle/<sequence>.txt.
# Extra arguments override defaults, e.g. --ortholoc_pnp_prior none.

exec bash "${SCRIPT_DIR}/run_feicuiwan.sh" "$@"
