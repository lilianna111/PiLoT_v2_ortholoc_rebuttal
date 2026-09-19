#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORTHOLOC_MATCHER='RoMa'
export ORTHOLOC_PNP_PRIOR="${ORTHOLOC_PNP_PRIOR:-gravity}"
export ORTHOLOC_PRIOR_FUSION="${ORTHOLOC_PRIOR_FUSION:-balanced}"
export ORTHOLOC_GRAVITY_PRIOR_DIR="${ORTHOLOC_GRAVITY_PRIOR_DIR:-/media/amax/AE0E2AFD0E2ABE69/datasets/angle}"
export ORTHOLOC_DEPTH_PRIOR_DIR="${ORTHOLOC_DEPTH_PRIOR_DIR:-/media/amax/AE0E2AFD0E2ABE69/datasets/depth_1}"
export ORTHOLOC_OUTPUT_ROOT="${ORTHOLOC_OUTPUT_ROOT:-/media/amax/PS2000/rebuttal/ortholoc}"

# Read image roll_deg pitch_deg from angle/<sequence>.txt.
# Extra arguments override defaults, e.g. --ortholoc_pnp_prior none.
# Balanced mode keeps visual P3P / MSAC / LO-RANSAC selection and applies
# sensor priors only in guarded final pose-only BA.
# Depth is opt-in: --ortholoc_pnp_prior depth or gravity_depth.
# Residual scales default to 10 deg / 10 m, NOT rejection thresholds.
# Legacy hard gating and soft P3P scoring remain available explicitly.

exec bash "${SCRIPT_DIR}/run_feicuiwan.sh" "$@"
