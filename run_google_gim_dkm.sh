#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORTHOLOC_MATCHER='GIM(dkm)'
export ORTHOLOC_GIM_DKM_CKPT="${ORTHOLOC_GIM_DKM_CKPT:-${SCRIPT_DIR}/OrthoLoC/OrthoLoC/model/gim_dkm_100h.ckpt}"

bash "${SCRIPT_DIR}/run_google_0.sh" "$@"
bash "${SCRIPT_DIR}/run_google_1.sh" "$@"
bash "${SCRIPT_DIR}/run_google_2.sh" "$@"
bash "${SCRIPT_DIR}/run_google_3.sh" "$@"
exec bash "${SCRIPT_DIR}/run_google_4.sh" "$@"
