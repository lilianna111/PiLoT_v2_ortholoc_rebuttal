#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK_PYTHON="$(command -v "${PYTHON_EXECUTABLE:-python}")"

cmake -S "${REPO_DIR}/native/gravity_pnp" -B "${REPO_DIR}/native/gravity_pnp/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="${TASK_PYTHON}" \
  -DGRAVITY_OUTPUT_DIR="${REPO_DIR}/OrthoLoC/OrthoLoC/ortholoc"
cmake --build "${REPO_DIR}/native/gravity_pnp/build" --parallel "${BUILD_JOBS:-2}"
PYTHONPATH="${REPO_DIR}/OrthoLoC/OrthoLoC${PYTHONPATH:+:${PYTHONPATH}}" \
  "${TASK_PYTHON}" -B -c 'from ortholoc import _gravity_pnp; print("Gravity PnP built:", _gravity_pnp.__file__)'
