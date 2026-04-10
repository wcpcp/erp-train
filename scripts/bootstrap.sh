#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv "${ROOT_DIR}/.venv"
source "${ROOT_DIR}/.venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${ROOT_DIR}/third_party/ms-swift"
python -m pip install -e "${ROOT_DIR}"

cat <<EOF
Bootstrap complete.

If torch is not installed yet on the server, install the CUDA-matching build now.
Then run:

  source ${ROOT_DIR}/.venv/bin/activate
EOF

