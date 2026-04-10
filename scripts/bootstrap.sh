#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==5.2.*}"
QWEN_VL_UTILS_SPEC="${QWEN_VL_UTILS_SPEC:-qwen_vl_utils>=0.0.14}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CANDIDATES=("${PYTHON_BIN}")
else
  PYTHON_CANDIDATES=(python3.12 python3.11 python3.10 python3)
fi

PYTHON_BIN=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    PYTHON_BIN="${candidate}"
    break
  fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No usable Python interpreter found. Please install Python 3.10+ and rerun." >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if ! "${PYTHON_BIN}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  echo "Bootstrap requires Python 3.10+, but ${PYTHON_BIN} is ${PYTHON_VERSION}." >&2
  echo "Set PYTHON_BIN to a Python 3.10/3.11/3.12 executable, for example:" >&2
  echo "  PYTHON_BIN=python3.11 bash scripts/bootstrap.sh" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${ROOT_DIR}/.venv"
source "${ROOT_DIR}/.venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${ROOT_DIR}/third_party/ms-swift"
python -m pip install -U "${TRANSFORMERS_SPEC}" "${QWEN_VL_UTILS_SPEC}"
if ! python -m pip install -U decord; then
  cat <<'EOF'
Warning: failed to install `decord`.
This is expected on some local macOS environments. If you only train on images, you can continue.
For server-side video support, install decord on a Linux Python 3.10+ environment.
EOF
fi
python -m pip install -e "${ROOT_DIR}"

cat <<EOF
Bootstrap complete.

If torch is not installed yet on the server, install the CUDA-matching build now.
The bootstrap script also installs a Qwen3.5-compatible stack:

  ${TRANSFORMERS_SPEC}
  ${QWEN_VL_UTILS_SPEC}
  decord (best-effort; optional for image-only local runs)

Then run:

  source ${ROOT_DIR}/.venv/bin/activate
EOF
