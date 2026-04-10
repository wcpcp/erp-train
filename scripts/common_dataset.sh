#!/usr/bin/env bash

_resolve_dataset_python() {
  if command -v python >/dev/null 2>&1; then
    echo python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo python3
    return 0
  fi
  echo "Unable to find python or python3 for dataset normalization." >&2
  return 1
}

normalize_dataset_for_swift() {
  local input_path="${1:-}"
  local split_name="${2:-dataset}"
  local python_bin
  python_bin="$(_resolve_dataset_python)"

  if [[ -z "${input_path}" ]]; then
    echo ""
    return 0
  fi

  if [[ "${AUTO_NORMALIZE_DATASETS:-true}" != "true" ]]; then
    echo "${input_path}"
    return 0
  fi

  local cache_dir="${NORMALIZED_DATA_DIR:-${ROOT_DIR}/.cache/normalized_datasets}"
  mkdir -p "${cache_dir}"

  local input_abs
  input_abs="$("${python_bin}" - <<'PY' "${input_path}"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

  local fingerprint
  fingerprint="$("${python_bin}" - <<'PY' "${input_abs}" "${DATA_SYSTEM_PROMPT:-}" "${INJECT_ERP_METADATA:-false}"
from pathlib import Path
import hashlib
import os
import sys

path = Path(sys.argv[1])
prompt = sys.argv[2]
inject = sys.argv[3]
payload = f"{path}|{path.stat().st_mtime_ns}|{path.stat().st_size}|{prompt}|{inject}"
print(hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16])
PY
)"

  local output_path="${cache_dir}/${split_name}_${fingerprint}.jsonl"
  if [[ ! -f "${output_path}" ]]; then
    local -a cmd=("${python_bin}" -m pano_qwen_erp.data.prepare_sft --input "${input_abs}" --output "${output_path}")
    if [[ -n "${DATA_SYSTEM_PROMPT:-}" ]]; then
      cmd+=(--system-prompt "${DATA_SYSTEM_PROMPT}")
    fi
    if [[ "${INJECT_ERP_METADATA:-false}" == "true" ]]; then
      cmd+=(--inject-erp-metadata)
    fi
    "${cmd[@]}"
  fi

  echo "${output_path}"
}
