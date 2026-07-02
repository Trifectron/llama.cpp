#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

cd "${repo_root}"

build_dir="${LLAMA_BUILD_DIR:-${repo_root}/build-rpc}"
jobs="${JOBS:-}"
targets_raw="${TARGETS:-ggml-rpc-server llama-cli llama-distributed-tap}"

read -r -a targets <<< "${targets_raw}"

if [[ -z "${jobs}" ]]; then
    jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4')"
fi

cmake -S . -B "${build_dir}" -DGGML_RPC=ON "$@"
cmake --build "${build_dir}" --config Release --target "${targets[@]}" -j "${jobs}"

printf '\nBuilt RPC prototype binaries in %s/bin\n' "${build_dir}"
