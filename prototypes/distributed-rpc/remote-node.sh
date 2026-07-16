#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

cd "${repo_root}"

build_dir="${LLAMA_BUILD_DIR:-${repo_root}/build-rpc}"
bin="${GGML_RPC_SERVER:-${build_dir}/bin/ggml-rpc-server}"

host="${HOST:-127.0.0.1}"
port="${PORT:-50052}"
threads="${THREADS:-}"
device="${DEVICE:-}"
cache="${CACHE:-1}"

if [[ ! -x "${bin}" ]]; then
    printf 'error: %s is not executable\n' "${bin}" >&2
    printf 'hint: run ./prototypes/distributed-rpc/build-rpc.sh first\n' >&2
    exit 1
fi

args=(--host "${host}" --port "${port}")

if [[ -n "${threads}" ]]; then
    args+=(--threads "${threads}")
fi

if [[ -n "${device}" ]]; then
    args+=(--device "${device}")
fi

if [[ "${cache}" != "0" ]]; then
    export LLAMA_CACHE="${LLAMA_CACHE:-${build_dir}/cache}"
    mkdir -p "${LLAMA_CACHE}"
    args+=(--cache)
fi

printf 'Starting RPC node: %s %s\n' "${bin}" "${args[*]}"
exec "${bin}" "${args[@]}"
