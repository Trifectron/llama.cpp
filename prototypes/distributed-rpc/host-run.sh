#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

cd "${repo_root}"

build_dir="${LLAMA_BUILD_DIR:-${repo_root}/build-rpc}"
bin="${LLAMA_CLI:-${build_dir}/bin/llama-cli}"

rpc_servers="${RPC_SERVERS:-}"
model="${MODEL:-}"
hf_repo="${HF_REPO:-}"
hf_file="${HF_FILE:-}"
ngl="${N_GPU_LAYERS:-all}"
tensor_split="${TENSOR_SPLIT:-}"
ctx_size="${CTX_SIZE:-}"
prompt="${PROMPT:-Hello from a distributed llama.cpp RPC prototype.}"

if [[ ! -x "${bin}" ]]; then
    printf 'error: %s is not executable\n' "${bin}" >&2
    printf 'hint: run ./prototypes/distributed-rpc/build-rpc.sh first\n' >&2
    exit 1
fi

args=(--split-mode layer --n-gpu-layers "${ngl}")

if [[ -n "${rpc_servers}" ]]; then
    args+=(--distributed "${rpc_servers}")
fi

if [[ -n "${tensor_split}" ]]; then
    args+=(--tensor-split "${tensor_split}")
fi

if [[ -n "${ctx_size}" ]]; then
    args+=(--ctx-size "${ctx_size}")
fi

if [[ -n "${model}" ]]; then
    args+=(--model "${model}")
elif [[ -n "${hf_repo}" ]]; then
    args+=(--hf-repo "${hf_repo}")
    if [[ -n "${hf_file}" ]]; then
        args+=(--hf-file "${hf_file}")
    fi
else
    printf 'error: set MODEL=/path/to/model.gguf or HF_REPO=user/model:quant\n' >&2
    exit 1
fi

printf 'Starting host run: %s %s\n' "${bin}" "${args[*]}"
exec "${bin}" "${args[@]}" "${prompt}"
