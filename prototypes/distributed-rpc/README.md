# Distributed RPC Prototype

This is a local prototype for testing distributed layer placement with the
existing llama.cpp RPC backend. It is the smallest useful step toward the
pipeline-parallel cluster described in `plan.md`.

This prototype is not the U-shaped hidden-state design yet. It does not split
the llama.cpp graph at custom layer boundaries, and it does not provide a
privacy guarantee for intermediate activations. Its job is to answer one
question first: can the available network and machines run a remote layer split
well enough to justify deeper runtime changes?

## Safety

The RPC backend is proof-of-concept code. Do not expose `ggml-rpc-server` on an
open network. Use a trusted LAN, VPN, or SSH tunnel, and firewall the port.

## Topology

```text
host llama-cli
  |
  | --distributed node-a:50052,node-b:50052
  v
remote ggml-rpc-server processes
  |
  v
remote CPU/GPU devices
```

The host process still owns the normal llama.cpp inference flow. Remote nodes
expose ggml devices. `--distributed` is a prototype-friendly alias for the
existing llama.cpp RPC device path. llama.cpp then places model weights and KV
cache across local and remote devices, normally in proportion to available
memory unless you pass `--tensor-split`.

## Build

Build on every machine that participates:

```sh
./prototypes/distributed-rpc/build-rpc.sh
```

Pass accelerator flags as CMake arguments. For example, on CUDA machines:

```sh
./prototypes/distributed-rpc/build-rpc.sh -DGGML_CUDA=ON
```

The default build directory is `build-rpc`. Override it with
`LLAMA_BUILD_DIR=/path/to/build`.

By default the script builds `ggml-rpc-server`, `llama-cli`, and
`llama-distributed-tap`. Override with
`TARGETS="ggml-rpc-server llama-cli llama-server"` if you want more.

## Start A Remote Node

On each worker:

```sh
HOST=0.0.0.0 PORT=50052 ./prototypes/distributed-rpc/remote-node.sh
```

Optional settings:

```sh
DEVICE=CUDA0                 # expose only one device
THREADS=8                    # CPU backend threads
CACHE=1                      # enable RPC tensor cache, default on
LLAMA_BUILD_DIR=build-rpc    # build directory
LLAMA_CACHE=/fast/cache      # optional; defaults under build-rpc/cache
```

Use `HOST=127.0.0.1` when testing through SSH tunnels.

## Run From The Host

With a local GGUF:

```sh
RPC_SERVERS=192.168.1.20:50052,192.168.1.21:50052 \
MODEL=/models/model.gguf \
./prototypes/distributed-rpc/host-run.sh
```

With Hugging Face download support:

```sh
RPC_SERVERS=192.168.1.20:50052 \
HF_REPO=ggml-org/gemma-3-1b-it-GGUF:Q4_K_M \
./prototypes/distributed-rpc/host-run.sh
```

Useful knobs:

```sh
N_GPU_LAYERS=all       # default
TENSOR_SPLIT=1,1       # split proportions across visible RPC/local devices
CTX_SIZE=4096
PROMPT="Write one paragraph about distributed inference."
```

The wrapper expands `RPC_SERVERS` to `llama-cli --distributed ...`. You can use
that option directly too:

```sh
build-rpc/bin/llama-cli \
  --distributed 192.168.1.20:50052 \
  --split-mode layer \
  --n-gpu-layers all \
  --model /models/model.gguf \
  "Hello from distributed llama.cpp."
```

## C-Side GGML Tap

`llama-distributed-tap` replaces the host bash wrapper with a small C program.
It registers RPC servers before model load, installs a ggml scheduler callback,
then runs a normal `llama_decode()` loop.

```sh
build-rpc/bin/llama-distributed-tap \
  --distributed 192.168.1.20:50052 \
  --model /models/model.gguf \
  --prompt "Hello from distributed llama.cpp."
```

The callback is a tap, not a tensor replacement layer. It can observe named
ggml tensors during execution and force scheduler synchronization around matching
nodes. It cannot yet inject remote activations back into the graph. By default it
prints tensors whose names contain `l_out`; use `--trace all` for every node or
`--trace none` to only test programmatic RPC registration.

## What To Look For

In the host logs, confirm:

- RPC support is enabled.
- Remote devices are registered.
- Model layers are offloaded.
- RPC cache helps subsequent loads.

For a first performance pass, compare:

```sh
# Local only
MODEL=/models/model.gguf RPC_SERVERS= ./prototypes/distributed-rpc/host-run.sh

# Remote split
MODEL=/models/model.gguf RPC_SERVERS=192.168.1.20:50052 ./prototypes/distributed-rpc/host-run.sh
```

Track prompt processing tokens/sec, generation tokens/sec, and wall-clock time.

## Why This Comes Before U-Shaped Splitting

The U-shaped design needs new runtime surfaces:

- execute only a prefix or suffix of model layers;
- export an activation tensor at a layer boundary;
- import an activation tensor as the next layer input;
- keep per-request KV cache state on the node that owns each layer;
- schedule many in-flight requests without corrupting sequence state.

Those are core inference changes. This RPC prototype tests the less invasive
foundation first: remote layer placement, remote KV storage, loading behavior,
and network cost.
