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

By default the script builds `ggml-rpc-server`, `llama-cli`, `llama-app` (the
`llama` binary - `download`/`serve`/`cli` subcommands), `llama-distributed-tap`,
`llama-split-tap`, `llama-tailscale-discover`, and `llama-cluster-ui`. Override
with e.g. `TARGETS="ggml-rpc-server llama-cli llama-server"` if you want fewer.

## Start A Node

The simplest way to join the cluster: build once, then run **`llama-cluster-ui`**
on every machine, including workers. By default it always plays both roles at
once - it exposes this machine's own compute to the pool (an embedded RPC
server, no separate process) *and* serves the browser control panel you can
use to discover peers, pick a model, and launch a run:

```sh
build-rpc/bin/llama-cluster-ui
```

That's the whole setup for a worker that only ever contributes compute and
never drives a run itself. Env vars tune the embedded server half:

```sh
RPC_SERVE_HOST=0.0.0.0        # default - must be reachable by other peers
RPC_SERVE_PORT=50052          # default, matches TAILSCALE_RPC_PORT's default
RPC_SERVE_DEVICE=CUDA0        # optional: expose only one device
RPC_SERVE_THREADS=8           # CPU backend threads
RPC_SERVE_CACHE=1             # enable RPC tensor cache, default on
RPC_SERVE_DISABLE=1           # opt this machine out of contributing compute entirely
```

See "Web Control Panel" below for the browser UI's own env vars
(`CLUSTER_UI_HOST`, `CLUSTER_UI_PORT`, `MODELS_DIR`, ...).

### Lower-level alternative: `ggml-rpc-server` directly

If you'd rather run just the bare RPC server with no UI/download/discovery
attached (e.g. a resource-constrained worker, or scripting/debugging), the
standalone binary is still available:

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

There is no shell wrapper for this - `llama-cli` supports `--distributed`/`--rpc` natively,
including Tailscale auto-discovery (`auto`), so invoke it directly:

```sh
build-rpc/bin/llama-cli \
  --distributed 192.168.1.20:50052,192.168.1.21:50052 \
  --split-mode layer \
  --n-gpu-layers all \
  --model /models/model.gguf \
  --prompt "Hello from distributed llama.cpp."
```

Or let it discover nodes on your Tailscale tailnet itself instead of listing them by hand
(requires nodes already running `remote-node.sh`, and `tailscale` in `PATH`):

```sh
build-rpc/bin/llama-cli \
  --distributed auto \
  --split-mode layer \
  --n-gpu-layers all \
  --model /models/model.gguf \
  --prompt "Hello from distributed llama.cpp."
```

`TAILSCALE_RPC_PORT` (default 50052) and `TAILSCALE_CONNECT_TIMEOUT_MS` (default 500) tune the
discovery probe. See `--help` for `--tensor-split`, `--ctx-size`, and other flags.

For a browser-based control panel instead (discover nodes, pick a model, set node order, launch,
watch streamed output), see "Web Control Panel" below.

## Web Control Panel

`llama-cluster-ui` is the single binary for the whole cluster workflow - see "Start A Node"
above for how it plays both the server and client/host role at once. Run it and open
`http://127.0.0.1:8787` (default) in a browser:

```sh
build-rpc/bin/llama-cluster-ui
```

From there you can:

- **Discover nodes** - scans the Tailscale tailnet for peers confirmed to be running a
  ggml-rpc-server (their own `llama-cluster-ui`'s embedded server, or a standalone
  `ggml-rpc-server`/`remote-node.sh` - both look identical to discovery), and lets you reorder
  them (pipeline/layer order).
- **Pick or download a model** - lists local `.gguf` files under `MODELS_DIR` (scanned
  recursively, so a downloaded model's nested Hugging Face cache layout is found too - see
  below), or download one directly from Hugging Face (`llama download -hf <repo>[:tag]` under
  the hood) with a live progress panel.
- **Launch** - runs `llama-cli --distributed <endpoints> --split-mode layer` with the chosen
  model/prompt, and streams its output back to the page. **Stop** terminates the run.

Env vars (all optional, sane defaults):

```sh
CLUSTER_UI_HOST=127.0.0.1     # UI bind address - loopback by default, unlike RPC_SERVE_HOST
CLUSTER_UI_PORT=8787
MODELS_DIR=prototypes/distributed-rpc/testdata
CLUSTER_DOWNLOAD_DIR=$MODELS_DIR   # default - see below
CLUSTER_UI_STATIC_DIR=prototypes/distributed-rpc/ui
TAILSCALE_RPC_PORT=50052
TAILSCALE_CONNECT_TIMEOUT_MS=500
```

`CLUSTER_DOWNLOAD_DIR` is per-node, local config only - each machine decides for itself where
its own downloads land (there is no field on the cluster gRPC protocol's `DownloadModel` RPC to
override this remotely). It defaults to `MODELS_DIR`'s value, so anything downloaded - whether
via the HTTP download panel or a remote `DownloadModel` gRPC call from another node - immediately
shows up in this same node's own model list without extra config. It's passed as an `LLAMA_CACHE`
override to the `llama download` subprocess's own environment only (never this process's global
environment), so it doesn't affect any other tool's caching. Point it elsewhere if you'd rather
keep downloaded models separate from `MODELS_DIR`'s manually-placed ones.

The UI's own HTTP server defaults to loopback-only (`127.0.0.1`) since it can launch arbitrary
`llama-cli` invocations with no authentication - unlike the embedded RPC server half
(`RPC_SERVE_HOST`, see "Start A Node"), which needs to be reachable by other machines to be
useful and so defaults open. Setting `CLUSTER_UI_HOST` to a non-loopback address prints a loud
warning; only do that over a trusted tailnet/VPN plus your own access control, never on an open
network.

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
build-rpc/bin/llama-cli --model /models/model.gguf --prompt "..."

# Remote split
build-rpc/bin/llama-cli --distributed 192.168.1.20:50052 --split-mode layer \
  --n-gpu-layers all --model /models/model.gguf --prompt "..."
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
