#pragma once

// Shared Tailscale-based node discovery logic for the distributed-rpc prototype. Used by both
// llama-tailscale-discover (a thin standalone CLI wrapper around this) and llama-cluster-ui
// (calls this in-process, no subprocess involved) - one implementation, two entry points.

#include <cstddef>
#include <string>
#include <vector>

struct discovered_node {
    std::string hostname;
    std::string ip;
    int         port;
    std::string endpoint;    // "ip:port"
    size_t      free_bytes;
    size_t      total_bytes;
};

// Discovers Tailscale peers confirmed to be running a real ggml-rpc-server on `port`:
//   1. Asks the local Tailscale daemon (`tailscale status --json`) which peers are online.
//   2. TCP-probes each one on `port` with `timeout_ms` as a fast pre-filter.
//   3. Confirms survivors with a real RPC_CMD_HELLO handshake via
//      ggml_backend_rpc_get_device_memory() (also yields free/total VRAM).
//
// Requires the caller to have already called ggml_backend_load_all() (so the caller controls
// when that one-time backend-scanning cost happens, rather than paying it on every call).
//
// Returns false only on a hard error (tailscale not installed / daemon not running / bad JSON),
// with a message in out_error. An empty out_nodes with a true return means "no peers found" -
// that's not an error.
bool tailscale_discover_nodes(
        int                              port,
        int                              timeout_ms,
        std::vector<discovered_node> &   out_nodes,
        std::string &                    out_error);
