// Standalone CLI wrapper around the shared discovery logic in tailscale_discovery.{h,cpp}. Kept
// as a lower-level tool for scripting/debugging (RPC_SERVERS=$(...), quick terminal checks) -
// llama-cluster-ui calls the same underlying function in-process rather than shelling out to
// this binary; this file is just arg parsing + output formatting on top of it.

#include "tailscale_discovery.h"

#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using json = nlohmann::json;

struct discover_options {
    int  port;
    int  connect_timeout_ms;
    bool json_output;
};

static void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s [--port PORT] [--connect-timeout-ms MS] [--json]\n"
        "\n"
        "Prints a comma-separated host:port list of Tailscale peers confirmed to be running\n"
        "ggml-rpc-server, suitable for RPC_SERVERS=$(...) or --distributed \"$(...)\".\n"
        "With --json, prints a structured JSON array instead (for programmatic consumers).\n"
        "\n"
        "options:\n"
        "  --port PORT               port to probe on each peer, default %s\n"
        "  --connect-timeout-ms MS   TCP connect timeout per peer, default %s\n"
        "  --json                    print a JSON array on stdout instead of a plain host:port list\n"
        "  -h, --help                show this help\n",
        prog, "50052 (TAILSCALE_RPC_PORT)", "500 (TAILSCALE_CONNECT_TIMEOUT_MS)");
}

static bool parse_args(int argc, char ** argv, struct discover_options * opts) {
    const char * env_port    = getenv("TAILSCALE_RPC_PORT");
    const char * env_timeout = getenv("TAILSCALE_CONNECT_TIMEOUT_MS");

    opts->port               = env_port    ? atoi(env_port)    : 50052;
    opts->connect_timeout_ms = env_timeout ? atoi(env_timeout) : 500;
    opts->json_output        = false;

    for (int i = 1; i < argc; ++i) {
        const char * arg   = argv[i];
        const char * value = (i + 1 < argc) ? argv[i + 1] : NULL;

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--port") == 0) {
            if (!value) return false;
            opts->port = atoi(value);
            i++;
        } else if (strcmp(arg, "--connect-timeout-ms") == 0) {
            if (!value) return false;
            opts->connect_timeout_ms = atoi(value);
            i++;
        } else if (strcmp(arg, "--json") == 0) {
            opts->json_output = true;
        } else {
            fprintf(stderr, "[tailscale-discover] error: unknown argument: %s\n", arg);
            return false;
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    struct discover_options opts;
    if (!parse_args(argc, argv, &opts)) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    std::vector<discovered_node> nodes;
    std::string                  error;
    if (!tailscale_discover_nodes(opts.port, opts.connect_timeout_ms, nodes, error)) {
        fprintf(stderr, "[tailscale-discover] error: %s\n", error.c_str());
        return 1;
    }

    if (opts.json_output) {
        json arr = json::array();
        for (const auto & node : nodes) {
            arr.push_back({
                {"hostname",    node.hostname},
                {"ip",          node.ip},
                {"port",        node.port},
                {"endpoint",    node.endpoint},
                {"free_bytes",  node.free_bytes},
                {"total_bytes", node.total_bytes},
            });
        }
        printf("%s\n", arr.dump().c_str());
    } else {
        std::string joined;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (i > 0) {
                joined += ",";
            }
            joined += nodes[i].endpoint;
        }
        printf("%s\n", joined.c_str());
    }

    for (const auto & node : nodes) {
        fprintf(stderr, "[tailscale-discover]   %-20s %-16s free=%.1f MiB total=%.1f MiB - OK\n",
                node.hostname.c_str(), node.ip.c_str(),
                node.free_bytes / 1024.0 / 1024.0, node.total_bytes / 1024.0 / 1024.0);
    }
    fprintf(stderr, "[tailscale-discover] %zu node(s) confirmed as ggml-rpc-server nodes\n", nodes.size());

    return 0;
}
