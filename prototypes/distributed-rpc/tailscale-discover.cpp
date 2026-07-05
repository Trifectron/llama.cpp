// Tailscale-based node discovery for the distributed-rpc prototype (see plan.md, "Additional
// phase: Tailscale-based node discovery"). Finds candidate GPU-cluster nodes by asking the
// locally running Tailscale daemon which peers are online, then confirms each candidate is
// actually running a ggml-rpc-server before including it - it does not touch the RPC wire
// protocol at all, it just resolves a `--distributed`/`RPC_SERVERS`-style host:port list the
// same way a user would type it by hand.
//
// Flow:
//   1. popen("tailscale status --json") and parse the peer list with the already-vendored
//      nlohmann::json (no new dependency).
//   2. For each peer marked Online, try a short raw TCP connect to a candidate port as a fast
//      pre-filter, so one unreachable peer can't stall the whole scan.
//   3. For peers that accept the connection, call ggml_backend_rpc_get_device_memory() - this
//      already performs the RPC_CMD_HELLO handshake as a side effect, so a successful call means
//      "this is really a ggml-rpc-server", not just some other service on that port. Free/total
//      VRAM comes along for the ride.
//   4. Print a comma-separated host:port list on stdout (the exact format --distributed and
//      RPC_SERVERS already accept) and a human-readable summary on stderr.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using json = nlohmann::json;

struct discover_options {
    int         port;
    int         connect_timeout_ms;
    bool        json_output;
};

static void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s [--port PORT] [--connect-timeout-ms MS] [--json]\n"
        "\n"
        "Prints a comma-separated host:port list of Tailscale peers confirmed to be running\n"
        "ggml-rpc-server, suitable for RPC_SERVERS=$(...) or --distributed \"$(...)\".\n"
        "With --json, prints a structured JSON array instead (for programmatic consumers, e.g.\n"
        "llama-cluster-ui's /api/discover).\n"
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

struct tailscale_peer {
    std::string hostname;
    std::string ip;
};

// Runs `tailscale status --json` and returns the online peers (hostname + first TailscaleIP).
// Returns false if the command can't be run at all (tailscale not installed / daemon not
// running) - that's a hard error. An empty peer list (no peers on the tailnet yet) is not an
// error, just an empty result.
static bool list_online_tailscale_peers(std::vector<tailscale_peer> & out_peers) {
    FILE * pipe = popen("tailscale status --json 2>/dev/null", "r");
    if (pipe == NULL) {
        fprintf(stderr, "[tailscale-discover] error: failed to run 'tailscale status --json' "
                        "(is tailscale installed and in PATH?)\n");
        return false;
    }

    std::string output;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        output.append(buf, n);
    }
    const int rc = pclose(pipe);
    if (rc != 0 || output.empty()) {
        fprintf(stderr, "[tailscale-discover] error: 'tailscale status --json' failed "
                        "(is the tailscaled daemon running and are you logged in?)\n");
        return false;
    }

    json j;
    try {
        j = json::parse(output);
    } catch (const json::exception & e) {
        fprintf(stderr, "[tailscale-discover] error: failed to parse tailscale status JSON: %s\n", e.what());
        return false;
    }

    if (!j.contains("Peer") || !j["Peer"].is_object()) {
        // no peers on this tailnet yet - not an error.
        return true;
    }

    for (const auto & [node_id, peer] : j["Peer"].items()) {
        (void) node_id;

        if (!peer.value("Online", false)) {
            continue;
        }
        if (!peer.contains("TailscaleIPs") || !peer["TailscaleIPs"].is_array() || peer["TailscaleIPs"].empty()) {
            continue;
        }

        tailscale_peer p;
        p.hostname = peer.value("HostName", "(unknown)");
        p.ip       = peer["TailscaleIPs"][0].get<std::string>();
        out_peers.push_back(p);
    }

    return true;
}

// Fast pre-filter: is anything listening on ip:port at all? Avoids paying the RPC handshake
// (and its own internal timeout behavior) for peers that are online in Tailscale but simply
// don't have a ggml-rpc-server running.
static bool tcp_port_open(const std::string & ip, int port, int timeout_ms) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }

    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) port);
    if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) {
        close(fd);
        return false;
    }

    bool ok = false;
    int  rc = connect(fd, (struct sockaddr *) &addr, sizeof(addr));
    if (rc == 0) {
        ok = true;
    } else if (errno == EINPROGRESS) {
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(fd, &wfds);

        struct timeval tv;
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;

        if (select(fd + 1, NULL, &wfds, NULL, &tv) > 0) {
            int       err     = 0;
            socklen_t err_len = sizeof(err);
            if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &err_len) == 0 && err == 0) {
                ok = true;
            }
        }
    }

    close(fd);
    return ok;
}

int main(int argc, char ** argv) {
    struct discover_options opts;
    if (!parse_args(argc, argv, &opts)) {
        usage(argv[0]);
        return 1;
    }

    std::vector<tailscale_peer> peers;
    if (!list_online_tailscale_peers(peers)) {
        return 1;
    }

    fprintf(stderr, "[tailscale-discover] %zu online tailnet peer(s), probing port %d\n",
            peers.size(), opts.port);

    ggml_backend_load_all();

    struct confirmed_node {
        std::string hostname;
        std::string ip;
        int         port;
        size_t      free_bytes;
        size_t      total_bytes;
    };

    std::vector<confirmed_node> confirmed;

    for (const auto & peer : peers) {
        if (!tcp_port_open(peer.ip, opts.port, opts.connect_timeout_ms)) {
            fprintf(stderr, "[tailscale-discover]   %-20s %-16s port %d closed/unreachable - skipped\n",
                    peer.hostname.c_str(), peer.ip.c_str(), opts.port);
            continue;
        }

        const std::string endpoint = peer.ip + ":" + std::to_string(opts.port);

        // NOTE: unlike tcp_port_open() above, this handshake has no timeout of its own - if
        // something other than ggml-rpc-server is listening on this port and never responds or
        // disconnects, this call can block indefinitely. Acceptable for a prototype scanning a
        // small, trusted tailnet; a production version would want a bounded read timeout here.
        size_t free_mem  = 0;
        size_t total_mem = 0;
        ggml_backend_rpc_get_device_memory(endpoint.c_str(), 0, &free_mem, &total_mem);

        if (total_mem == 0) {
            fprintf(stderr, "[tailscale-discover]   %-20s %-16s port open but not a ggml-rpc-server - skipped\n",
                    peer.hostname.c_str(), peer.ip.c_str());
            continue;
        }

        fprintf(stderr, "[tailscale-discover]   %-20s %-16s free=%.1f MiB total=%.1f MiB - OK\n",
                peer.hostname.c_str(), peer.ip.c_str(),
                free_mem / 1024.0 / 1024.0, total_mem / 1024.0 / 1024.0);

        confirmed.push_back({peer.hostname, peer.ip, opts.port, free_mem, total_mem});
    }

    if (opts.json_output) {
        json arr = json::array();
        for (const auto & node : confirmed) {
            arr.push_back({
                {"hostname",    node.hostname},
                {"ip",          node.ip},
                {"port",        node.port},
                {"endpoint",    node.ip + ":" + std::to_string(node.port)},
                {"free_bytes",  node.free_bytes},
                {"total_bytes", node.total_bytes},
            });
        }
        printf("%s\n", arr.dump().c_str());
    } else {
        std::string joined;
        for (size_t i = 0; i < confirmed.size(); ++i) {
            if (i > 0) {
                joined += ",";
            }
            joined += confirmed[i].ip + ":" + std::to_string(confirmed[i].port);
        }
        printf("%s\n", joined.c_str());
    }

    fprintf(stderr, "[tailscale-discover] %zu of %zu online peer(s) confirmed as ggml-rpc-server nodes\n",
            confirmed.size(), peers.size());

    return 0;
}
