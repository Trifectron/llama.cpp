#include "tailscale_discovery.h"

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <nlohmann/json.hpp>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <sys/select.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

#include <cstdio>
#include <cstring>
#include <mutex>

#ifdef _WIN32
typedef SOCKET sockfd_t;
#else
typedef int sockfd_t;
#endif

using json = nlohmann::json;

namespace {

struct tailscale_peer {
    std::string hostname;
    std::string ip;
};

// Runs `tailscale status --json` and returns the online peers (hostname + first TailscaleIP).
// Returns false if the command can't be run at all (tailscale not installed / daemon not
// running) - that's a hard error. An empty peer list (no peers on the tailnet yet) is not an
// error, just an empty result.
bool list_online_tailscale_peers(std::vector<tailscale_peer> & out_peers, std::string & out_error) {
#ifdef _WIN32
    FILE * pipe = _popen("tailscale status --json 2>nul", "r");
#else
    FILE * pipe = popen("tailscale status --json 2>/dev/null", "r");
#endif
    if (pipe == NULL) {
        out_error = "failed to run 'tailscale status --json' (is tailscale installed and in PATH?)";
        return false;
    }

    std::string output;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        output.append(buf, n);
    }
#ifdef _WIN32
    const int rc = _pclose(pipe);
#else
    const int rc = pclose(pipe);
#endif
    if (rc != 0 || output.empty()) {
        out_error = "'tailscale status --json' failed (is the tailscaled daemon running and are you logged in?)";
        return false;
    }

    json j;
    try {
        j = json::parse(output);
    } catch (const json::exception & e) {
        out_error = std::string("failed to parse tailscale status JSON: ") + e.what();
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
bool tcp_port_open(const std::string & ip, int port, int timeout_ms) {
#ifdef _WIN32
    // one-time winsock init; refcounted by the OS, so it coexists with ggml-rpc's own WSAStartup
    static std::once_flag wsa_once;
    std::call_once(wsa_once, []() {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
    });

    sockfd_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == INVALID_SOCKET) {
        return false;
    }

    u_long nonblock = 1;
    ioctlsocket(fd, FIONBIO, &nonblock);
#else
    sockfd_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }

    const int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) port);
    if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) {
#ifdef _WIN32
        closesocket(fd);
#else
        close(fd);
#endif
        return false;
    }

    bool ok = false;
    const int rc = connect(fd, (struct sockaddr *) &addr, sizeof(addr));
    if (rc == 0) {
        ok = true;
#ifdef _WIN32
    } else if (WSAGetLastError() == WSAEWOULDBLOCK) {
#else
    } else if (errno == EINPROGRESS) {
#endif
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(fd, &wfds);

        struct timeval tv;
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;

        // first select() arg is ignored on Windows
        if (select((int) fd + 1, NULL, &wfds, NULL, &tv) > 0) {
            int err = 0;
#ifdef _WIN32
            int err_len = sizeof(err);
#else
            socklen_t err_len = sizeof(err);
#endif
            if (getsockopt(fd, SOL_SOCKET, SO_ERROR, (char *) &err, &err_len) == 0 && err == 0) {
                ok = true;
            }
        }
    }

#ifdef _WIN32
    closesocket(fd);
#else
    close(fd);
#endif
    return ok;
}

} // namespace

bool tailscale_discover_nodes(
        int                              port,
        int                              timeout_ms,
        std::vector<discovered_node> &   out_nodes,
        std::string &                    out_error) {
    std::vector<tailscale_peer> peers;
    if (!list_online_tailscale_peers(peers, out_error)) {
        return false;
    }

    for (const auto & peer : peers) {
        if (!tcp_port_open(peer.ip, port, timeout_ms)) {
            fprintf(stderr, "[tailscale-discovery]   %-20s %-16s port %d closed/unreachable - skipped\n",
                    peer.hostname.c_str(), peer.ip.c_str(), port);
            continue;
        }

        const std::string endpoint = peer.ip + ":" + std::to_string(port);

        // NOTE: unlike tcp_port_open() above, this handshake has no timeout of its own - if
        // something other than ggml-rpc-server is listening on this port and never responds or
        // disconnects, this call can block indefinitely. Acceptable for a prototype scanning a
        // small, trusted tailnet; a production version would want a bounded read timeout here.
        size_t free_mem  = 0;
        size_t total_mem = 0;
        ggml_backend_rpc_get_device_memory(endpoint.c_str(), 0, &free_mem, &total_mem);

        if (total_mem == 0) {
            fprintf(stderr, "[tailscale-discovery]   %-20s %-16s port open but not a ggml-rpc-server - skipped\n",
                    peer.hostname.c_str(), peer.ip.c_str());
            continue;
        }

        out_nodes.push_back({peer.hostname, peer.ip, port, endpoint, free_mem, total_mem});
    }

    return true;
}
