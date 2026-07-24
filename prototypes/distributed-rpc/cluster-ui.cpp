// Live control-panel backend for the distributed-rpc prototype (see plan.md and the plan file
// history: "Live Cluster Control-Panel UI"). This is the single binary a friend on the cluster
// ever needs to run: by default it always plays BOTH roles at once -
//   - server: exposes this machine's own compute to the pool via an embedded ggml-rpc-server
//     (same underlying ggml_backend_rpc_start_server() call tools/rpc/rpc-server.cpp uses, just
//     run on a background thread here instead of its own process - no separate binary needed).
//   - client/host: serves a browser UI that discovers Tailscale peers (in-process, via the
//     shared tailscale_discovery module), lists/downloads GGUF models, and launches llama-cli
//     directly (fork()+execve(), no shell) to actually run inference.
// RPC_SERVE_DISABLE=1 opts a machine out of the server half if you only want it to drive/watch.
//
// This binary shells out to two others purely as implementation details, never as something a
// user runs by hand: `llama-cli` (inference) and `llama download` (the app/download.cpp
// subcommand, for pulling models from Hugging Face).

#include "tailscale_discovery.h"
#include "cluster_layer_runner.h"
#include "cluster_node_service.h"
#include "coordinator_service.h"
#include "subprocess_runner.h"

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>
#include <grpcpp/grpcpp.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/wait.h>
#  include <fcntl.h>
#  include <signal.h>
#  include <unistd.h>
#endif

#include <sys/stat.h>
#include <dirent.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

#ifndef _WIN32
extern char ** environ;
#endif

// ---------------------------------------------------------------------------------------------
// GGUF header reader - C++ port of the reference teaching artifact's read_gguf_meta(): walks
// just the KV metadata section far enough to learn the architecture name and layer count, then
// stops before the (potentially huge) tokenizer arrays and tensor data. Mirrors the Python
// version's actual semantics: store scalar/string values generically as they're seen, skip
// array contents without storing them, and stop as soon as both `general.architecture` and
// `<arch>.block_count` are known (regardless of which order they appeared in).
// ---------------------------------------------------------------------------------------------

struct gguf_meta {
    std::string arch;
    uint32_t    n_layers   = 0;
    uint64_t    size_bytes = 0;
};

static bool gguf_read_string(FILE * f, std::string & out) {
    uint64_t len = 0;
    if (fread(&len, sizeof(len), 1, f) != 1) {
        return false;
    }
    out.resize(len);
    if (len > 0 && fread(&out[0], 1, len, f) != len) {
        return false;
    }
    return true;
}

// Reads (or skips) one GGUF value of the given type. If it's a scalar integer type, stores the
// (possibly widened) value in *out_u64 when non-null. If it's a string, stores it in *out_str
// when non-null. Arrays are walked recursively but never stored - the whole point of this
// reader is to avoid materializing the multi-thousand-entry tokenizer arrays.
static bool gguf_read_value(FILE * f, uint32_t vtype, uint64_t * out_u64, std::string * out_str) {
    switch (vtype) {
        case 0: { uint8_t  v = 0; if (fread(&v, 1, 1, f) != 1) return false; if (out_u64) *out_u64 = v; return true; }
        case 1: { int8_t   v = 0; if (fread(&v, 1, 1, f) != 1) return false; if (out_u64) *out_u64 = (uint64_t) (int64_t) v; return true; }
        case 2: { uint16_t v = 0; if (fread(&v, 2, 1, f) != 1) return false; if (out_u64) *out_u64 = v; return true; }
        case 3: { int16_t  v = 0; if (fread(&v, 2, 1, f) != 1) return false; if (out_u64) *out_u64 = (uint64_t) (int64_t) v; return true; }
        case 4: { uint32_t v = 0; if (fread(&v, 4, 1, f) != 1) return false; if (out_u64) *out_u64 = v; return true; }
        case 5: { int32_t  v = 0; if (fread(&v, 4, 1, f) != 1) return false; if (out_u64) *out_u64 = (uint64_t) (int64_t) v; return true; }
        case 6: { float    v = 0; if (fread(&v, 4, 1, f) != 1) return false; return true; }
        case 7: { uint8_t  v = 0; if (fread(&v, 1, 1, f) != 1) return false; return true; } // bool
        case 8: {
            std::string s;
            if (!gguf_read_string(f, s)) return false;
            if (out_str) *out_str = s;
            return true;
        }
        case 9: { // array: elem type, count, elems (recursively skipped, never stored)
            uint32_t etype = 0;
            uint64_t n     = 0;
            if (fread(&etype, 4, 1, f) != 1) return false;
            if (fread(&n,     8, 1, f) != 1) return false;
            for (uint64_t i = 0; i < n; ++i) {
                if (!gguf_read_value(f, etype, nullptr, nullptr)) return false;
            }
            return true;
        }
        case 10: { uint64_t v = 0; if (fread(&v, 8, 1, f) != 1) return false; if (out_u64) *out_u64 = v; return true; }
        case 11: { int64_t  v = 0; if (fread(&v, 8, 1, f) != 1) return false; if (out_u64) *out_u64 = (uint64_t) v; return true; }
        case 12: { double   v = 0; if (fread(&v, 8, 1, f) != 1) return false; return true; }
        default: return false; // unknown value type
    }
}

static bool read_gguf_meta(const std::string & path, gguf_meta & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }

    bool ok = true;
    char magic[4];
    ok = ok && fread(magic, 1, 4, f) == 4 && memcmp(magic, "GGUF", 4) == 0;

    uint32_t version      = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count     = 0;
    ok = ok && fread(&version,      4, 1, f) == 1;
    ok = ok && fread(&tensor_count, 8, 1, f) == 1;
    ok = ok && fread(&kv_count,     8, 1, f) == 1;

    std::map<std::string, std::string> strs;
    std::map<std::string, uint64_t>    ints;

    for (uint64_t i = 0; ok && i < kv_count; ++i) {
        std::string key;
        ok = ok && gguf_read_string(f, key);
        uint32_t vtype = 0;
        ok = ok && fread(&vtype, 4, 1, f) == 1;
        if (!ok) {
            break;
        }

        std::string sval;
        uint64_t    ival = 0;
        if (!gguf_read_value(f, vtype, &ival, &sval)) {
            ok = false;
            break;
        }

        if (vtype == 8) {
            strs[key] = sval;
        } else if (vtype <= 5 || vtype == 10 || vtype == 11) {
            ints[key] = ival;
        }

        auto arch_it = strs.find("general.architecture");
        if (arch_it != strs.end()) {
            auto layers_it = ints.find(arch_it->second + ".block_count");
            if (layers_it != ints.end()) {
                out.arch     = arch_it->second;
                out.n_layers = (uint32_t) layers_it->second;
                fclose(f);
                struct stat st;
                out.size_bytes = (stat(path.c_str(), &st) == 0) ? (uint64_t) st.st_size : 0;
                return true;
            }
        }
    }

    fclose(f);
    return false;
}

// Recursively scans a directory tree for *.gguf files, depth-bounded (defensive against
// pathological nesting/symlink cycles rather than genuinely expected). A flat, single-level scan
// alone would never find a model placed by CLUSTER_DOWNLOAD_DIR/DownloadModel: Hugging Face's
// own cache layout nests several directories deep
// (models--org--repo/snapshots/<sha>/file.gguf, itself a symlink to .../blobs/<oid> - stat()
// follows the symlink transparently, so it's reported as a regular file here).
static void scan_gguf_files_recursive(const std::string & dir_path, int depth_remaining, std::vector<std::string> & out_paths) {
    if (depth_remaining <= 0) {
        return;
    }

    DIR * dir = opendir(dir_path.c_str());
    if (!dir) {
        return;
    }

    struct dirent * entry;
    while ((entry = readdir(dir)) != nullptr) {
        const std::string name(entry->d_name);
        if (name == "." || name == "..") {
            continue;
        }
        const std::string path = dir_path + "/" + name;

        struct stat st;
        if (stat(path.c_str(), &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            scan_gguf_files_recursive(path, depth_remaining - 1, out_paths);
        } else if (S_ISREG(st.st_mode) && name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) {
            out_paths.push_back(path);
        }
    }
    closedir(dir);
}

// ---------------------------------------------------------------------------------------------
// Subprocess launching (run_state, launch_process(), sibling_binary_path()) now lives in
// subprocess_runner.h/.cpp - shared with cluster_node_service.cpp's DownloadModel gRPC handler,
// which launches the same `llama download` subprocess this file's /api/models/download does.
// ---------------------------------------------------------------------------------------------

static std::mutex                                    g_runs_mtx;
static std::map<std::string, std::shared_ptr<run_state>> g_runs;
static int                                            g_next_run_id = 1;

// Origin's own head runner (cluster_layer_runner bounded to [0, head_end)), keyed by request_id -
// kept alive across multiple /api/launch-grpc round-trips for the same request instead of being
// destroyed when the HTTP handler returns, so its KV cache carries over between generated tokens
// (see PROGRESS.md step 3). Mirrors g_cluster_node's own request_id-keyed session map, just for
// the head segment, which never goes through the gRPC service.
static std::mutex                                                   g_head_runners_mtx;
static std::map<std::string, std::unique_ptr<cluster_layer_runner>> g_head_runners;

// ---------------------------------------------------------------------------------------------
// Embedded RPC server (the "server" half of this binary's dual role) - a direct port of
// tools/rpc/rpc-server.cpp's device-selection logic, calling the exact same underlying
// ggml_backend_rpc_start_server() that binary uses. That function blocks forever serving
// connections, so it runs on its own detached thread here rather than in main()'s thread, which
// needs to go on to run the httplib server.
// ---------------------------------------------------------------------------------------------

// Mirrors tools/rpc/rpc-server.cpp's get_devices(): explicit device names if given, else all
// non-CPU devices, falling back to the CPU device only if nothing else is available.
static std::vector<ggml_backend_dev_t> rpc_serve_select_devices(const std::string & device_list_csv) {
    std::vector<ggml_backend_dev_t> devices;

    if (!device_list_csv.empty()) {
        const std::regex regex{R"([,/]+)"};
        std::sregex_token_iterator iter(device_list_csv.begin(), device_list_csv.end(), regex, -1);
        std::sregex_token_iterator end;
        for (; iter != end; ++iter) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(iter->str().c_str());
            if (dev) {
                devices.push_back(dev);
            } else {
                fprintf(stderr, "[cluster-ui] error: unknown RPC_SERVE_DEVICE entry: %s\n", iter->str().c_str());
                return {};
            }
        }
        return devices;
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            devices.push_back(dev);
        }
    }
    if (devices.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (dev) {
            devices.push_back(dev);
        }
    }

    return devices;
}

// Starts the embedded RPC server on a detached background thread. Returns false (logs and does
// not start a thread) on a setup error; once the thread is running, failures inside
// ggml_backend_rpc_start_server() itself are unrecoverable for the lifetime of the process, same
// as the standalone ggml-rpc-server binary.
static bool start_rpc_serve_background() {
    const char * env_disable = getenv("RPC_SERVE_DISABLE");
    if (env_disable && std::string(env_disable) == "1") {
        fprintf(stderr, "[cluster-ui] RPC serve: disabled (RPC_SERVE_DISABLE=1) - this node will not contribute compute\n");
        return true;
    }

    const char * env_host    = getenv("RPC_SERVE_HOST");
    const char * env_port    = getenv("RPC_SERVE_PORT");
    const char * env_threads = getenv("RPC_SERVE_THREADS");
    const char * env_devices = getenv("RPC_SERVE_DEVICE");
    const char * env_cache   = getenv("RPC_SERVE_CACHE");

    // defaults to 0.0.0.0:50052 - unlike CLUSTER_UI_HOST (the browser UI, loopback by default for
    // safety), this half of the binary is only useful if other tailnet peers can actually reach
    // it, so it defaults open rather than closed. Same port default as TAILSCALE_RPC_PORT, so two
    // machines both running this binary with zero configuration can discover each other.
    const std::string host    = env_host    ? env_host    : "0.0.0.0";
    const int         port    = env_port    ? atoi(env_port) : 50052;
    const int         threads = env_threads ? atoi(env_threads) : (int) std::max(1U, std::thread::hardware_concurrency() / 2);
    const bool        use_cache = env_cache ? (std::string(env_cache) != "0") : true;

    if (host != "127.0.0.1" && host != "localhost" && host != "::1") {
        fprintf(stderr,
            "\n"
            "*** NOTE: RPC serve is bound to %s:%d - reachable by other machines on your      ***\n"
            "*** network/tailnet, which is required for them to use this node's compute.      ***\n"
            "*** The RPC protocol has no authentication - only run this on a trusted network  ***\n"
            "*** (e.g. a private Tailscale tailnet), never expose it to the open internet.    ***\n"
            "*** Set RPC_SERVE_DISABLE=1 to opt this machine out of contributing compute.      ***\n"
            "\n",
            host.c_str(), port);
    }

    std::vector<ggml_backend_dev_t> devices = rpc_serve_select_devices(env_devices ? env_devices : "");
    if (devices.empty()) {
        fprintf(stderr, "[cluster-ui] error: RPC serve found no usable devices\n");
        return false;
    }

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (!reg) {
        fprintf(stderr, "[cluster-ui] error: RPC backend not found; rebuild with -DGGML_RPC=ON\n");
        return false;
    }
    auto start_server_fn = (decltype(ggml_backend_rpc_start_server) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server");
    if (!start_server_fn) {
        fprintf(stderr, "[cluster-ui] error: RPC start-server function not found\n");
        return false;
    }

    std::string cache_dir_str;
    if (use_cache) {
        const char * llama_cache_env = getenv("LLAMA_CACHE");
        const char * home_env        = getenv("HOME");
        cache_dir_str = (llama_cache_env ? std::string(llama_cache_env)
                                          : std::string(home_env ? home_env : ".") + "/.cache/llama.cpp") + "/rpc/";

        // best-effort recursive mkdir (no shell involved) - if this doesn't exist,
        // ggml_backend_rpc_start_server can still run without a working cache, not worth
        // failing startup over.
        std::string partial;
        for (size_t pos = 1; pos <= cache_dir_str.size(); ++pos) {
            if (pos == cache_dir_str.size() || cache_dir_str[pos] == '/') {
                partial = cache_dir_str.substr(0, pos);
                if (!partial.empty()) {
                    mkdir(partial.c_str(), 0755);
                }
            }
        }
    }

    fprintf(stderr, "[cluster-ui] RPC serve:      %s:%d (%zu device(s), %d thread(s), cache=%s)\n",
            host.c_str(), port, devices.size(), threads, use_cache ? "on" : "off");

    const std::string endpoint = host + ":" + std::to_string(port);
    // cache_dir_str/devices are captured by value, so they live in the thread's own storage for
    // as long as the (never-returning) server call runs.
    std::thread([start_server_fn, endpoint, cache_dir_str, use_cache, threads, devices]() mutable {
        start_server_fn(endpoint.c_str(), use_cache ? cache_dir_str.c_str() : nullptr,
                         threads, devices.size(), const_cast<ggml_backend_dev_t *>(devices.data()));
    }).detach();

    return true;
}

// ---------------------------------------------------------------------------------------------
// Embedded gRPC server - hosts both the Coordinator and ClusterNode services (see
// proto/cluster.proto) on one port, symmetric with the RPC-serve half above: every
// llama-cluster-ui instance is reachable both as a cluster member (this) and as a plain compute
// device (ggml-rpc, above). grpc++ is a build-time library dependency only, exactly like
// cpp-httplib - there is no separate running process for any of this.
// ---------------------------------------------------------------------------------------------

// Global service instances - the gRPC server holds non-owning pointers to these (standard
// grpc::ServerBuilder::RegisterService() usage), and the /api/launch-grpc handler drives
// cluster_node_service directly (register_local_runner()/wait_for_next_token()) for the
// origin's own head/tail segments, so both need process-lifetime storage.
static coordinator_service   g_coordinator;
static cluster_node_service  g_cluster_node;

static bool start_cluster_grpc_server(const std::string & host, int port) {
    const std::string endpoint = host + ":" + std::to_string(port);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(endpoint, grpc::InsecureServerCredentials());
    builder.RegisterService(&g_coordinator);
    builder.RegisterService(&g_cluster_node);

    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server) {
        fprintf(stderr, "[cluster-ui] error: failed to start cluster gRPC server on %s\n", endpoint.c_str());
        return false;
    }

    fprintf(stderr, "[cluster-ui] cluster gRPC:    %s (Coordinator + ClusterNode)\n", endpoint.c_str());

    // server->Wait() blocks forever - same "own detached thread" pattern as the RPC-serve half.
    // The unique_ptr is moved into the thread's storage so the server object outlives this call.
    std::thread([srv = std::shared_ptr<grpc::Server>(std::move(server))]() {
        srv->Wait();
    }).detach();

    return true;
}

// ---------------------------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    const char * env_host        = getenv("CLUSTER_UI_HOST");
    const char * env_port        = getenv("CLUSTER_UI_PORT");
    const char * env_models_dir  = getenv("MODELS_DIR");
    const char * env_static_dir  = getenv("CLUSTER_UI_STATIC_DIR");
    const char * env_download_dir = getenv("CLUSTER_DOWNLOAD_DIR");

    const std::string host       = env_host       ? env_host       : "127.0.0.1";
    const int          port      = env_port       ? atoi(env_port) : 8787;
    const std::string models_dir = env_models_dir ? env_models_dir : "prototypes/distributed-rpc/testdata";
    const std::string static_dir = env_static_dir ? env_static_dir : "prototypes/distributed-rpc/ui";

    // Per-node download destination: each machine decides for itself (no field on the gRPC
    // DownloadModelRequest to override this remotely) - defaults to MODELS_DIR itself, so a
    // download immediately shows up in this same node's own /api/models scan (which is
    // recursive - see below) without any extra config. Passed as an LLAMA_CACHE override to the
    // `llama download` subprocess's environment specifically, never by mutating this process's
    // own environment - see subprocess_runner.h's env_overrides parameter.
    const std::string download_dir = env_download_dir ? env_download_dir : models_dir;

    if (host != "127.0.0.1" && host != "localhost" && host != "::1") {
        fprintf(stderr,
            "\n"
            "*** WARNING: CLUSTER_UI_HOST=%s - binding to a non-loopback address. ***\n"
            "*** This UI can launch processes with no authentication. Anyone who  ***\n"
            "*** can reach this address and port can run arbitrary llama-cli      ***\n"
            "*** invocations on this machine. Only do this on a trusted network,  ***\n"
            "*** and prefer a loopback address plus an SSH tunnel instead.        ***\n"
            "\n",
            host.c_str());
    }

    const std::string llama_cli_path      = sibling_binary_path(argv[0], "llama-cli");
    const std::string llama_download_path = sibling_binary_path(argv[0], "llama");

    // lets the ClusterNode service's DownloadModel gRPC handler launch the same subprocess
    // /api/models/download uses over HTTP, on this node instead of just the origin's own.
    g_cluster_node.set_download_binary_path(llama_download_path);
    g_cluster_node.set_download_dir(download_dir);

    const char * env_ts_port    = getenv("TAILSCALE_RPC_PORT");
    const char * env_ts_timeout = getenv("TAILSCALE_CONNECT_TIMEOUT_MS");
    const int    ts_port        = env_ts_port    ? atoi(env_ts_port)    : 50052;
    const int    ts_timeout_ms  = env_ts_timeout ? atoi(env_ts_timeout) : 500;

    // one-time backend registry scan, needed by tailscale_discover_nodes()'s RPC handshake step
    // and by the embedded RPC server below.
    ggml_backend_load_all();

    // "server" half of this binary's dual role - see the file header comment. Runs on its own
    // background thread; failure here is logged but doesn't prevent the UI half from starting.
    if (!start_rpc_serve_background()) {
        fprintf(stderr, "[cluster-ui] warning: RPC serve failed to start; this node will not contribute compute\n");
    }

    const char * env_grpc_host = getenv("CLUSTER_GRPC_HOST");
    const char * env_grpc_port = getenv("CLUSTER_GRPC_PORT");
    const std::string grpc_host = env_grpc_host ? env_grpc_host : "0.0.0.0"; // same reasoning as RPC_SERVE_HOST
    const int          grpc_port = env_grpc_port ? atoi(env_grpc_port) : 50053;

    if (!start_cluster_grpc_server(grpc_host, grpc_port)) {
        fprintf(stderr, "[cluster-ui] warning: cluster gRPC server failed to start; this node cannot join a gRPC pipeline\n");
    }

    // Where a remote trunk node's final PassOff should reach *this* node when it's the origin -
    // best-effort default for local/loopback testing; override for real multi-machine tailnet
    // use (this phase does not attempt automatic Tailscale self-IP detection - see plan notes).
    const char * env_self_endpoint = getenv("CLUSTER_GRPC_SELF_ENDPOINT");
    const std::string self_endpoint = env_self_endpoint ? env_self_endpoint
                                                          : ("127.0.0.1:" + std::to_string(grpc_port));

    httplib::Server svr;

    if (!svr.set_mount_point("/", static_dir)) {
        fprintf(stderr, "warning: static UI directory not found: %s (GET / will 404)\n", static_dir.c_str());
    }

    svr.Get("/api/discover", [ts_port, ts_timeout_ms](const httplib::Request &, httplib::Response & res) {
        std::vector<discovered_node> found;
        std::string                  error;
        if (!tailscale_discover_nodes(ts_port, ts_timeout_ms, found, error)) {
            res.status = 500;
            res.set_content(json{{"error", error}}.dump(), "application/json");
            return;
        }

        json arr = json::array();
        for (const auto & node : found) {
            arr.push_back({
                {"hostname",    node.hostname},
                {"ip",          node.ip},
                {"port",        node.port},
                {"endpoint",    node.endpoint},
                {"free_bytes",  node.free_bytes},
                {"total_bytes", node.total_bytes},
            });
        }
        res.set_content(arr.dump(), "application/json");
    });

    // Debug/inspection endpoint for the Coordinator service's in-memory member table (see
    // plan file history's verification notes) - also generally useful once the UI wants to show
    // "who has said Hello to me" independent of a fresh Tailscale scan.
    svr.Get("/api/members", [](const httplib::Request &, httplib::Response & res) {
        json arr = json::array();
        for (const auto & m : g_coordinator.snapshot_members()) {
            arr.push_back({
                {"node_id",              m.node_id()},
                {"rpc_endpoint",         m.rpc_endpoint()},
                {"cluster_node_endpoint", m.cluster_node_endpoint()},
                {"free_vram_bytes",      m.caps().free_vram_bytes()},
            });
        }
        res.set_content(arr.dump(), "application/json");
    });

    svr.Get("/api/models", [models_dir](const httplib::Request &, httplib::Response & res) {
        json arr = json::array();

        std::vector<std::string> gguf_paths;
        scan_gguf_files_recursive(models_dir, /*depth_remaining=*/6, gguf_paths);

        for (const auto & path : gguf_paths) {
            const size_t slash = path.find_last_of('/');
            const std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);

            gguf_meta meta;
            json entry_json = {{"name", name}, {"path", path}};
            if (read_gguf_meta(path, meta)) {
                entry_json["arch"]       = meta.arch;
                entry_json["n_layers"]   = meta.n_layers;
                entry_json["size_bytes"] = meta.size_bytes;
            }
            arr.push_back(entry_json);
        }

        res.set_content(arr.dump(), "application/json");
    });

    // Reads GGUF metadata for an arbitrary absolute path - used for models resolved via
    // /api/models/download, which typically live in llama.cpp's HF cache directory rather than
    // `models_dir`, so they don't show up in the /api/models scan above.
    svr.Get("/api/model-meta", [](const httplib::Request & req, httplib::Response & res) {
        const std::string path = req.get_param_value("path");
        if (path.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "path query parameter is required"}}.dump(), "application/json");
            return;
        }

        gguf_meta meta;
        if (!read_gguf_meta(path, meta)) {
            res.status = 404;
            res.set_content(json{{"error", "failed to read GGUF metadata for path"}}.dump(), "application/json");
            return;
        }

        res.set_content(json{
            {"path",       path},
            {"arch",       meta.arch},
            {"n_layers",   meta.n_layers},
            {"size_bytes", meta.size_bytes},
        }.dump(), "application/json");
    });

    svr.Post("/api/launch", [llama_cli_path](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("invalid JSON: ") + e.what()}}.dump(), "application/json");
            return;
        }

        const std::string model_path = body.value("model_path", "");
        const std::string prompt     = body.value("prompt", "");

        // Read as int64 rather than int: nlohmann::json's value<int>() silently truncates an
        // out-of-range JSON number (e.g. wraps to INT32_MIN) instead of throwing, so validating
        // only after a narrowing get<int>() would already be too late. Bound-check explicitly
        // before it ever becomes the process's -n argument.
        int64_t n_predict_wide = 64;
        try {
            n_predict_wide = body.value("n_predict", (int64_t) 64);
        } catch (const json::exception &) {
            // non-numeric n_predict (e.g. a string) - fall through to the range check below,
            // which will reject the default-preserving 64 only if that's somehow out of range.
        }
        if (n_predict_wide < 1 || n_predict_wide > 65536) {
            res.status = 400;
            res.set_content(json{{"error", "n_predict must be between 1 and 65536"}}.dump(), "application/json");
            return;
        }
        const int n_predict = (int) n_predict_wide;

        struct stat st;
        if (model_path.empty() || stat(model_path.c_str(), &st) != 0) {
            res.status = 400;
            res.set_content(json{{"error", "model_path does not exist"}}.dump(), "application/json");
            return;
        }

        std::string endpoints;
        if (body.contains("nodes") && body["nodes"].is_array()) {
            for (const auto & node : body["nodes"]) {
                if (!endpoints.empty()) {
                    endpoints += ",";
                }
                endpoints += node.value("endpoint", "");
            }
        }

        std::vector<std::string> argv_strs = {
            llama_cli_path,
            "--split-mode", "layer",
            "--n-gpu-layers", "all",
            "--model", model_path,
            "--prompt", prompt,
            "-n", std::to_string(n_predict),
            // without this, llama-cli finishes the predefined --prompt turn and then sits in an
            // interactive REPL loop forever (emitting empty "> " prompts) since its stdin is
            // redirected from /dev/null here and never yields EOF-driven exit on its own.
            "--single-turn",
        };
        if (!endpoints.empty()) {
            argv_strs.push_back("--distributed");
            argv_strs.push_back(endpoints);
        }

        auto state = launch_process(argv_strs);

        std::string run_id;
        {
            std::lock_guard<std::mutex> lock(g_runs_mtx);
            run_id = std::to_string(g_next_run_id++);
            g_runs[run_id] = state;
        }

        res.set_content(json{{"run_id", run_id}}.dump(), "application/json");
    });

    // Real U-shaped gRPC pipeline (see plan file history: "gRPC Coordinator/ClusterNode
    // Implementation"). Unlike /api/launch above (plain --distributed, whole model split flatly
    // across visible devices), this keeps the head (embedding + first layers) and tail (last
    // layers + lm_head + sampling) local to this node, and delegates only the middle "trunk"
    // range to remote peers via real Hello + AssignLayers + PassOff gRPC calls. [phase scope]
    // proves one forward pass (prompt -> hidden-state hand-off chain -> one sampled token), not
    // full autoregressive generation - see plan notes for what's deferred.
    //
    // Body: {model_path, prompt, head_layers, tail_layers,
    //        trunk: [{endpoint, layer_start, layer_end}, ...]}  (already computed client-side by
    //        the same proportional-split JS used for the illustrative preview elsewhere in the UI)
    svr.Post("/api/launch-grpc", [self_endpoint](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("invalid JSON: ") + e.what()}}.dump(), "application/json");
            return;
        }

        const std::string model_path = body.value("model_path", "");
        const std::string prompt     = body.value("prompt", "");

        // Same bound-checked read as /api/launch's n_predict (see its comment) - reused here
        // now that this handler actually loops instead of producing a single token.
        int64_t n_predict_wide = 64;
        try {
            n_predict_wide = body.value("n_predict", (int64_t) 64);
        } catch (const json::exception &) {
        }
        if (n_predict_wide < 1 || n_predict_wide > 65536) {
            res.status = 400;
            res.set_content(json{{"error", "n_predict must be between 1 and 65536"}}.dump(), "application/json");
            return;
        }
        const int n_predict = (int) n_predict_wide;

        struct stat st;
        if (model_path.empty() || stat(model_path.c_str(), &st) != 0) {
            res.status = 400;
            res.set_content(json{{"error", "model_path does not exist"}}.dump(), "application/json");
            return;
        }

        gguf_meta meta;
        if (!read_gguf_meta(model_path, meta) || meta.n_layers == 0) {
            res.status = 400;
            res.set_content(json{{"error", "failed to read model layer count"}}.dump(), "application/json");
            return;
        }
        const int32_t n_layer = (int32_t) meta.n_layers;

        int64_t head_layers_wide = body.value("head_layers", (int64_t) 1);
        int64_t tail_layers_wide = body.value("tail_layers", (int64_t) 1);
        // head_layers + tail_layers == n_layer is valid (empty trunk, fully local); only
        // exceeding n_layer is an actual error (would mean tail_start < head_end, an overlap).
        if (head_layers_wide < 1 || tail_layers_wide < 1 || head_layers_wide + tail_layers_wide > n_layer) {
            res.status = 400;
            res.set_content(json{{"error", "head_layers/tail_layers out of range for this model's layer count"}}.dump(), "application/json");
            return;
        }
        const int32_t head_end   = (int32_t) head_layers_wide;
        const int32_t tail_start = n_layer - (int32_t) tail_layers_wide;

        struct trunk_hop {
            std::string endpoint;
            int32_t     layer_start;
            int32_t     layer_end;
        };
        std::vector<trunk_hop> trunk;
        if (body.contains("trunk") && body["trunk"].is_array()) {
            for (const auto & hop : body["trunk"]) {
                trunk.push_back({hop.value("endpoint", ""), (int32_t) hop.value("layer_start", 0),
                                  (int32_t) hop.value("layer_end", 0)});
            }
        }

        // Every layer index in [head_end, tail_start) must be covered by exactly one hop, with
        // no gaps or overlaps - otherwise a hidden state gets handed to a runner expecting a
        // different layer boundary than what was actually computed, silently producing garbage
        // rather than a clear error.
        {
            int32_t expected_start = head_end;
            bool    coverage_ok    = true;
            for (const auto & hop : trunk) {
                if (hop.layer_start != expected_start) {
                    coverage_ok = false;
                    break;
                }
                expected_start = hop.layer_end;
            }
            if (!coverage_ok || expected_start != tail_start) {
                res.status = 400;
                res.set_content(json{{"error", "trunk layer ranges must contiguously cover [head_end, tail_start) with no gaps or overlaps"}}.dump(), "application/json");
                return;
            }
        }

        static std::mutex     request_id_mtx;
        static uint64_t       next_request_id = 1;
        std::string request_id;
        {
            std::lock_guard<std::mutex> lock(request_id_mtx);
            request_id = "req-" + std::to_string(next_request_id++);
        }

        // 1. Hello + AssignLayers each trunk node in order, chaining next_node_endpoint through
        //    to either the next trunk node or, for the last one, back to this node (self_endpoint).
        for (size_t i = 0; i < trunk.size(); ++i) {
            const std::string next = (i + 1 < trunk.size()) ? trunk[i + 1].endpoint : self_endpoint;

            llama_cluster::HelloRequest  hello_req;
            llama_cluster::HelloResponse hello_res;
            hello_req.set_node_id("origin-" + request_id);
            hello_req.set_cluster_node_endpoint(self_endpoint);
            std::string hello_err;
            if (!cluster_grpc_call_hello(trunk[i].endpoint, hello_req, hello_res, hello_err)) {
                res.status = 502;
                res.set_content(json{{"error", "Hello to " + trunk[i].endpoint + " failed: " + hello_err}}.dump(), "application/json");
                return;
            }

            llama_cluster::AssignLayersRequest  assign_req;
            llama_cluster::AssignLayersResponse assign_res;
            assign_req.set_request_id(request_id);
            assign_req.set_model_path(model_path);
            assign_req.set_layer_start((uint32_t) trunk[i].layer_start);
            assign_req.set_layer_end((uint32_t) trunk[i].layer_end);
            assign_req.set_next_node_endpoint(next);
            std::string assign_err;
            if (!cluster_grpc_call_assign_layers(trunk[i].endpoint, assign_req, assign_res, assign_err)) {
                res.status = 502;
                res.set_content(json{{"error", "AssignLayers to " + trunk[i].endpoint + " failed: " + assign_err}}.dump(), "application/json");
                return;
            }
            if (!assign_res.accepted()) {
                res.status = 502;
                res.set_content(json{{"error", trunk[i].endpoint + " rejected AssignLayers: " + assign_res.reject_reason()}}.dump(), "application/json");
                return;
            }
        }

        // 2. Register this node's own tail runner *before* kicking off the chain, so an
        //    early-arriving PassOff can never race past an unregistered request_id.
        auto tail_runner = std::make_unique<cluster_layer_runner>();
        std::string tail_error;
        if (!tail_runner->prepare(model_path, tail_start, n_layer, /*n_ctx=*/4096, tail_error)) {
            res.status = 500;
            res.set_content(json{{"error", "failed to prepare local tail: " + tail_error}}.dump(), "application/json");
            return;
        }
        g_cluster_node.register_local_runner(request_id, std::move(tail_runner), /*next_node_endpoint=*/"");

        // 3. Run this node's own head locally (never a self-directed gRPC hop). Kept in
        //    g_head_runners under request_id (rather than as a stack local) so its KV cache
        //    survives past this handler call - see PROGRESS.md step 3.
        auto head_runner = std::make_unique<cluster_layer_runner>();
        std::string head_error;
        if (!head_runner->prepare(model_path, 0, head_end, /*n_ctx=*/4096, head_error)) {
            res.status = 500;
            res.set_content(json{{"error", "failed to prepare local head: " + head_error}}.dump(), "application/json");
            return;
        }
        cluster_hidden_state hidden;
        if (!head_runner->run_head(prompt, hidden, head_error)) {
            res.status = 500;
            res.set_content(json{{"error", "local head failed: " + head_error}}.dump(), "application/json");
            return;
        }
        cluster_layer_runner * head_runner_ptr = head_runner.get();
        {
            std::lock_guard<std::mutex> lock(g_head_runners_mtx);
            g_head_runners[request_id] = std::move(head_runner);
        }

        // Tears down every trunk node's session for this request_id (their AssignLayers-
        // registered runner/KV-cache never gets freed otherwise - see PROGRESS.md step 5).
        // Best-effort: a trunk node that's already gone is nothing to retry, so errors are
        // swallowed rather than failing the whole response over cleanup.
        auto end_trunk_sessions = [&trunk, &request_id]() {
            for (const auto & hop : trunk) {
                std::string end_session_error;
                cluster_grpc_call_end_session(hop.endpoint, request_id, end_session_error);
            }
        };

        // 4. Hand off to the first trunk node (or straight back to this node's own tail if no
        //    trunk nodes were selected - a degenerate but valid all-local U "split").
        const std::string first_hop = trunk.empty() ? self_endpoint : trunk.front().endpoint;
        std::string pass_off_error;
        if (!cluster_grpc_send_pass_off(first_hop, request_id, hidden, pass_off_error)) {
            g_cluster_node.end_session(request_id);
            end_trunk_sessions();
            {
                std::lock_guard<std::mutex> lock(g_head_runners_mtx);
                g_head_runners.erase(request_id);
            }
            res.status = 502;
            res.set_content(json{{"error", "initial PassOff to " + first_hop + " failed: " + pass_off_error}}.dump(), "application/json");
            return;
        }

        // 5. Generation loop: repeatedly wait for the tail's sampled token, feed it back into
        //    the head as the next position, and PassOff the resulting hidden state through the
        //    same trunk chain again - until EOG or n_predict tokens have been produced. Every
        //    node's llama_context (head here, trunk/tail via g_cluster_node) just grows its own
        //    KV cache by one position per iteration; no cache-sharding code needed, only this
        //    loop (see PROGRESS.md step 4).
        std::vector<cluster_tail_result> generated;
        std::string loop_error;
        bool loop_ok = true;
        int32_t next_pos = hidden.n_tokens; // prompt occupied positions [0, hidden.n_tokens)

        for (int step = 0; step < n_predict; ++step) {
            cluster_tail_result result;
            std::string wait_error;
            if (!g_cluster_node.wait_for_next_token(request_id, /*timeout_ms=*/30000, result, wait_error)) {
                loop_ok   = false;
                loop_error = "pipeline did not complete: " + wait_error;
                break;
            }
            generated.push_back(result);
            if (result.is_eog || step + 1 >= n_predict) {
                break;
            }

            cluster_hidden_state next_hidden;
            std::string next_head_error;
            if (!head_runner_ptr->run_head_next(result.token, next_pos, next_hidden, next_head_error)) {
                loop_ok    = false;
                loop_error = "local head continuation failed: " + next_head_error;
                break;
            }
            ++next_pos;

            if (!cluster_grpc_send_pass_off(first_hop, request_id, next_hidden, pass_off_error)) {
                loop_ok    = false;
                loop_error = "PassOff to " + first_hop + " failed: " + pass_off_error;
                break;
            }
        }

        g_cluster_node.end_session(request_id);
        end_trunk_sessions();
        {
            std::lock_guard<std::mutex> lock(g_head_runners_mtx);
            g_head_runners.erase(request_id);
        }

        if (!loop_ok) {
            res.status = 502;
            res.set_content(json{{"error", loop_error}}.dump(), "application/json");
            return;
        }

        json tokens_json = json::array();
        std::string text;
        for (const auto & r : generated) {
            tokens_json.push_back({{"token", r.token}, {"piece", r.piece}, {"is_eog", r.is_eog}});
            text += r.piece;
        }
        res.set_content(json{
            {"text",   text},
            {"tokens", tokens_json},
        }.dump(), "application/json");
    });

    // Downloads a model from Hugging Face via the `llama download` subcommand (app/download.cpp)
    // - the same tool `llama-cli --hf-repo` uses internally, just as a standalone step here so
    // the resolved local path can be surfaced to the UI and reused across launches without
    // re-downloading. Tracked as a run exactly like /api/launch, so GET /api/runs/{id} already
    // knows how to poll it - no new tracking code needed.
    svr.Post("/api/models/download", [llama_download_path, download_dir](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("invalid JSON: ") + e.what()}}.dump(), "application/json");
            return;
        }

        const std::string hf_repo = body.value("hf_repo", "");
        const std::string hf_file = body.value("hf_file", "");

        if (hf_repo.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "hf_repo is required"}}.dump(), "application/json");
            return;
        }

        std::vector<std::string> argv_strs = {
            llama_download_path,
            "download",
            "-hf", hf_repo,
        };
        if (!hf_file.empty()) {
            argv_strs.push_back("-hff");
            argv_strs.push_back(hf_file);
        }

        // CLUSTER_DOWNLOAD_DIR (default: MODELS_DIR) overrides where this download lands, via
        // the child process's own LLAMA_CACHE - never by mutating this process's global env.
        auto state = launch_process(argv_strs, {{"LLAMA_CACHE", download_dir}});

        std::string run_id;
        {
            std::lock_guard<std::mutex> lock(g_runs_mtx);
            run_id = std::to_string(g_next_run_id++);
            g_runs[run_id] = state;
        }

        res.set_content(json{{"run_id", run_id}}.dump(), "application/json");
    });

    // Remote-node model provisioning over the cluster gRPC protocol's DownloadModel/
    // GetDownloadStatus RPCs (proto/cluster.proto) - lets the origin ensure a trunk node has a
    // model on disk before AssignLayers is called against it, the same way /api/models/download
    // above provisions the origin's own local model. Body: {endpoint, hf_repo, hf_file}.
    svr.Post("/api/nodes/download", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("invalid JSON: ") + e.what()}}.dump(), "application/json");
            return;
        }

        const std::string endpoint = body.value("endpoint", "");
        const std::string hf_repo  = body.value("hf_repo", "");
        const std::string hf_file  = body.value("hf_file", "");

        if (endpoint.empty() || hf_repo.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "endpoint and hf_repo are required"}}.dump(), "application/json");
            return;
        }

        llama_cluster::DownloadModelRequest  grpc_req;
        llama_cluster::DownloadModelResponse grpc_res;
        grpc_req.set_hf_repo(hf_repo);
        grpc_req.set_hf_file(hf_file);

        std::string error;
        if (!cluster_grpc_call_download_model(endpoint, grpc_req, grpc_res, error)) {
            res.status = 502;
            res.set_content(json{{"error", "DownloadModel to " + endpoint + " failed: " + error}}.dump(), "application/json");
            return;
        }
        if (!grpc_res.accepted()) {
            res.status = 400;
            res.set_content(json{{"error", grpc_res.reject_reason()}}.dump(), "application/json");
            return;
        }

        res.set_content(json{{"download_id", grpc_res.download_id()}}.dump(), "application/json");
    });

    svr.Get("/api/nodes/download-status", [](const httplib::Request & req, httplib::Response & res) {
        const std::string endpoint    = req.get_param_value("endpoint");
        const std::string download_id = req.get_param_value("download_id");

        if (endpoint.empty() || download_id.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "endpoint and download_id query parameters are required"}}.dump(), "application/json");
            return;
        }

        llama_cluster::GetDownloadStatusRequest  grpc_req;
        llama_cluster::GetDownloadStatusResponse grpc_res;
        grpc_req.set_download_id(download_id);

        std::string error;
        if (!cluster_grpc_call_get_download_status(endpoint, grpc_req, grpc_res, error)) {
            res.status = 502;
            res.set_content(json{{"error", "GetDownloadStatus to " + endpoint + " failed: " + error}}.dump(), "application/json");
            return;
        }

        const char * state_name =
            grpc_res.state() == llama_cluster::DOWNLOAD_STATE_RUNNING ? "running" :
            grpc_res.state() == llama_cluster::DOWNLOAD_STATE_DONE    ? "done" :
            grpc_res.state() == llama_cluster::DOWNLOAD_STATE_FAILED  ? "failed" : "unknown";

        res.set_content(json{
            {"state",       state_name},
            {"model_path",  grpc_res.model_path()},
            {"error",       grpc_res.error()},
            {"output_tail", grpc_res.output_tail()},
        }.dump(), "application/json");
    });

    svr.Get("/api/runs/:id", [](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.path_params.at("id");

        std::shared_ptr<run_state> state;
        {
            std::lock_guard<std::mutex> lock(g_runs_mtx);
            auto it = g_runs.find(id);
            if (it == g_runs.end()) {
                res.status = 404;
                res.set_content(json{{"error", "unknown run id"}}.dump(), "application/json");
                return;
            }
            state = it->second;
        }

        std::lock_guard<std::mutex> lock(state->mtx);
        res.set_content(json{
            {"running",   state->running},
            {"output",    state->output},
            {"exit_code", state->exit_code},
        }.dump(), "application/json");
    });

    svr.Post("/api/runs/:id/stop", [](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.path_params.at("id");

        std::shared_ptr<run_state> state;
        {
            std::lock_guard<std::mutex> lock(g_runs_mtx);
            auto it = g_runs.find(id);
            if (it == g_runs.end()) {
                res.status = 404;
                res.set_content(json{{"error", "unknown run id"}}.dump(), "application/json");
                return;
            }
            state = it->second;
        }

#ifdef _WIN32
        if (state->process) {
            TerminateProcess(state->process, 1);
        }
#else
        if (state->pid > 0) {
            kill(state->pid, SIGTERM);
        }
#endif
        res.set_content(json{{"ok", true}}.dump(), "application/json");
    });

    fprintf(stderr, "[cluster-ui] llama-cli:      %s\n", llama_cli_path.c_str());
    fprintf(stderr, "[cluster-ui] llama download: %s\n", llama_download_path.c_str());
    fprintf(stderr, "[cluster-ui] discover:       in-process (port %d, timeout %d ms)\n", ts_port, ts_timeout_ms);
    fprintf(stderr, "[cluster-ui] models dir:     %s\n", models_dir.c_str());
    fprintf(stderr, "[cluster-ui] download dir:   %s%s\n", download_dir.c_str(),
            download_dir == models_dir ? " (= models dir)" : "");
    fprintf(stderr, "[cluster-ui] static dir:     %s\n", static_dir.c_str());
    fprintf(stderr, "[cluster-ui] listening on http://%s:%d\n", host.c_str(), port);

    if (!svr.listen(host, port)) {
        fprintf(stderr, "[cluster-ui] error: failed to listen on %s:%d\n", host.c_str(), port);
        return 1;
    }

    return 0;
}
