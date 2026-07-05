// Live control-panel backend for the distributed-rpc prototype (see plan.md and the plan file
// history: "Live Cluster Control-Panel UI"). Serves a small browser UI that discovers Tailscale
// peers running ggml-rpc-server, lists local GGUF models, and launches llama-cli directly - no
// shell script in between (host-run.sh was removed; this replaces its one job).
//
// Deliberately does not link ggml/llama - it has no need to load a model or speak RPC itself,
// it only orchestrates two other binaries in this same directory (llama-tailscale-discover,
// llama-cli) as subprocesses, the same way a human operator would from a terminal.

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

extern char ** environ;

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

// ---------------------------------------------------------------------------------------------
// Subprocess launching - fork()+execve() with an explicit argv array and explicit envp, never a
// shell string. Model paths and prompt text pass through as literal argv entries; there is no
// shell to interpret metacharacters in them.
// ---------------------------------------------------------------------------------------------

struct run_state {
    pid_t       pid     = -1;
    std::mutex  mtx;
    std::string output;
    bool        running = true;
    int         exit_code = -1;
};

static std::mutex                                    g_runs_mtx;
static std::map<std::string, std::shared_ptr<run_state>> g_runs;
static int                                            g_next_run_id = 1;

// Resolves a sibling binary in the same directory this binary was launched from (e.g.
// "llama-cli", "llama-tailscale-discover"), so cluster-ui works regardless of cwd as long as
// it's run from the build's bin/ directory - matching how the other prototype tools are already
// invoked (build-rpc/bin/...).
static std::string sibling_binary_path(const char * self_argv0, const char * name) {
    std::string self(self_argv0);
    const size_t slash = self.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? "." : self.substr(0, slash);
    return dir + "/" + name;
}

static std::shared_ptr<run_state> launch_process(const std::vector<std::string> & argv_strs) {
    auto state = std::make_shared<run_state>();

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        state->running   = false;
        state->exit_code = -1;
        state->output    = "error: pipe() failed\n";
        return state;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        state->running   = false;
        state->output    = "error: fork() failed\n";
        return state;
    }

    if (pid == 0) {
        // child
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);

        // never let the child block waiting for interactive input - it has no terminal here.
        const int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) {
            dup2(devnull, STDIN_FILENO);
            close(devnull);
        }

        std::vector<char *> argv;
        argv.reserve(argv_strs.size() + 1);
        for (const auto & s : argv_strs) {
            argv.push_back(const_cast<char *>(s.c_str()));
        }
        argv.push_back(nullptr);

        execve(argv[0], argv.data(), environ);
        // execve only returns on failure
        fprintf(stderr, "error: execve(%s) failed: %s\n", argv[0], strerror(errno));
        _exit(127);
    }

    // parent
    close(pipefd[1]);
    state->pid = pid;

    const int read_fd = pipefd[0];
    std::thread reader([state, read_fd, pid]() {
        char buf[4096];
        ssize_t n;
        while ((n = read(read_fd, buf, sizeof(buf))) > 0) {
            std::lock_guard<std::mutex> lock(state->mtx);
            state->output.append(buf, (size_t) n);
        }
        close(read_fd);

        int status = 0;
        waitpid(pid, &status, 0);

        std::lock_guard<std::mutex> lock(state->mtx);
        state->running   = false;
        state->exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    });
    reader.detach();

    return state;
}

// ---------------------------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    const char * env_host       = getenv("CLUSTER_UI_HOST");
    const char * env_port       = getenv("CLUSTER_UI_PORT");
    const char * env_models_dir = getenv("MODELS_DIR");
    const char * env_static_dir = getenv("CLUSTER_UI_STATIC_DIR");

    const std::string host       = env_host       ? env_host       : "127.0.0.1";
    const int          port      = env_port       ? atoi(env_port) : 8787;
    const std::string models_dir = env_models_dir ? env_models_dir : "prototypes/distributed-rpc/testdata";
    const std::string static_dir = env_static_dir ? env_static_dir : "prototypes/distributed-rpc/ui";

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

    const std::string llama_cli_path = sibling_binary_path(argv[0], "llama-cli");
    const std::string discover_path  = sibling_binary_path(argv[0], "llama-tailscale-discover");

    httplib::Server svr;

    if (!svr.set_mount_point("/", static_dir)) {
        fprintf(stderr, "warning: static UI directory not found: %s (GET / will 404)\n", static_dir.c_str());
    }

    svr.Get("/api/discover", [discover_path](const httplib::Request &, httplib::Response & res) {
        FILE * pipe = popen((discover_path + " --json 2>/dev/null").c_str(), "r");
        if (!pipe) {
            res.status = 500;
            res.set_content(R"({"error":"failed to run llama-tailscale-discover"})", "application/json");
            return;
        }
        std::string output;
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
            output.append(buf, n);
        }
        pclose(pipe);

        if (output.empty()) {
            output = "[]";
        }
        res.set_content(output, "application/json");
    });

    svr.Get("/api/models", [models_dir](const httplib::Request &, httplib::Response & res) {
        json arr = json::array();

        DIR * dir = opendir(models_dir.c_str());
        if (dir) {
            struct dirent * entry;
            while ((entry = readdir(dir)) != nullptr) {
                const std::string name(entry->d_name);
                if (name.size() < 5 || name.compare(name.size() - 5, 5, ".gguf") != 0) {
                    continue;
                }
                const std::string path = models_dir + "/" + name;
                gguf_meta meta;
                json entry_json = {{"name", name}, {"path", path}};
                if (read_gguf_meta(path, meta)) {
                    entry_json["arch"]        = meta.arch;
                    entry_json["n_layers"]    = meta.n_layers;
                    entry_json["size_bytes"]  = meta.size_bytes;
                }
                arr.push_back(entry_json);
            }
            closedir(dir);
        }

        res.set_content(arr.dump(), "application/json");
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
        const int          n_predict = body.value("n_predict", 64);

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

        if (state->pid > 0) {
            kill(state->pid, SIGTERM);
        }
        res.set_content(json{{"ok", true}}.dump(), "application/json");
    });

    fprintf(stderr, "[cluster-ui] llama-cli:      %s\n", llama_cli_path.c_str());
    fprintf(stderr, "[cluster-ui] discover:       %s\n", discover_path.c_str());
    fprintf(stderr, "[cluster-ui] models dir:     %s\n", models_dir.c_str());
    fprintf(stderr, "[cluster-ui] static dir:     %s\n", static_dir.c_str());
    fprintf(stderr, "[cluster-ui] listening on http://%s:%d\n", host.c_str(), port);

    if (!svr.listen(host, port)) {
        fprintf(stderr, "[cluster-ui] error: failed to listen on %s:%d\n", host.c_str(), port);
        return 1;
    }

    return 0;
}
