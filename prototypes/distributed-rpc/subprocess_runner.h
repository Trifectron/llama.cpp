#pragma once

// Shared subprocess-launching primitives, extracted from cluster-ui.cpp so both its own HTTP
// handlers and cluster_node_service.cpp's DownloadModel gRPC handler can launch tracked child
// processes (e.g. `llama download -hf ...`) the same way: fork()+execve() (CreateProcess on
// Windows) with an explicit argv array - never a shell string, so there is nothing for shell
// metacharacters in a model path or prompt to be interpreted by.

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/types.h>
#endif

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct run_state {
#ifdef _WIN32
    HANDLE      process = NULL;

    ~run_state() {
        if (process) {
            CloseHandle(process);
        }
    }
#else
    pid_t       pid     = -1;
#endif
    std::mutex  mtx;
    std::string output;
    bool        running = true;
    int         exit_code = -1;
};

// Resolves a sibling binary in the same directory this binary was launched from (e.g.
// "llama-cli", "llama"), so cluster-ui works regardless of cwd as long as it's run from the
// build's bin/ directory - matching how the other prototype tools are already invoked
// (build-rpc/bin/...).
std::string sibling_binary_path(const char * self_argv0, const char * name);

// Launches argv_strs[0] with the remaining entries as its argv, redirecting stdout+stderr into
// the returned run_state's output buffer (accumulated on a background thread) and stdin from
// the null device (the child never has a terminal to block on). Returns immediately; poll
// run_state::running/exit_code to observe completion.
//
// env_overrides: {name, value} pairs added to (or replacing, by name) this process's own
// environment for the child only - built into a fresh envp/environment-block passed directly to
// execve()/CreateProcessA(), never by mutating this process's global environment (setenv() would
// race against launch_process() being called concurrently from multiple request-handler
// threads, which it is).
std::shared_ptr<run_state> launch_process(const std::vector<std::string> & argv_strs,
                                            const std::vector<std::pair<std::string, std::string>> & env_overrides = {});
