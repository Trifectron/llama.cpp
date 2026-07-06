#include "subprocess_runner.h"

#ifndef _WIN32
#  include <sys/wait.h>
#  include <fcntl.h>
#  include <signal.h>
#  include <unistd.h>
extern char ** environ;
#endif

#include <cstdio>
#include <cstring>
#include <thread>

std::string sibling_binary_path(const char * self_argv0, const char * name) {
    std::string self(self_argv0);
#ifdef _WIN32
    const size_t slash = self.find_last_of("/\\");
    const std::string dir = (slash == std::string::npos) ? "." : self.substr(0, slash);
    // CreateProcess requires an explicit extension in the application name
    return dir + "\\" + name + ".exe";
#else
    const size_t slash = self.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? "." : self.substr(0, slash);
    return dir + "/" + name;
#endif
}

#ifdef _WIN32

// Quotes one argv entry per CommandLineToArgvW's rules: backslashes are literal unless they
// precede a double quote, in which case they (and the quote) must be escaped.
static std::string win_quote_arg(const std::string & arg) {
    if (!arg.empty() && arg.find_first_of(" \t\"") == std::string::npos) {
        return arg;
    }
    std::string out = "\"";
    size_t n_backslash = 0;
    for (const char c : arg) {
        if (c == '\\') {
            n_backslash++;
            continue;
        }
        if (c == '"') {
            out.append(n_backslash * 2 + 1, '\\');
        } else {
            out.append(n_backslash, '\\');
        }
        out += c;
        n_backslash = 0;
    }
    out.append(n_backslash * 2, '\\');
    out += '"';
    return out;
}

std::shared_ptr<run_state> launch_process(const std::vector<std::string> & argv_strs) {
    auto state = std::make_shared<run_state>();

    SECURITY_ATTRIBUTES sa;
    memset(&sa, 0, sizeof(sa));
    sa.nLength        = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE pipe_rd = NULL;
    HANDLE pipe_wr = NULL;
    if (!CreatePipe(&pipe_rd, &pipe_wr, &sa, 0)) {
        state->running   = false;
        state->exit_code = -1;
        state->output    = "error: CreatePipe() failed\n";
        return state;
    }
    // only the write end may be inherited by the child
    SetHandleInformation(pipe_rd, HANDLE_FLAG_INHERIT, 0);

    // never let the child block waiting for interactive input - it has no terminal here.
    HANDLE devnull = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                                 OPEN_EXISTING, 0, NULL);

    std::string cmdline;
    for (const auto & s : argv_strs) {
        if (!cmdline.empty()) {
            cmdline += ' ';
        }
        cmdline += win_quote_arg(s);
    }
    std::vector<char> cmdline_buf(cmdline.begin(), cmdline.end());
    cmdline_buf.push_back('\0');

    STARTUPINFOA si;
    memset(&si, 0, sizeof(si));
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdInput  = devnull;
    si.hStdOutput = pipe_wr;
    si.hStdError  = pipe_wr;

    PROCESS_INFORMATION pi;
    memset(&pi, 0, sizeof(pi));

    const BOOL ok = CreateProcessA(argv_strs[0].c_str(), cmdline_buf.data(), NULL, NULL,
                                   TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    CloseHandle(pipe_wr);
    if (devnull != INVALID_HANDLE_VALUE) {
        CloseHandle(devnull);
    }
    if (!ok) {
        CloseHandle(pipe_rd);
        state->running   = false;
        state->exit_code = -1;
        state->output    = "error: CreateProcess(" + argv_strs[0] + ") failed\n";
        return state;
    }
    CloseHandle(pi.hThread);
    state->process = pi.hProcess;

    std::thread reader([state, pipe_rd]() {
        char  buf[4096];
        DWORD n = 0;
        // ReadFile fails with ERROR_BROKEN_PIPE once the child exits and the pipe drains
        while (ReadFile(pipe_rd, buf, sizeof(buf), &n, NULL) && n > 0) {
            std::lock_guard<std::mutex> lock(state->mtx);
            state->output.append(buf, (size_t) n);
        }
        CloseHandle(pipe_rd);

        WaitForSingleObject(state->process, INFINITE);
        DWORD code = (DWORD) -1;
        GetExitCodeProcess(state->process, &code);

        std::lock_guard<std::mutex> lock(state->mtx);
        state->running   = false;
        state->exit_code = (int) code;
    });
    reader.detach();

    return state;
}

#else

std::shared_ptr<run_state> launch_process(const std::vector<std::string> & argv_strs) {
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

#endif // _WIN32
