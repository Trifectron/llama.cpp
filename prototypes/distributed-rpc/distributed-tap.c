#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tap_options {
    const char * model;
    const char * servers;
    const char * prompt;
    const char * trace;

    int32_t n_gpu_layers;
    uint32_t n_ctx;
    int32_t n_predict;
    int32_t n_batch;
    int32_t trace_limit;
};

struct tap_state {
    const char * trace;
    int32_t trace_limit;
    int32_t seen;
};

static void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s --model MODEL [options]\n"
        "\n"
        "options:\n"
        "  --distributed SERVERS   comma-separated RPC servers, host:port\n"
        "  --rpc SERVERS           alias for --distributed\n"
        "  --prompt TEXT           prompt to decode\n"
        "  --n-predict N           number of generated tokens, default 16\n"
        "  --ctx-size N            context size, default prompt + n-predict\n"
        "  --batch-size N          prompt batch size, default prompt tokens\n"
        "  --n-gpu-layers N|all    layers to offload, default all\n"
        "  --trace FILTER|all|none ggml tensor name substring, default l_out\n"
        "  --trace-limit N         max tensor lines to print, default 64\n"
        "  -h, --help              show this help\n",
        prog);
}

static bool parse_i32(const char * value, int32_t * out) {
    char * end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        return false;
    }
    if (parsed < INT32_MIN || parsed > INT32_MAX) {
        return false;
    }

    *out = (int32_t) parsed;
    return true;
}

static bool parse_u32(const char * value, uint32_t * out) {
    char * end = NULL;
    unsigned long parsed;

    errno = 0;
    parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > UINT32_MAX) {
        return false;
    }

    *out = (uint32_t) parsed;
    return true;
}

static char * tap_strdup(const char * value) {
    const size_t len = strlen(value);
    char * copy = (char *) malloc(len + 1);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, value, len + 1);
    return copy;
}

static char * trim(char * value) {
    char * end;

    while (*value == ' ' || *value == '\t' || *value == '\n' || *value == '\r') {
        value++;
    }

    end = value + strlen(value);
    while (end > value && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\n' || end[-1] == '\r')) {
        *--end = '\0';
    }

    return value;
}

static bool wants_trace(const char * trace, const char * name) {
    if (trace == NULL || strcmp(trace, "none") == 0) {
        return false;
    }
    if (strcmp(trace, "all") == 0) {
        return true;
    }
    return strstr(name, trace) != NULL;
}

static const char * dev_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        default: return "UNKNOWN";
    }
}

static void print_devices(void) {
    const size_t n_devices = ggml_backend_dev_count();

    fprintf(stderr, "[tap] visible ggml devices: %zu\n", n_devices);
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        size_t free_mem = 0;
        size_t total_mem = 0;

        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        fprintf(stderr,
            "[tap]   %zu: %s [%s] free=%zu total=%zu\n",
            i,
            ggml_backend_dev_name(dev),
            dev_type_name(ggml_backend_dev_type(dev)),
            free_mem,
            total_mem);
    }
}

static int register_rpc_servers(const char * servers) {
    typedef ggml_backend_reg_t (*add_rpc_server_fn)(const char * endpoint);

    ggml_backend_reg_t rpc_reg;
    add_rpc_server_fn add_server;
    char * list;
    char * cursor;
    int registered = 0;

    if (servers == NULL || servers[0] == '\0') {
        return 0;
    }

    rpc_reg = ggml_backend_reg_by_name("RPC");
    if (rpc_reg == NULL) {
        fprintf(stderr, "[tap] error: RPC backend was not found; rebuild with -DGGML_RPC=ON\n");
        return 1;
    }

    add_server = (add_rpc_server_fn) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (add_server == NULL) {
        fprintf(stderr, "[tap] error: RPC add-server function was not found\n");
        return 1;
    }

    list = tap_strdup(servers);
    if (list == NULL) {
        fprintf(stderr, "[tap] error: out of memory while parsing RPC servers\n");
        return 1;
    }

    cursor = list;
    while (cursor != NULL && *cursor != '\0') {
        char * comma = strchr(cursor, ',');
        char * endpoint;
        ggml_backend_reg_t server_reg;

        if (comma != NULL) {
            *comma = '\0';
        }

        endpoint = trim(cursor);
        if (endpoint[0] != '\0') {
            server_reg = add_server(endpoint);
            if (server_reg == NULL) {
                fprintf(stderr, "[tap] error: failed to add RPC server %s\n", endpoint);
                free(list);
                return 1;
            }

            ggml_backend_register(server_reg);
            fprintf(stderr, "[tap] registered RPC server %s\n", endpoint);
            registered++;
        }

        cursor = comma == NULL ? NULL : comma + 1;
    }

    free(list);

    if (registered == 0) {
        fprintf(stderr, "[tap] error: no RPC servers specified\n");
        return 1;
    }

    return 0;
}

static bool tap_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    struct tap_state * state = (struct tap_state *) user_data;
    const char * name = t->name[0] == '\0' ? "(unnamed)" : t->name;
    const bool trace = wants_trace(state->trace, name);

    if (ask) {
        return trace;
    }

    if (!trace) {
        return true;
    }

    if (state->trace_limit < 0 || state->seen < state->trace_limit) {
        const char * buffer = t->buffer == NULL ? "(no-buffer)" : ggml_backend_buffer_name(t->buffer);
        fprintf(stderr,
            "[tap] node=%-24s op=%-12s type=%-5s ne=[%lld,%lld,%lld,%lld] bytes=%zu buffer=%s\n",
            name,
            ggml_op_desc(t),
            ggml_type_name(t->type),
            (long long) t->ne[0],
            (long long) t->ne[1],
            (long long) t->ne[2],
            (long long) t->ne[3],
            ggml_nbytes(t),
            buffer);
    }

    state->seen++;
    return true;
}

static bool parse_args(int argc, char ** argv, struct tap_options * opts) {
    opts->model = NULL;
    opts->servers = NULL;
    opts->prompt = "Hello from a C-side distributed ggml tap.";
    opts->trace = "l_out";
    opts->n_gpu_layers = -1;
    opts->n_ctx = 0;
    opts->n_predict = 16;
    opts->n_batch = 0;
    opts->trace_limit = 64;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        const char * value = NULL;

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        }

        if (i + 1 < argc) {
            value = argv[i + 1];
        }

        if (strcmp(arg, "--model") == 0 || strcmp(arg, "-m") == 0) {
            if (value == NULL) {
                return false;
            }
            opts->model = value;
            i++;
        } else if (strcmp(arg, "--distributed") == 0 || strcmp(arg, "--rpc") == 0) {
            if (value == NULL) {
                return false;
            }
            opts->servers = value;
            i++;
        } else if (strcmp(arg, "--prompt") == 0 || strcmp(arg, "-p") == 0) {
            if (value == NULL) {
                return false;
            }
            opts->prompt = value;
            i++;
        } else if (strcmp(arg, "--trace") == 0) {
            if (value == NULL) {
                return false;
            }
            opts->trace = value;
            i++;
        } else if (strcmp(arg, "--n-gpu-layers") == 0 || strcmp(arg, "-ngl") == 0) {
            if (value == NULL) {
                return false;
            }
            if (strcmp(value, "all") == 0) {
                opts->n_gpu_layers = -1;
            } else if (!parse_i32(value, &opts->n_gpu_layers)) {
                return false;
            }
            i++;
        } else if (strcmp(arg, "--n-predict") == 0 || strcmp(arg, "-n") == 0) {
            if (value == NULL || !parse_i32(value, &opts->n_predict)) {
                return false;
            }
            i++;
        } else if (strcmp(arg, "--ctx-size") == 0 || strcmp(arg, "-c") == 0) {
            if (value == NULL || !parse_u32(value, &opts->n_ctx)) {
                return false;
            }
            i++;
        } else if (strcmp(arg, "--batch-size") == 0 || strcmp(arg, "-b") == 0) {
            if (value == NULL || !parse_i32(value, &opts->n_batch)) {
                return false;
            }
            i++;
        } else if (strcmp(arg, "--trace-limit") == 0) {
            if (value == NULL || !parse_i32(value, &opts->trace_limit)) {
                return false;
            }
            i++;
        } else {
            fprintf(stderr, "[tap] error: unknown argument: %s\n", arg);
            return false;
        }
    }

    if (opts->model == NULL) {
        fprintf(stderr, "[tap] error: --model is required\n");
        return false;
    }
    if (opts->n_predict < 0) {
        fprintf(stderr, "[tap] error: --n-predict must be >= 0\n");
        return false;
    }
    if (opts->n_batch < 0) {
        fprintf(stderr, "[tap] error: --batch-size must be >= 0\n");
        return false;
    }

    return true;
}

static int tokenize_prompt(const struct llama_vocab * vocab, const char * prompt, llama_token ** out_tokens, int32_t * out_n_tokens) {
    const int32_t n_prompt = -llama_tokenize(vocab, prompt, (int32_t) strlen(prompt), NULL, 0, true, true);
    llama_token * tokens;
    int32_t n_tokens;

    if (n_prompt <= 0) {
        fprintf(stderr, "[tap] error: failed to count prompt tokens\n");
        return 1;
    }

    tokens = (llama_token *) malloc((size_t) n_prompt * sizeof(llama_token));
    if (tokens == NULL) {
        fprintf(stderr, "[tap] error: out of memory while tokenizing prompt\n");
        return 1;
    }

    n_tokens = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt), tokens, n_prompt, true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "[tap] error: failed to tokenize prompt\n");
        free(tokens);
        return 1;
    }

    *out_tokens = tokens;
    *out_n_tokens = n_tokens;
    return 0;
}

static int print_token_piece(const struct llama_vocab * vocab, llama_token token) {
    char piece[256];
    const int32_t n = llama_token_to_piece(vocab, token, piece, (int32_t) sizeof(piece), 0, true);

    if (n < 0) {
        fprintf(stderr, "[tap] error: failed to convert token to text\n");
        return 1;
    }

    fwrite(piece, 1, (size_t) n, stdout);
    fflush(stdout);
    return 0;
}

int main(int argc, char ** argv) {
    struct tap_options opts;
    struct tap_state state;
    struct llama_model_params mparams;
    struct llama_context_params cparams;
    struct llama_sampler_chain_params sparams;
    struct llama_model * model = NULL;
    struct llama_context * ctx = NULL;
    struct llama_sampler * sampler = NULL;
    const struct llama_vocab * vocab = NULL;
    llama_token * prompt_tokens = NULL;
    int32_t n_prompt = 0;
    int rc = 1;

    if (!parse_args(argc, argv, &opts)) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    if (register_rpc_servers(opts.servers) != 0) {
        return 1;
    }
    print_devices();

    llama_backend_init();

    mparams = llama_model_default_params();
    mparams.n_gpu_layers = opts.n_gpu_layers;
    mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;

    fprintf(stderr, "[tap] loading model %s\n", opts.model);
    model = llama_model_load_from_file(opts.model, mparams);
    if (model == NULL) {
        fprintf(stderr, "[tap] error: failed to load model\n");
        goto cleanup;
    }

    vocab = llama_model_get_vocab(model);
    if (tokenize_prompt(vocab, opts.prompt, &prompt_tokens, &n_prompt) != 0) {
        goto cleanup;
    }

    state.trace = opts.trace;
    state.trace_limit = opts.trace_limit;
    state.seen = 0;

    cparams = llama_context_default_params();
    cparams.n_ctx = opts.n_ctx == 0 ? (uint32_t) (n_prompt + (opts.n_predict > 0 ? opts.n_predict : 1)) : opts.n_ctx;
    cparams.n_batch = opts.n_batch == 0 ? (uint32_t) n_prompt : (uint32_t) opts.n_batch;
    cparams.cb_eval = tap_eval_cb;
    cparams.cb_eval_user_data = &state;
    cparams.no_perf = false;

    fprintf(stderr,
        "[tap] init context n_ctx=%u n_batch=%u trace=%s\n",
        cparams.n_ctx,
        cparams.n_batch,
        opts.trace);
    ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        fprintf(stderr, "[tap] error: failed to create context\n");
        goto cleanup;
    }

    sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    for (int32_t i = 0; i < n_prompt; ++i) {
        if (print_token_piece(vocab, prompt_tokens[i]) != 0) {
            goto cleanup;
        }
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt);
    for (int32_t n_generated = 0;;) {
        llama_token token;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[tap] error: llama_decode failed\n");
            goto cleanup;
        }

        if (n_generated >= opts.n_predict) {
            break;
        }

        token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        if (print_token_piece(vocab, token) != 0) {
            goto cleanup;
        }

        batch = llama_batch_get_one(&token, 1);
        n_generated++;
    }

    fprintf(stderr, "\n[tap] traced tensors: %d\n", state.seen);
    llama_perf_context_print(ctx);
    rc = 0;

cleanup:
    free(prompt_tokens);
    if (sampler != NULL) {
        llama_sampler_free(sampler);
    }
    if (ctx != NULL) {
        llama_free(ctx);
    }
    if (model != NULL) {
        llama_model_free(model);
    }
    llama_backend_free();

    return rc;
}
