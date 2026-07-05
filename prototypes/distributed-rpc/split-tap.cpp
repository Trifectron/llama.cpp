// Step A of the U-shaped activation hand-off prototype (see plan.md / README.md).
//
// Loads one model, then creates three llama_context views of it that share the same weights:
//   - ref:  the normal, unsplit model (il_start=0, il_end=n_layer) - ground truth.
//   - head: il_start=0, il_end=K - computes layers [0,K) and hands off the hidden state
//           at the boundary via llama_get_embeddings_nextn() instead of running norm/lm_head.
//   - tail: il_start=K, il_end=n_layer - consumes that hidden state as its layer-0 input
//           (via the existing ubatch->embd path, same mechanism llama.cpp already uses for
//           multimodal/mtmd embedding input) and runs the rest of the model + lm_head.
//
// This is a correctness proof, not a memory-saving demo: all three contexts are built from the
// same fully-loaded model, so nothing is skipped at the tensor level yet - only at the graph
// level. It proves the split graph reproduces the unsplit graph's logits before any process- or
// network-boundary is introduced (that's split-tap's two-process mode, added in a later step).
//
// llama_set/get_embeddings_nextn() live in the WIP staging header src/llama-ext.h (not the
// stable public llama.h) - same header common/speculative.cpp and tools/fit-params already
// include from outside src/, so this follows an existing convention rather than a new one.

#include "../../src/llama-ext.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct tap_options {
    const char * model;
    const char * prompt;
    int32_t      split_layer;
    const char * head_rpc; // optional: host:port of a ggml-rpc-server to host the head graph on
    const char * tail_rpc; // optional: host:port of a ggml-rpc-server to host the tail graph on
};

static void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s --model MODEL --split-layer K [--prompt TEXT] [--head-rpc HOST:PORT] [--tail-rpc HOST:PORT]\n"
        "\n"
        "options:\n"
        "  --model MODEL        path to a GGUF model (llama architecture)\n"
        "  --split-layer K      layer index to hand off the hidden state at (0 < K < n_layer)\n"
        "  --prompt TEXT        prompt to decode, default a short built-in sentence\n"
        "  --head-rpc HOST:PORT run the head graph (layers [0,K)) on a remote ggml-rpc-server\n"
        "                       instead of the local CPU - proves the hand-off survives a real\n"
        "                       process/network boundary using the existing RPC SET_TENSOR/\n"
        "                       GET_TENSOR primitives (no protocol changes)\n"
        "  --tail-rpc HOST:PORT run the tail graph (layers [K,n_layer)) on a remote ggml-rpc-server\n"
        "  -h, --help           show this help\n",
        prog);
}

static bool parse_args(int argc, char ** argv, struct tap_options * opts) {
    opts->model       = NULL;
    opts->prompt      = "The distributed cluster splits a model across";
    opts->split_layer = -1;
    opts->head_rpc    = NULL;
    opts->tail_rpc    = NULL;

    for (int i = 1; i < argc; ++i) {
        const char * arg   = argv[i];
        const char * value = (i + 1 < argc) ? argv[i + 1] : NULL;

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--model") == 0 || strcmp(arg, "-m") == 0) {
            if (!value) return false;
            opts->model = value;
            i++;
        } else if (strcmp(arg, "--prompt") == 0 || strcmp(arg, "-p") == 0) {
            if (!value) return false;
            opts->prompt = value;
            i++;
        } else if (strcmp(arg, "--split-layer") == 0) {
            if (!value) return false;
            opts->split_layer = (int32_t) atoi(value);
            i++;
        } else if (strcmp(arg, "--head-rpc") == 0) {
            if (!value) return false;
            opts->head_rpc = value;
            i++;
        } else if (strcmp(arg, "--tail-rpc") == 0) {
            if (!value) return false;
            opts->tail_rpc = value;
            i++;
        } else {
            fprintf(stderr, "[split-tap] error: unknown argument: %s\n", arg);
            return false;
        }
    }

    if (opts->model == NULL) {
        fprintf(stderr, "[split-tap] error: --model is required\n");
        return false;
    }
    if (opts->split_layer <= 0) {
        fprintf(stderr, "[split-tap] error: --split-layer is required and must be > 0\n");
        return false;
    }

    return true;
}

// Registers a single ggml-rpc-server endpoint and returns its first device. Each call to
// ggml_backend_rpc_add_server() creates a fresh backend_reg scoped to just that one endpoint
// (same pattern distributed-tap.c uses), so device index 0 is always "the device this endpoint
// exposes" rather than an index into some global device list.
static ggml_backend_dev_t connect_rpc_device(const char * endpoint) {
    typedef ggml_backend_reg_t (*add_rpc_server_fn)(const char * endpoint);

    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (rpc_reg == NULL) {
        fprintf(stderr, "[split-tap] error: RPC backend not found; rebuild with -DGGML_RPC=ON\n");
        return NULL;
    }

    auto add_server = (add_rpc_server_fn) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (add_server == NULL) {
        fprintf(stderr, "[split-tap] error: RPC add-server function not found\n");
        return NULL;
    }

    ggml_backend_reg_t server_reg = add_server(endpoint);
    if (server_reg == NULL) {
        fprintf(stderr, "[split-tap] error: failed to add RPC server %s\n", endpoint);
        return NULL;
    }
    ggml_backend_register(server_reg);

    if (ggml_backend_reg_dev_count(server_reg) == 0) {
        fprintf(stderr, "[split-tap] error: RPC server %s exposes no devices\n", endpoint);
        return NULL;
    }

    fprintf(stderr, "[split-tap] connected to RPC server %s\n", endpoint);
    return ggml_backend_reg_dev_get(server_reg, 0);
}

static int tokenize_prompt(const struct llama_vocab * vocab, const char * prompt, llama_token ** out_tokens, int32_t * out_n_tokens) {
    const int32_t n_prompt = -llama_tokenize(vocab, prompt, (int32_t) strlen(prompt), NULL, 0, true, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "[split-tap] error: failed to count prompt tokens\n");
        return 1;
    }

    llama_token * tokens = (llama_token *) malloc((size_t) n_prompt * sizeof(llama_token));
    if (tokens == NULL) {
        fprintf(stderr, "[split-tap] error: out of memory while tokenizing prompt\n");
        return 1;
    }

    const int32_t n_tokens = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt), tokens, n_prompt, true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "[split-tap] error: failed to tokenize prompt\n");
        free(tokens);
        return 1;
    }

    *out_tokens   = tokens;
    *out_n_tokens = n_tokens;
    return 0;
}

static int32_t argmax(const float * v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; ++i) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    struct tap_options opts;
    if (!parse_args(argc, argv, &opts)) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();

    // reference model: always local, ground truth for the comparison at the end.
    struct llama_model_params mparams_ref = llama_model_default_params();
    struct llama_model * model_ref = llama_model_load_from_file(opts.model, mparams_ref);
    if (model_ref == NULL) {
        fprintf(stderr, "[split-tap] error: failed to load reference model %s\n", opts.model);
        return 1;
    }

    // head/tail models: local by default (Step A, single process), or hosted on a remote
    // ggml-rpc-server when --head-rpc/--tail-rpc is given (Step B, two processes). Restricting
    // llama_model_params.devices to exactly one device keeps the whole truncated graph on that
    // one device rather than letting it spill onto other local devices too.
    ggml_backend_dev_t   dev_head = NULL;
    ggml_backend_dev_t   dev_tail = NULL;
    ggml_backend_dev_t   devs_head[2] = { NULL, NULL };
    ggml_backend_dev_t   devs_tail[2] = { NULL, NULL };

    struct llama_model_params mparams_head = llama_model_default_params();
    if (opts.head_rpc != NULL) {
        dev_head = connect_rpc_device(opts.head_rpc);
        if (dev_head == NULL) {
            return 1;
        }
        devs_head[0] = dev_head;
        mparams_head.devices = devs_head;
    }

    struct llama_model_params mparams_tail = llama_model_default_params();
    if (opts.tail_rpc != NULL) {
        dev_tail = connect_rpc_device(opts.tail_rpc);
        if (dev_tail == NULL) {
            return 1;
        }
        devs_tail[0] = dev_tail;
        mparams_tail.devices = devs_tail;
    }

    struct llama_model * model_head = llama_model_load_from_file(opts.model, mparams_head);
    if (model_head == NULL) {
        fprintf(stderr, "[split-tap] error: failed to load head model %s\n", opts.model);
        return 1;
    }

    struct llama_model * model_tail = llama_model_load_from_file(opts.model, mparams_tail);
    if (model_tail == NULL) {
        fprintf(stderr, "[split-tap] error: failed to load tail model %s\n", opts.model);
        return 1;
    }

    const struct llama_vocab * vocab   = llama_model_get_vocab(model_ref);
    const int32_t              n_embd  = llama_model_n_embd(model_ref);
    const int32_t              n_vocab = llama_vocab_n_tokens(vocab);

    llama_token * tokens   = NULL;
    int32_t       n_tokens = 0;
    if (tokenize_prompt(vocab, opts.prompt, &tokens, &n_tokens) != 0) {
        llama_model_free(model_ref);
        llama_model_free(model_head);
        llama_model_free(model_tail);
        return 1;
    }

    const uint32_t n_ctx_needed = (uint32_t) n_tokens + 8;

    fprintf(stderr, "[split-tap] model=%s n_tokens=%d n_embd=%d n_vocab=%d split_layer=%d\n",
            opts.model, n_tokens, n_embd, n_vocab, opts.split_layer);

    // ---- reference context: normal, unsplit decode ----
    struct llama_context_params cparams_ref = llama_context_default_params();
    cparams_ref.n_ctx    = n_ctx_needed;
    cparams_ref.n_batch  = (uint32_t) n_tokens;
    cparams_ref.n_ubatch = (uint32_t) n_tokens;

    struct llama_context * ctx_ref = llama_init_from_model(model_ref, cparams_ref);
    if (ctx_ref == NULL) {
        fprintf(stderr, "[split-tap] error: failed to create reference context\n");
        return 1;
    }

    llama_token * tokens_ref = (llama_token *) malloc((size_t) n_tokens * sizeof(llama_token));
    memcpy(tokens_ref, tokens, (size_t) n_tokens * sizeof(llama_token));

    struct llama_batch batch_ref = llama_batch_get_one(tokens_ref, n_tokens);
    if (llama_decode(ctx_ref, batch_ref) != 0) {
        fprintf(stderr, "[split-tap] error: reference llama_decode failed\n");
        return 1;
    }

    const float * logits_ref_live = llama_get_logits_ith(ctx_ref, -1);
    if (logits_ref_live == NULL) {
        fprintf(stderr, "[split-tap] error: failed to get reference logits\n");
        return 1;
    }
    float * logits_ref = (float *) malloc((size_t) n_vocab * sizeof(float));
    memcpy(logits_ref, logits_ref_live, (size_t) n_vocab * sizeof(float));

    // ---- head context: layers [0, split_layer) only ----
    struct llama_context_params cparams_head = llama_context_default_params();
    cparams_head.n_ctx    = n_ctx_needed;
    cparams_head.n_batch  = (uint32_t) n_tokens;
    cparams_head.n_ubatch = (uint32_t) n_tokens;
    cparams_head.il_start = 0;
    cparams_head.il_end   = opts.split_layer;

    struct llama_context * ctx_head = llama_init_from_model(model_head, cparams_head);
    if (ctx_head == NULL) {
        fprintf(stderr, "[split-tap] error: failed to create head context (is --split-layer < n_layer?)\n");
        return 1;
    }

    // request the boundary hidden state for every position, not just the last one - the tail
    // needs every position's hidden state to build its own KV cache correctly.
    llama_set_embeddings_nextn(ctx_head, true, false);

    struct llama_batch batch_head = llama_batch_init(n_tokens, 0, 1);
    batch_head.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch_head.token[i]     = tokens[i];
        batch_head.pos[i]       = i;
        batch_head.n_seq_id[i]  = 1;
        batch_head.seq_id[i][0] = 0;
        batch_head.logits[i]    = 1;
    }

    if (llama_decode(ctx_head, batch_head) != 0) {
        fprintf(stderr, "[split-tap] error: head llama_decode failed\n");
        return 1;
    }

    const float * h_live = llama_get_embeddings_nextn(ctx_head);
    if (h_live == NULL) {
        fprintf(stderr, "[split-tap] error: head context produced no hidden state "
                        "(embeddings_nextn not populated - is --split-layer < n_layer?)\n");
        return 1;
    }
    float * h_boundary = (float *) malloc((size_t) n_tokens * n_embd * sizeof(float));
    memcpy(h_boundary, h_live, (size_t) n_tokens * n_embd * sizeof(float));

    // ---- tail context: layers [split_layer, n_layer) + lm_head ----
    struct llama_context_params cparams_tail = llama_context_default_params();
    cparams_tail.n_ctx    = n_ctx_needed;
    cparams_tail.n_batch  = (uint32_t) n_tokens;
    cparams_tail.n_ubatch = (uint32_t) n_tokens;
    cparams_tail.il_start = opts.split_layer;
    cparams_tail.il_end   = -1; // all remaining layers

    struct llama_context * ctx_tail = llama_init_from_model(model_tail, cparams_tail);
    if (ctx_tail == NULL) {
        fprintf(stderr, "[split-tap] error: failed to create tail context\n");
        return 1;
    }

    // embd batch: feed the captured hidden state instead of token ids, same positions as the
    // original prompt (RoPE must see the true sequence positions, not a restart from 0).
    struct llama_batch batch_tail = llama_batch_init(n_tokens, n_embd, 1);
    batch_tail.n_tokens = n_tokens;
    memcpy(batch_tail.embd, h_boundary, (size_t) n_tokens * n_embd * sizeof(float));
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch_tail.pos[i]       = i;
        batch_tail.n_seq_id[i]  = 1;
        batch_tail.seq_id[i][0] = 0;
        batch_tail.logits[i]    = (i == n_tokens - 1) ? 1 : 0;
    }

    if (llama_decode(ctx_tail, batch_tail) != 0) {
        fprintf(stderr, "[split-tap] error: tail llama_decode failed\n");
        return 1;
    }

    const float * logits_tail = llama_get_logits_ith(ctx_tail, -1);
    if (logits_tail == NULL) {
        fprintf(stderr, "[split-tap] error: failed to get tail logits\n");
        return 1;
    }

    // ---- compare ----
    float max_abs_diff = 0.0f;
    for (int32_t i = 0; i < n_vocab; ++i) {
        const float diff = fabsf(logits_ref[i] - logits_tail[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    const int32_t tok_ref  = argmax(logits_ref, n_vocab);
    const int32_t tok_tail = argmax(logits_tail, n_vocab);

    char piece_ref[256];
    char piece_tail[256];
    llama_token_to_piece(vocab, tok_ref,  piece_ref,  (int32_t) sizeof(piece_ref),  0, true);
    llama_token_to_piece(vocab, tok_tail, piece_tail, (int32_t) sizeof(piece_tail), 0, true);

    fprintf(stderr, "[split-tap] ref  argmax token = %d (%s)\n", tok_ref, piece_ref);
    fprintf(stderr, "[split-tap] tail argmax token = %d (%s)\n", tok_tail, piece_tail);
    fprintf(stderr, "[split-tap] max abs logit diff = %g\n", (double) max_abs_diff);

    const bool pass = (tok_ref == tok_tail) && (max_abs_diff < 1e-3f);
    fprintf(stderr, "[split-tap] %s\n", pass ? "PASS" : "FAIL");

    free(logits_ref);
    free(h_boundary);
    free(tokens_ref);
    free(tokens);
    llama_batch_free(batch_head);
    llama_batch_free(batch_tail);
    llama_free(ctx_ref);
    llama_free(ctx_head);
    llama_free(ctx_tail);
    llama_model_free(model_ref);
    llama_model_free(model_head);
    llama_model_free(model_tail);
    llama_backend_free();

    return pass ? 0 : 1;
}
