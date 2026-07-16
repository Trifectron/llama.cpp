#pragma once

// Shared U-shaped layer-range execution core, extracted from split-tap.cpp's proven head/tail
// logic so it's usable both by cluster_node_service.cpp's gRPC PassOff handler (remote trunk
// nodes) and directly (no self-directed gRPC hop) by the origin node's own local head/tail
// computation - see plan file history: "gRPC Coordinator/ClusterNode Implementation".
//
// Same mechanism proven bit-exact in split-tap.cpp: il_start/il_end bounds the layer loop
// (src/models/llama.cpp), a head (il_end < n_layer) exposes its boundary hidden state via
// llama_get_embeddings_nextn(), and a non-head (il_start > 0) consumes an externally-supplied
// hidden state via the existing ubatch.embd input path instead of a token lookup.

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct cluster_hidden_state {
    std::vector<float>   data; // [n_tokens * n_embd], row-major by token
    int32_t               n_tokens = 0;
    int32_t               n_embd   = 0;
    std::vector<int32_t> positions; // sequence positions, size n_tokens - needed for correct RoPE
                                     // on the receiving node
};

struct cluster_tail_result {
    int32_t     token = -1;
    std::string piece;      // decoded text for `token`
    bool        is_eog = false;
};

// One instance = one prepared llama_context bounded to [il_start, il_end) for one in-flight
// request. Not thread-safe; callers (cluster_node_service.cpp's request_id-keyed map) provide
// their own external synchronization.
class cluster_layer_runner {
public:
    ~cluster_layer_runner();

    cluster_layer_runner(const cluster_layer_runner &) = delete;
    cluster_layer_runner & operator=(const cluster_layer_runner &) = delete;
    cluster_layer_runner() = default;

    // Loads model_path fresh (each runner owns an independent llama_model - simplest correct
    // thing for a prototype; a production version would want to share one loaded model across
    // requests). n_ctx must be large enough for n_tokens + a few generated tokens.
    bool prepare(const std::string & model_path, int32_t il_start, int32_t il_end, uint32_t n_ctx,
                 std::string & out_error);

    bool is_tail() const { return il_end_ >= n_layer_; }
    bool is_head() const { return il_start_ == 0; }
    int32_t n_layer() const { return n_layer_; }
    int32_t n_embd()  const { return n_embd_; }

    // Head only (il_start == 0): tokenizes prompt, decodes layers [0, il_end), returns the
    // boundary hidden state. If il_end already covers the whole model (single-node case), the
    // caller should use run_tail_from_prompt() instead - not handled here to keep this one
    // narrowly head-shaped, matching split-tap's own head/tail split.
    bool run_head(const std::string & prompt, cluster_hidden_state & out, std::string & out_error);

    // Head only, laps > 0: feeds a single already-sampled token at `position` through layers
    // [0, il_end), reusing the head's persisted KV cache from run_head()/prior steps, and returns
    // the 1-token boundary hidden state. This is the autoregressive step; run_head() bootstraps it
    // by processing the prompt into KV first.
    bool run_head_token(llama_token token, int32_t position, cluster_hidden_state & out, std::string & out_error);

    // Non-head, non-tail (0 < il_start, il_end < n_layer): feeds an incoming hidden state as this
    // context's layer-0 input, decodes [il_start, il_end), returns the new boundary hidden state.
    bool run_trunk(const cluster_hidden_state & in, cluster_hidden_state & out, std::string & out_error);

    // Non-head tail (il_start > 0, il_end == n_layer): feeds an incoming hidden state, decodes
    // through the output head, samples one token.
    bool run_tail(const cluster_hidden_state & in, cluster_tail_result & out, std::string & out_error);

private:
    llama_model         * model_   = nullptr;
    llama_context       * ctx_     = nullptr;
    const llama_vocab   * vocab_   = nullptr;
    llama_sampler       * sampler_ = nullptr;

    int32_t n_layer_  = 0;
    int32_t n_embd_   = 0;
    int32_t il_start_ = 0;
    int32_t il_end_   = 0;
};
