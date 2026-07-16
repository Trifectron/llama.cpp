#include "cluster_layer_runner.h"

// llama_set/get_embeddings_nextn() live in the WIP staging header src/llama-ext.h (not the
// stable public llama.h) - same convention split-tap.cpp already established for this exact
// mechanism.
#include "../../src/llama-ext.h"

#include "ggml-backend.h"

#include <cstring>

cluster_layer_runner::~cluster_layer_runner() {
    if (sampler_) {
        llama_sampler_free(sampler_);
    }
    if (ctx_) {
        llama_free(ctx_);
    }
    if (model_) {
        llama_model_free(model_);
    }
}

bool cluster_layer_runner::prepare(const std::string & model_path, int32_t il_start, int32_t il_end,
                                    uint32_t n_ctx, std::string & out_error) {
    il_start_ = il_start;

    struct llama_model_params mparams = llama_model_default_params();
    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model_ == nullptr) {
        out_error = "failed to load model: " + model_path;
        return false;
    }

    vocab_  = llama_model_get_vocab(model_);
    n_embd_ = llama_model_n_embd(model_);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = n_ctx;
    cparams.n_batch  = n_ctx;
    cparams.n_ubatch = n_ctx;
    cparams.il_start = il_start;
    cparams.il_end   = il_end; // caller passes the model's real n_layer for a tail, not -1,
                                // since we need n_layer_ below regardless of tail-ness

    ctx_ = llama_init_from_model(model_, cparams);
    if (ctx_ == nullptr) {
        out_error = "failed to create context for il_start=" + std::to_string(il_start) +
                    " il_end=" + std::to_string(il_end);
        return false;
    }

    // il_end as actually clamped by llama_init_from_model (il_end<0 means "all layers" - not
    // used by this module, callers always pass an explicit end, but keep this robust anyway).
    n_layer_ = (int32_t) llama_model_n_layer(model_);
    il_end_  = il_end < 0 ? n_layer_ : il_end;

    if (!is_tail()) {
        // request the boundary hidden state for every position - the next hop needs all of
        // them to build its own KV cache/positions correctly, not just the last one.
        llama_set_embeddings_nextn(ctx_, true, false);
    }

    if (is_tail()) {
        struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        sampler_ = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());
    }

    return true;
}

bool cluster_layer_runner::run_head(const std::string & prompt, cluster_hidden_state & out, std::string & out_error) {
    if (!is_head()) {
        out_error = "run_head() called on a non-head runner";
        return false;
    }

    const int32_t n_prompt = -llama_tokenize(vocab_, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        out_error = "failed to count prompt tokens";
        return false;
    }

    std::vector<llama_token> tokens((size_t) n_prompt);
    const int32_t n_tokens = llama_tokenize(vocab_, prompt.c_str(), (int32_t) prompt.size(),
                                             tokens.data(), n_prompt, true, true);
    if (n_tokens < 0) {
        out_error = "failed to tokenize prompt";
        return false;
    }

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i]     = tokens[(size_t) i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }

    const int rc = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        out_error = "head llama_decode failed (rc=" + std::to_string(rc) + ")";
        return false;
    }

    const float * h = llama_get_embeddings_nextn(ctx_);
    if (h == nullptr) {
        out_error = "head produced no hidden state";
        return false;
    }

    out.n_tokens = n_tokens;
    out.n_embd   = n_embd_;
    out.data.assign(h, h + (size_t) n_tokens * (size_t) n_embd_);
    out.positions.resize((size_t) n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) {
        out.positions[(size_t) i] = i;
    }

    return true;
}

bool cluster_layer_runner::run_head_token(llama_token token, int32_t position, cluster_hidden_state & out, std::string & out_error) {
    if (!is_head()) {
        out_error = "run_head_token() called on a non-head runner";
        return false;
    }

    // Single-token batch at `position`. llama_decode appends this to the head's existing KV cache
    // (built by run_head() for the prompt, extended by prior steps) - it does not reset it, which
    // is exactly what makes this an O(1)-per-token autoregressive step instead of a reprocess.
    struct llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens     = 1;
    batch.token[0]     = token;
    batch.pos[0]       = position;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;

    const int rc = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        out_error = "head decode-step llama_decode failed (rc=" + std::to_string(rc) + ")";
        return false;
    }

    const float * h = llama_get_embeddings_nextn(ctx_);
    if (h == nullptr) {
        out_error = "head decode-step produced no hidden state";
        return false;
    }

    out.n_tokens = 1;
    out.n_embd   = n_embd_;
    out.data.assign(h, h + (size_t) n_embd_);
    out.positions = { position };

    return true;
}

// Shared by run_trunk()/run_tail(): builds an embd batch from an incoming hidden state using its
// own recorded positions (RoPE must see the true sequence positions, not a restart from 0), and
// decodes. Returns the batch's token count so callers can index into logits/embeddings.
// need_all_outputs: a trunk runner must produce a hidden state for *every* position (the next
// hop needs all of them), so every batch.logits[i] must be 1 - matching split-tap.cpp's own head
// batch construction. A tail runner only ever reads the final position's logits, so only the
// last token needs to be flagged; flagging all of them for a tail would just waste output-buffer
// space, not cause incorrectness, but keeping the two consistent with their actual buffer sizing
// avoids the "requested more rows than the output buffer was ever sized for" bug this masked.
static bool decode_hidden_state(llama_context * ctx, int32_t n_embd, const cluster_hidden_state & in,
                                 bool need_all_outputs, std::string & out_error) {
    if (in.n_embd != n_embd) {
        out_error = "hidden state n_embd mismatch: got " + std::to_string(in.n_embd) +
                    ", expected " + std::to_string(n_embd);
        return false;
    }
    if ((int32_t) in.positions.size() != in.n_tokens) {
        out_error = "hidden state positions size mismatch";
        return false;
    }

    struct llama_batch batch = llama_batch_init(in.n_tokens, n_embd, 1);
    batch.n_tokens = in.n_tokens;
    std::memcpy(batch.embd, in.data.data(), (size_t) in.n_tokens * (size_t) n_embd * sizeof(float));
    for (int32_t i = 0; i < in.n_tokens; ++i) {
        batch.pos[i]       = in.positions[(size_t) i];
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = need_all_outputs ? 1 : ((i == in.n_tokens - 1) ? 1 : 0);
    }

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        out_error = "llama_decode failed (rc=" + std::to_string(rc) + ")";
        return false;
    }

    return true;
}

bool cluster_layer_runner::run_trunk(const cluster_hidden_state & in, cluster_hidden_state & out, std::string & out_error) {
    if (is_head() || is_tail()) {
        out_error = "run_trunk() called on a head or tail runner";
        return false;
    }

    // (already armed once in prepare() for any non-tail runner - not re-called here)
    if (!decode_hidden_state(ctx_, n_embd_, in, /*need_all_outputs=*/true, out_error)) {
        return false;
    }

    const float * h = llama_get_embeddings_nextn(ctx_);
    if (h == nullptr) {
        out_error = "trunk produced no hidden state";
        return false;
    }

    out.n_tokens = in.n_tokens;
    out.n_embd   = n_embd_;
    out.data.assign(h, h + (size_t) in.n_tokens * (size_t) n_embd_);
    out.positions = in.positions;

    return true;
}

bool cluster_layer_runner::run_tail(const cluster_hidden_state & in, cluster_tail_result & out, std::string & out_error) {
    if (!is_tail()) {
        out_error = "run_tail() called on a non-tail runner";
        return false;
    }

    if (!decode_hidden_state(ctx_, n_embd_, in, /*need_all_outputs=*/false, out_error)) {
        return false;
    }

    const llama_token token = llama_sampler_sample(sampler_, ctx_, -1);

    char piece_buf[256];
    const int32_t n = llama_token_to_piece(vocab_, token, piece_buf, (int32_t) sizeof(piece_buf), 0, true);
    if (n < 0) {
        out_error = "failed to convert token to text";
        return false;
    }

    out.token  = token;
    out.piece  = std::string(piece_buf, (size_t) n);
    out.is_eog = llama_vocab_is_eog(vocab_, token);

    return true;
}
