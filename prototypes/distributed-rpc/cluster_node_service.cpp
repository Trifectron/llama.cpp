#include "cluster_node_service.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstring>

// ---------------------------------------------------------------------------------------------
// cluster_node_service - server side
// ---------------------------------------------------------------------------------------------

grpc::Status cluster_node_service::AssignLayers(grpc::ServerContext *, const llama_cluster::AssignLayersRequest * request,
                                                  llama_cluster::AssignLayersResponse * response) {
    auto runner = std::make_unique<cluster_layer_runner>();
    std::string error;

    // n_ctx: fixed generous default for this phase (one forward pass, no long-running generation
    // yet) - not part of the wire protocol, just an implementation-level sizing choice.
    const uint32_t n_ctx = 4096;

    if (!runner->prepare(request->model_path(), (int32_t) request->layer_start(), (int32_t) request->layer_end(),
                          n_ctx, error)) {
        response->set_accepted(false);
        response->set_reject_reason(error);
        return grpc::Status::OK;
    }

    {
        std::lock_guard<std::mutex> lock(requests_mtx_);
        auto req = std::make_shared<pending_request>();
        req->request_id         = request->request_id();
        req->runner              = std::move(runner);
        req->next_node_endpoint  = request->next_node_endpoint();
        requests_[request->request_id()] = req;
    }

    response->set_accepted(true);
    return grpc::Status::OK;
}

grpc::Status cluster_node_service::PassOff(grpc::ServerContext *, grpc::ServerReader<llama_cluster::PassOffChunk> * reader,
                                            llama_cluster::PassOffAck * response) {
    llama_cluster::PassOffChunk chunk;
    llama_cluster::TensorMeta   meta;
    bool        got_meta = false;
    std::string raw_bytes;

    while (reader->Read(&chunk)) {
        if (chunk.has_meta()) {
            meta     = chunk.meta();
            got_meta = true;
        } else if (chunk.has_data()) {
            raw_bytes += chunk.data();
        }
    }

    if (!got_meta) {
        response->set_ok(false);
        response->set_error("no TensorMeta received");
        return grpc::Status::OK;
    }

    cluster_hidden_state in;
    in.n_tokens = (int32_t) meta.n_tokens();
    in.n_embd   = (int32_t) meta.n_embd();
    in.positions.assign(meta.positions().begin(), meta.positions().end());

    const size_t expected_bytes = (size_t) in.n_tokens * (size_t) in.n_embd * sizeof(float);
    if (raw_bytes.size() != expected_bytes) {
        response->set_ok(false);
        response->set_error("hidden state byte size mismatch: got " + std::to_string(raw_bytes.size()) +
                             ", expected " + std::to_string(expected_bytes));
        return grpc::Status::OK;
    }
    in.data.resize((size_t) in.n_tokens * (size_t) in.n_embd);
    std::memcpy(in.data.data(), raw_bytes.data(), expected_bytes);

    std::shared_ptr<pending_request> req;
    {
        std::lock_guard<std::mutex> lock(requests_mtx_);
        auto it = requests_.find(meta.request_id());
        if (it == requests_.end()) {
            response->set_ok(false);
            response->set_error("unknown request_id - no prior AssignLayers for this request: " + meta.request_id());
            return grpc::Status::OK;
        }
        req = it->second;
    }

    handle_hidden_state(req, in);

    response->set_ok(true);
    return grpc::Status::OK;
}

grpc::Status cluster_node_service::DownloadModel(grpc::ServerContext *, const llama_cluster::DownloadModelRequest * request,
                                                   llama_cluster::DownloadModelResponse * response) {
    if (request->hf_repo().empty()) {
        response->set_accepted(false);
        response->set_reject_reason("hf_repo is required");
        return grpc::Status::OK;
    }
    if (download_binary_path_.empty()) {
        response->set_accepted(false);
        response->set_reject_reason("this node has no download binary path configured");
        return grpc::Status::OK;
    }

    std::vector<std::string> argv_strs = {
        download_binary_path_,
        "download",
        "-hf", request->hf_repo(),
    };
    if (!request->hf_file().empty()) {
        argv_strs.push_back("-hff");
        argv_strs.push_back(request->hf_file());
    }

    auto state = launch_process(argv_strs);

    std::string download_id;
    {
        std::lock_guard<std::mutex> lock(downloads_mtx_);
        download_id = "dl-" + std::to_string(next_download_id_++);
        downloads_[download_id] = state;
    }

    response->set_accepted(true);
    response->set_download_id(download_id);
    return grpc::Status::OK;
}

grpc::Status cluster_node_service::GetDownloadStatus(grpc::ServerContext *, const llama_cluster::GetDownloadStatusRequest * request,
                                                       llama_cluster::GetDownloadStatusResponse * response) {
    std::shared_ptr<run_state> state;
    {
        std::lock_guard<std::mutex> lock(downloads_mtx_);
        auto it = downloads_.find(request->download_id());
        if (it == downloads_.end()) {
            response->set_state(llama_cluster::DOWNLOAD_STATE_UNKNOWN);
            response->set_error("unknown download_id: " + request->download_id());
            return grpc::Status::OK;
        }
        state = it->second;
    }

    std::lock_guard<std::mutex> lock(state->mtx);

    // trailing slice only - the full subprocess output isn't meant to be polled repeatedly in
    // full, just enough to show recent progress (mirrors the HTTP /api/runs/{id} contract,
    // which does return the whole buffer since browsers poll it far less frequently than a
    // tight gRPC status-check loop might).
    constexpr size_t kTailBytes = 2048;
    const std::string tail = state->output.size() > kTailBytes
        ? state->output.substr(state->output.size() - kTailBytes)
        : state->output;
    response->set_output_tail(tail);

    if (state->running) {
        response->set_state(llama_cluster::DOWNLOAD_STATE_RUNNING);
        return grpc::Status::OK;
    }

    if (state->exit_code != 0) {
        response->set_state(llama_cluster::DOWNLOAD_STATE_FAILED);
        response->set_error("download exited with code " + std::to_string(state->exit_code));
        return grpc::Status::OK;
    }

    // `llama download` prints the resolved local path(s) as the last line(s) of stdout on
    // success - same convention the HTTP /api/models/download handler already relies on.
    const std::string & output = state->output;
    size_t end = output.find_last_not_of("\r\n");
    if (end == std::string::npos) {
        response->set_state(llama_cluster::DOWNLOAD_STATE_FAILED);
        response->set_error("download finished but produced no output path");
        return grpc::Status::OK;
    }
    size_t start = output.find_last_of('\n', end);
    start = (start == std::string::npos) ? 0 : start + 1;
    const std::string model_path = output.substr(start, end - start + 1);

    if (model_path.empty()) {
        response->set_state(llama_cluster::DOWNLOAD_STATE_FAILED);
        response->set_error("download finished but produced no output path");
        return grpc::Status::OK;
    }

    response->set_state(llama_cluster::DOWNLOAD_STATE_DONE);
    response->set_model_path(model_path);
    return grpc::Status::OK;
}

void cluster_node_service::register_local_runner(const std::string & request_id,
                                                   std::unique_ptr<cluster_layer_runner> runner,
                                                   const std::string & next_node_endpoint) {
    std::lock_guard<std::mutex> lock(requests_mtx_);
    auto req = std::make_shared<pending_request>();
    req->request_id        = request_id;
    req->runner             = std::move(runner);
    req->next_node_endpoint = next_node_endpoint;
    requests_[request_id]   = req;
}

bool cluster_node_service::wait_for_tail_result(const std::string & request_id, int timeout_ms,
                                                  cluster_tail_result & out, std::string & out_error) {
    std::shared_ptr<pending_request> req;
    {
        std::lock_guard<std::mutex> lock(requests_mtx_);
        auto it = requests_.find(request_id);
        if (it == requests_.end()) {
            out_error = "unknown request_id: " + request_id;
            return false;
        }
        req = it->second;
    }

    std::unique_lock<std::mutex> lock(req->result_mtx);
    const bool signaled = req->result_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                                    [&] { return req->done; });

    {
        std::lock_guard<std::mutex> cleanup_lock(requests_mtx_);
        requests_.erase(request_id);
    }

    if (!signaled) {
        out_error = "timed out waiting for tail result";
        return false;
    }
    if (!req->ok) {
        out_error = req->error;
        return false;
    }

    out = req->tail_result;
    return true;
}

void cluster_node_service::handle_hidden_state(const std::shared_ptr<pending_request> & req, const cluster_hidden_state & in) {
    std::string error;

    if (req->runner->is_tail()) {
        cluster_tail_result result;
        const bool ok = req->runner->run_tail(in, result, error);

        std::lock_guard<std::mutex> lock(req->result_mtx);
        req->ok          = ok;
        req->tail_result = result;
        req->error       = error;
        req->done        = true;
        req->result_cv.notify_all();
        return;
    }

    cluster_hidden_state out;
    if (!req->runner->run_trunk(in, out, error)) {
        std::lock_guard<std::mutex> lock(req->result_mtx);
        req->ok    = false;
        req->error = error;
        req->done  = true;
        req->result_cv.notify_all();
        return;
    }

    if (req->next_node_endpoint.empty()) {
        std::lock_guard<std::mutex> lock(req->result_mtx);
        req->ok    = false;
        req->error = "trunk runner has no next_node_endpoint to forward to";
        req->done  = true;
        req->result_cv.notify_all();
        return;
    }

    std::string send_error;
    if (!cluster_grpc_send_pass_off(req->next_node_endpoint, req->request_id, out, send_error)) {
        std::lock_guard<std::mutex> lock(req->result_mtx);
        req->ok    = false;
        req->error = "failed to forward PassOff to " + req->next_node_endpoint + ": " + send_error;
        req->done  = true;
        req->result_cv.notify_all();
    }
    // on success, this trunk node's own job is done - the eventual result (or failure) will
    // surface at whichever node's wait_for_tail_result() is actually waiting on it (the origin).
}

// ---------------------------------------------------------------------------------------------
// gRPC client helpers
// ---------------------------------------------------------------------------------------------

static std::shared_ptr<grpc::Channel> make_channel(const std::string & endpoint) {
    return grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials());
}

bool cluster_grpc_call_hello(const std::string & endpoint, const llama_cluster::HelloRequest & request,
                              llama_cluster::HelloResponse & response, std::string & out_error) {
    auto stub = llama_cluster::Coordinator::NewStub(make_channel(endpoint));
    grpc::ClientContext context;
    const grpc::Status status = stub->Hello(&context, request, &response);
    if (!status.ok()) {
        out_error = status.error_message();
        return false;
    }
    return true;
}

bool cluster_grpc_call_heartbeat(const std::string & endpoint, const llama_cluster::HeartbeatRequest & request,
                                  llama_cluster::HeartbeatResponse & response, std::string & out_error) {
    auto stub = llama_cluster::Coordinator::NewStub(make_channel(endpoint));
    grpc::ClientContext context;
    const grpc::Status status = stub->Heartbeat(&context, request, &response);
    if (!status.ok()) {
        out_error = status.error_message();
        return false;
    }
    return true;
}

bool cluster_grpc_call_download_model(const std::string & endpoint, const llama_cluster::DownloadModelRequest & request,
                                       llama_cluster::DownloadModelResponse & response, std::string & out_error) {
    auto stub = llama_cluster::ClusterNode::NewStub(make_channel(endpoint));
    grpc::ClientContext context;
    const grpc::Status status = stub->DownloadModel(&context, request, &response);
    if (!status.ok()) {
        out_error = status.error_message();
        return false;
    }
    return true;
}

bool cluster_grpc_call_get_download_status(const std::string & endpoint, const llama_cluster::GetDownloadStatusRequest & request,
                                            llama_cluster::GetDownloadStatusResponse & response, std::string & out_error) {
    auto stub = llama_cluster::ClusterNode::NewStub(make_channel(endpoint));
    grpc::ClientContext context;
    const grpc::Status status = stub->GetDownloadStatus(&context, request, &response);
    if (!status.ok()) {
        out_error = status.error_message();
        return false;
    }
    return true;
}

bool cluster_grpc_call_assign_layers(const std::string & endpoint, const llama_cluster::AssignLayersRequest & request,
                                      llama_cluster::AssignLayersResponse & response, std::string & out_error) {
    auto stub = llama_cluster::ClusterNode::NewStub(make_channel(endpoint));
    grpc::ClientContext context;
    const grpc::Status status = stub->AssignLayers(&context, request, &response);
    if (!status.ok()) {
        out_error = status.error_message();
        return false;
    }
    return true;
}

bool cluster_grpc_send_pass_off(const std::string & endpoint, const std::string & request_id,
                                 const cluster_hidden_state & hidden, std::string & out_error) {
    auto stub = llama_cluster::ClusterNode::NewStub(make_channel(endpoint));

    grpc::ClientContext        context;
    llama_cluster::PassOffAck  ack;
    std::unique_ptr<grpc::ClientWriter<llama_cluster::PassOffChunk>> writer(stub->PassOff(&context, &ack));

    llama_cluster::PassOffChunk meta_chunk;
    llama_cluster::TensorMeta *  meta = meta_chunk.mutable_meta();
    meta->set_request_id(request_id);
    meta->set_n_tokens((uint32_t) hidden.n_tokens);
    meta->set_n_embd((uint32_t) hidden.n_embd);
    meta->set_dtype("f32");
    for (int32_t pos : hidden.positions) {
        meta->add_positions(pos);
    }
    if (!writer->Write(meta_chunk)) {
        out_error = "failed to write TensorMeta chunk";
        return false;
    }

    llama_cluster::PassOffChunk data_chunk;
    data_chunk.set_data(hidden.data.data(), hidden.data.size() * sizeof(float));
    if (!writer->Write(data_chunk)) {
        out_error = "failed to write hidden-state data chunk";
        return false;
    }

    writer->WritesDone();
    const grpc::Status status = writer->Finish();
    if (!status.ok()) {
        out_error = status.error_message();
        return false;
    }
    if (!ack.ok()) {
        out_error = ack.error();
        return false;
    }

    return true;
}
