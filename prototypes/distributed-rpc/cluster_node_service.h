#pragma once

// ClusterNode service impl (see proto/cluster.proto): AssignLayers prepares a request_id-keyed
// cluster_layer_runner for a layer range; PassOff receives a streamed hidden state, runs it
// through the prepared runner, and either forwards the result onward via its own outbound
// PassOff client call (trunk) or completes the request locally (tail).
//
// U-shape note (see plan file history): the origin node's own head/tail segments are computed
// via a *direct* call (register_local_runner()), never a self-directed gRPC AssignLayers hop -
// only the middle "trunk" range is ever the target of a real AssignLayers RPC. The final PassOff
// from the last trunk node still arrives over the real network at the origin's own, genuinely
// listening PassOff handler, which is what completes the tail.

#include "cluster.grpc.pb.h"
#include "cluster_layer_runner.h"
#include "subprocess_runner.h"

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>

class cluster_node_service final : public llama_cluster::ClusterNode::Service {
public:
    grpc::Status AssignLayers(grpc::ServerContext * context, const llama_cluster::AssignLayersRequest * request,
                               llama_cluster::AssignLayersResponse * response) override;
    grpc::Status PassOff(grpc::ServerContext * context, grpc::ServerReader<llama_cluster::PassOffChunk> * reader,
                         llama_cluster::PassOffAck * response) override;
    grpc::Status DownloadModel(grpc::ServerContext * context, const llama_cluster::DownloadModelRequest * request,
                                llama_cluster::DownloadModelResponse * response) override;
    grpc::Status GetDownloadStatus(grpc::ServerContext * context, const llama_cluster::GetDownloadStatusRequest * request,
                                    llama_cluster::GetDownloadStatusResponse * response) override;

    // Path to the `llama` binary (its `download` subcommand, app/download.cpp) - must be set once
    // at startup, before any DownloadModel call, mirroring how cluster-ui.cpp resolves this same
    // sibling binary for its own HTTP /api/models/download handler.
    void set_download_binary_path(const std::string & path) { download_binary_path_ = path; }

    // Per-node download destination (see cluster-ui.cpp's CLUSTER_DOWNLOAD_DIR handling) - passed
    // as an LLAMA_CACHE override to the download subprocess's environment, never global env.
    void set_download_dir(const std::string & dir) { download_dir_ = dir; }

    // Registers a runner prepared directly (no gRPC hop) by the origin's own launch handler for
    // its local head/tail segments. next_node_endpoint: where a *trunk* runner should forward to
    // after computing (empty for a tail runner - it never forwards, it completes the request).
    void register_local_runner(const std::string & request_id, std::unique_ptr<cluster_layer_runner> runner,
                                const std::string & next_node_endpoint);

    // Blocks up to timeout_ms for PassOff to complete the tail runner registered under
    // request_id (whether that runner was registered via a remote AssignLayers or via
    // register_local_runner()). Used by the origin's launch handler to get one generated token.
    // Re-arms the request for the next generation lap on success (does NOT erase it), so the loop
    // can wait again after feeding the sampled token back through the head - call finish_request()
    // once generation is done.
    bool wait_for_tail_result(const std::string & request_id, int timeout_ms,
                               cluster_tail_result & out, std::string & out_error);

    // Drops the request (and its runner + KV cache) once the generation loop ends. Origin only,
    // for the local tail it registered. ponytail: trunk nodes' runners still leak per request
    // (pre-existing - no cleanup RPC yet); add a ClusterNode.Cleanup RPC if long-lived nodes OOM.
    void finish_request(const std::string & request_id);

private:
    struct pending_request {
        std::string request_id;
        std::unique_ptr<cluster_layer_runner> runner;
        std::string next_node_endpoint;

        std::mutex              result_mtx;
        std::condition_variable result_cv;
        bool                    done  = false;
        bool                    ok    = false;
        cluster_tail_result     tail_result;
        std::string             error;
    };

    std::mutex requests_mtx_;
    std::map<std::string, std::shared_ptr<pending_request>> requests_;

    std::string download_binary_path_;
    std::string download_dir_;
    std::mutex  downloads_mtx_;
    std::map<std::string, std::shared_ptr<run_state>> downloads_;
    uint64_t    next_download_id_ = 1;

    // Runs the prepared runner against an incoming hidden state (shared by PassOff's own
    // request-id lookup and any future direct-invocation path); trunk results are forwarded via
    // an outbound PassOff client call, tail results signal result_cv for wait_for_tail_result().
    void handle_hidden_state(const std::shared_ptr<pending_request> & req, const cluster_hidden_state & in);
};

// ---------------------------------------------------------------------------------------------
// Free functions - small gRPC client helpers, used both internally (forwarding a trunk's output
// onward) and by cluster-ui.cpp's /api/launch-grpc handler (Hello + AssignLayers to each trunk
// node, and the initial PassOff from the origin's own head computation).
// ---------------------------------------------------------------------------------------------

bool cluster_grpc_call_hello(const std::string & endpoint, const llama_cluster::HelloRequest & request,
                              llama_cluster::HelloResponse & response, std::string & out_error);

bool cluster_grpc_call_heartbeat(const std::string & endpoint, const llama_cluster::HeartbeatRequest & request,
                                  llama_cluster::HeartbeatResponse & response, std::string & out_error);

bool cluster_grpc_call_assign_layers(const std::string & endpoint, const llama_cluster::AssignLayersRequest & request,
                                      llama_cluster::AssignLayersResponse & response, std::string & out_error);

bool cluster_grpc_send_pass_off(const std::string & endpoint, const std::string & request_id,
                                 const cluster_hidden_state & hidden, std::string & out_error);

bool cluster_grpc_call_download_model(const std::string & endpoint, const llama_cluster::DownloadModelRequest & request,
                                       llama_cluster::DownloadModelResponse & response, std::string & out_error);

bool cluster_grpc_call_get_download_status(const std::string & endpoint, const llama_cluster::GetDownloadStatusRequest & request,
                                            llama_cluster::GetDownloadStatusResponse & response, std::string & out_error);
