#include "coordinator_service.h"

grpc::Status coordinator_service::Hello(grpc::ServerContext *, const llama_cluster::HelloRequest * request,
                                          llama_cluster::HelloResponse * response) {
    std::lock_guard<std::mutex> lock(mtx_);

    llama_cluster::ClusterMember member;
    member.set_node_id(request->node_id());
    member.set_rpc_endpoint(request->rpc_endpoint());
    member.set_cluster_node_endpoint(request->cluster_node_endpoint());
    *member.mutable_caps() = request->caps();

    member_state & state = members_[request->node_id()];
    state.member          = member;
    state.free_vram_bytes = request->caps().free_vram_bytes();

    response->set_accepted(true);
    for (const auto & [node_id, st] : members_) {
        *response->add_members() = st.member;
    }

    return grpc::Status::OK;
}

grpc::Status coordinator_service::Heartbeat(grpc::ServerContext *, const llama_cluster::HeartbeatRequest * request,
                                              llama_cluster::HeartbeatResponse * response) {
    std::lock_guard<std::mutex> lock(mtx_);

    member_state & state = members_[request->node_id()]; // implicit-Hello if unknown
    if (state.member.node_id().empty()) {
        state.member.set_node_id(request->node_id());
    }
    state.free_vram_bytes    = request->free_vram_bytes();
    state.busy               = request->busy();
    state.current_request_id = request->current_request_id();

    response->set_ack(true);
    return grpc::Status::OK;
}

grpc::Status coordinator_service::Leave(grpc::ServerContext *, const llama_cluster::LeaveRequest * request,
                                          llama_cluster::LeaveResponse * response) {
    std::lock_guard<std::mutex> lock(mtx_);
    members_.erase(request->node_id());
    response->set_ack(true);
    return grpc::Status::OK;
}

std::vector<llama_cluster::ClusterMember> coordinator_service::snapshot_members() {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<llama_cluster::ClusterMember> result;
    result.reserve(members_.size());
    for (const auto & [node_id, st] : members_) {
        result.push_back(st.member);
    }
    return result;
}
