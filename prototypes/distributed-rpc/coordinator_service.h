#pragma once

// Coordinator service impl (see proto/cluster.proto): in-memory cluster membership - Hello
// registers a node and returns the current member list, Heartbeat updates liveness/status, Leave
// removes a node immediately. Every llama-cluster-ui instance runs this (symmetric "always both"
// design - see cluster-ui.cpp's file header), so any node can act as the coordinator for a
// session it happens to be driving.
//
// [phase scope] Heartbeat genuinely updates liveness state; acting on a missed heartbeat
// (reassigning layers mid-flight) is not implemented yet - see plan file history.

#include "cluster.grpc.pb.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

class coordinator_service final : public llama_cluster::Coordinator::Service {
public:
    grpc::Status Hello(grpc::ServerContext * context, const llama_cluster::HelloRequest * request,
                        llama_cluster::HelloResponse * response) override;
    grpc::Status Heartbeat(grpc::ServerContext * context, const llama_cluster::HeartbeatRequest * request,
                           llama_cluster::HeartbeatResponse * response) override;
    grpc::Status Leave(grpc::ServerContext * context, const llama_cluster::LeaveRequest * request,
                       llama_cluster::LeaveResponse * response) override;

    // For local inspection/debugging - returns a snapshot of the current member table.
    std::vector<llama_cluster::ClusterMember> snapshot_members();

private:
    struct member_state {
        llama_cluster::ClusterMember member;
        uint64_t    free_vram_bytes = 0;
        bool        busy = false;
        std::string current_request_id;
    };

    std::mutex mtx_;
    std::map<std::string, member_state> members_; // keyed by node_id
};
