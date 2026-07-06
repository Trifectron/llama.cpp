find_package(Threads REQUIRED)
# gRPC's own CMake package config already internally does find_dependency(Protobuf) - calling
# find_package(Protobuf) ourselves first causes a conflicting duplicate-targets error against
# that transitive call, so only find gRPC directly; protobuf::* targets and the
# protobuf_generate() function both come along for free via gRPCConfig.cmake's own dependency.
find_package(gRPC REQUIRED CONFIG)

# Cluster control-plane protocol (Coordinator/ClusterNode services, see proto/cluster.proto) -
# gRPC is a build-time library dependency only, exactly like cpp-httplib already is. There is no
# separate running process: every RPC server/client lives inside llama-cluster-ui itself.
set(CLUSTER_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(CLUSTER_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen)
file(MAKE_DIRECTORY ${CLUSTER_PROTO_GEN_DIR})

protobuf_generate(
    LANGUAGE cpp
    PROTOS ${CLUSTER_PROTO_DIR}/cluster.proto
    IMPORT_DIRS ${CLUSTER_PROTO_DIR}
    PROTOC_OUT_DIR ${CLUSTER_PROTO_GEN_DIR}
    OUT_VAR CLUSTER_PROTO_CPP_SRCS
)
protobuf_generate(
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
    PROTOS ${CLUSTER_PROTO_DIR}/cluster.proto
    IMPORT_DIRS ${CLUSTER_PROTO_DIR}
    PROTOC_OUT_DIR ${CLUSTER_PROTO_GEN_DIR}
    OUT_VAR CLUSTER_PROTO_GRPC_SRCS
)

set(TARGET llama-cluster-proto)
add_library(${TARGET} STATIC ${CLUSTER_PROTO_CPP_SRCS} ${CLUSTER_PROTO_GRPC_SRCS})
target_include_directories(${TARGET} PUBLIC ${CLUSTER_PROTO_GEN_DIR})
target_link_libraries(${TARGET} PUBLIC protobuf::libprotobuf gRPC::grpc++)
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # suppress warnings in generated code, not ours to fix
    target_compile_options(${TARGET} PRIVATE -w)
endif()

set(TARGET llama-distributed-tap)
add_executable(${TARGET} distributed-tap.c)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET ${TARGET} PROPERTY C_STANDARD 11)

set(TARGET llama-split-tap)
add_executable(${TARGET} split-tap.cpp)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)

set(TARGET llama-tailscale-discover)
add_executable(${TARGET} tailscale-discover.cpp tailscale_discovery.cpp)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)
if (WIN32)
    # mingw-w64 headers may default _WIN32_WINNT to pre-Windows-10; inet_pton needs >= Vista
    target_compile_definitions(${TARGET} PRIVATE _WIN32_WINNT=0x0A00)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

# llama-cluster-ui links the full ggml/llama stack (not just cpp-httplib) because discovery now
# runs in-process via the shared tailscale_discovery.cpp, rather than shelling out to
# llama-tailscale-discover - this is a deliberate tradeoff (fewer moving parts, one binary owns
# the whole cluster workflow) traded against the smaller, dependency-light binary this used to be.
set(TARGET llama-cluster-ui)
add_executable(${TARGET}
    cluster-ui.cpp
    tailscale_discovery.cpp
    cluster_layer_runner.cpp
    cluster_node_service.cpp
    coordinator_service.cpp
    subprocess_runner.cpp
)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE llama ggml cpp-httplib llama-cluster-proto ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)
if (WIN32)
    target_compile_definitions(${TARGET} PRIVATE _WIN32_WINNT=0x0A00)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

