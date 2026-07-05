find_package(Threads REQUIRED)

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

# llama-cluster-ui links the full ggml/llama stack (not just cpp-httplib) because discovery now
# runs in-process via the shared tailscale_discovery.cpp, rather than shelling out to
# llama-tailscale-discover - this is a deliberate tradeoff (fewer moving parts, one binary owns
# the whole cluster workflow) traded against the smaller, dependency-light binary this used to be.
set(TARGET llama-cluster-ui)
add_executable(${TARGET} cluster-ui.cpp tailscale_discovery.cpp)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE llama ggml cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)

