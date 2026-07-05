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
add_executable(${TARGET} tailscale-discover.cpp)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)

set(TARGET llama-cluster-ui)
add_executable(${TARGET} cluster-ui.cpp)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)

