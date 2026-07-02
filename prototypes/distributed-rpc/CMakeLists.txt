find_package(Threads REQUIRED)

set(TARGET llama-distributed-tap)
add_executable(${TARGET} distributed-tap.c)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET ${TARGET} PROPERTY C_STANDARD 11)

