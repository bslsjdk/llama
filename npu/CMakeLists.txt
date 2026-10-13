cmake_minimum_required(VERSION 3.22)
project(llama_qnn_terminal_probe LANGUAGES CXX)

set(QNN_INC "" CACHE PATH "Directory containing QNN SDK headers")
if(NOT QNN_INC)
  message(FATAL_ERROR "Set -DQNN_INC=/path/to/QNN/include (headers are compile-time dependencies)")
endif()

add_executable(qnn-probe qnn_probe.cpp)
target_include_directories(qnn-probe PRIVATE "${QNN_INC}")
target_compile_features(qnn-probe PRIVATE cxx_std_17)
target_compile_options(qnn-probe PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(qnn-probe PRIVATE dl)

# Executes one tiny FP16 matmul graph on HTP and checks its output against CPU.
add_executable(qnn-matmul-smoke qnn_matmul_smoke.cpp)
target_include_directories(qnn-matmul-smoke PRIVATE "${QNN_INC}")
target_compile_features(qnn-matmul-smoke PRIVATE cxx_std_17)
target_compile_options(qnn-matmul-smoke PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(qnn-matmul-smoke PRIVATE dl)
