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
