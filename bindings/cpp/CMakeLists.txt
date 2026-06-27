# llmcore C++ client — CMake build.
#
# Requires gRPC + Protobuf development packages providing CMake config + the
# protoc compiler and grpc_cpp_plugin, e.g.:
#   apt install protobuf-compiler protobuf-compiler-grpc libgrpc++-dev libprotobuf-dev
#   (or build gRPC from source; vcpkg/conan also work)
cmake_minimum_required(VERSION 3.16)
project(llmcore_client_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# Resolve the protoc binary and the C++ gRPC plugin from the imported targets.
set(PROTOC $<TARGET_FILE:protobuf::protoc>)
set(GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

set(PROTO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../proto")
set(PROTO_FILES
  common errors inference catalog control audio
)

set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}/llmcore/v1")

set(GEN_SRCS "")
foreach(name ${PROTO_FILES})
  set(proto "${PROTO_ROOT}/llmcore/v1/${name}.proto")
  set(pb_cc   "${GENERATED_DIR}/llmcore/v1/${name}.pb.cc")
  set(pb_h    "${GENERATED_DIR}/llmcore/v1/${name}.pb.h")
  set(grpc_cc "${GENERATED_DIR}/llmcore/v1/${name}.grpc.pb.cc")
  set(grpc_h  "${GENERATED_DIR}/llmcore/v1/${name}.grpc.pb.h")
  add_custom_command(
    OUTPUT "${pb_cc}" "${pb_h}" "${grpc_cc}" "${grpc_h}"
    COMMAND ${PROTOC}
    ARGS --proto_path="${PROTO_ROOT}"
         --cpp_out="${GENERATED_DIR}"
         --grpc_out="${GENERATED_DIR}"
         --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}"
         "${proto}"
    DEPENDS "${proto}"
    COMMENT "protoc (C++/gRPC) ${name}.proto"
    VERBATIM)
  list(APPEND GEN_SRCS "${pb_cc}" "${grpc_cc}")
endforeach()

add_library(llmcore_client STATIC ${GEN_SRCS} src/client.cpp)
target_include_directories(llmcore_client PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${GENERATED_DIR}")
target_link_libraries(llmcore_client PUBLIC gRPC::grpc++ protobuf::libprotobuf)

add_executable(quickstart examples/quickstart.cpp)
target_link_libraries(quickstart PRIVATE llmcore_client)

enable_testing()
add_executable(e2e tests/e2e.cpp)
target_link_libraries(e2e PRIVATE llmcore_client)
add_test(NAME e2e COMMAND e2e)
# Forward the python interpreter selection into the test environment.
if(DEFINED ENV{LLMCORE_BRIDGE_PYTHON})
  set_tests_properties(e2e PROPERTIES
    ENVIRONMENT "LLMCORE_BRIDGE_PYTHON=$ENV{LLMCORE_BRIDGE_PYTHON}")
endif()
