# llmcore C++ client — CMake build.
#
# gRPC + Protobuf are resolved automatically:
#   1. A system/installed gRPC CMake config is used when present (fastest), e.g.
#      `apt install protobuf-compiler protobuf-compiler-grpc libgrpc++-dev
#      libprotobuf-dev`, or `brew install grpc`, or vcpkg/conan.
#   2. Otherwise gRPC (with its bundled protobuf + abseil + grpc_cpp_plugin) is
#      fetched and built from source via FetchContent — no system packages
#      needed, just a compiler + network. This is slow on the first configure
#      (gRPC is large) but makes the build work on any machine out of the box.
#      Override the pinned tag with -DLLMCORE_GRPC_TAG=vX.Y.Z, or force fetching
#      with -DLLMCORE_FETCH_GRPC=ON.
cmake_minimum_required(VERSION 3.16)
project(llmcore_client_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LLMCORE_FETCH_GRPC "Always build gRPC from source via FetchContent" OFF)
set(LLMCORE_GRPC_TAG "v1.65.0" CACHE STRING "gRPC git tag when fetched from source")

if(NOT LLMCORE_FETCH_GRPC)
  find_package(Protobuf CONFIG QUIET)
  find_package(gRPC CONFIG QUIET)
endif()

if(gRPC_FOUND)
  message(STATUS "llmcore: using installed gRPC ${gRPC_VERSION}")
  set(LLMCORE_GRPCPP gRPC::grpc++)
  set(LLMCORE_LIBPROTOBUF protobuf::libprotobuf)
  set(PROTOC $<TARGET_FILE:protobuf::protoc>)
  set(GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
  # Installed protoc bundles the well-known-type .proto files; expose their
  # include dir too (harmless if already on the default path).
  set(LLMCORE_WKT_PROTO_PATH "${Protobuf_INCLUDE_DIR}")
else()
  message(STATUS "llmcore: gRPC not found — fetching ${LLMCORE_GRPC_TAG} via FetchContent "
                 "(first configure builds gRPC from source; this is slow)")
  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)
  # Some gRPC third_party modules (c-ares, ...) still declare a pre-3.5
  # cmake_minimum_required, which CMake >= 4 rejects; this shim lets them
  # configure.
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
  # gRPC build knobs: client-only, no tests. We link the in-tree targets and
  # never install, so disable install/export rules across the bundled deps —
  # otherwise utf8_range's install(EXPORT) references abseil targets that are
  # not in any export set and CMake's generate step fails.
  set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(gRPC_BUILD_CSHARP_EXT OFF CACHE BOOL "" FORCE)
  set(gRPC_INSTALL OFF CACHE BOOL "" FORCE)
  set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(utf8_range_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
  FetchContent_Declare(grpc
    GIT_REPOSITORY https://github.com/grpc/grpc
    GIT_TAG ${LLMCORE_GRPC_TAG}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
  FetchContent_MakeAvailable(grpc)
  # In-tree targets created by gRPC's build.
  set(LLMCORE_GRPCPP grpc++)
  set(LLMCORE_LIBPROTOBUF libprotobuf)
  set(PROTOC $<TARGET_FILE:protoc>)
  set(GRPC_CPP_PLUGIN $<TARGET_FILE:grpc_cpp_plugin>)
  # The source-built protoc does not know where the well-known-type .proto files
  # live; point at the bundled protobuf's src so `import
  # "google/protobuf/struct.proto"` resolves.
  set(LLMCORE_WKT_PROTO_PATH "${grpc_SOURCE_DIR}/third_party/protobuf/src")
endif()

# Normalize to a real absolute path: protoc rejects proto_paths containing ".."
# (it does not canonicalize, so "<dir>/../proto" fails as a non-prefix match).
get_filename_component(PROTO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../proto" REALPATH)
set(PROTO_FILES
  common errors inference catalog control audio sessions vector presets
)

# Proto import path: our contract dir, plus the well-known-types dir when set
# (needed by a source-built protoc that lacks the bundled WKT on its path).
set(PROTO_PATH_ARGS --proto_path=${PROTO_ROOT})
if(LLMCORE_WKT_PROTO_PATH)
  list(APPEND PROTO_PATH_ARGS --proto_path=${LLMCORE_WKT_PROTO_PATH})
endif()

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
    # No literal quotes around the paths: VERBATIM already quotes each argument
    # for the shell, so embedding `"` here would pass the quote characters
    # through to protoc (which then can't find the "quoted" directory).
    COMMAND ${PROTOC}
    ARGS ${PROTO_PATH_ARGS}
         --cpp_out=${GENERATED_DIR}
         --grpc_out=${GENERATED_DIR}
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         ${proto}
    DEPENDS "${proto}"
    COMMENT "protoc (C++/gRPC) ${name}.proto"
    VERBATIM)
  list(APPEND GEN_SRCS "${pb_cc}" "${grpc_cc}")
endforeach()

add_library(llmcore_client STATIC ${GEN_SRCS} src/client.cpp)
target_include_directories(llmcore_client PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${GENERATED_DIR}")
target_link_libraries(llmcore_client PUBLIC ${LLMCORE_GRPCPP} ${LLMCORE_LIBPROTOBUF})

add_executable(quickstart examples/quickstart.cpp)
target_link_libraries(quickstart PRIVATE llmcore_client)

add_executable(sessions examples/sessions.cpp)
target_link_libraries(sessions PRIVATE llmcore_client)

enable_testing()
add_executable(e2e tests/e2e.cpp)
target_link_libraries(e2e PRIVATE llmcore_client)
add_test(NAME e2e COMMAND e2e)
# Forward the python interpreter selection into the test environment.
if(DEFINED ENV{LLMCORE_BRIDGE_PYTHON})
  set_tests_properties(e2e PROPERTIES
    ENVIRONMENT "LLMCORE_BRIDGE_PYTHON=$ENV{LLMCORE_BRIDGE_PYTHON}")
endif()
