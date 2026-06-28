# llmcore C client — CMake build.
#
# libcurl is required (a system dev package: `apt install libcurl4-openssl-dev`,
# or `brew install curl`). cJSON is resolved automatically:
#   1. An installed cJSON (CMake config or pkg-config `libcjson`) is used when
#      present (`apt install libcjson-dev`).
#   2. Otherwise cJSON is fetched and built from source via FetchContent — it is
#      a single small translation unit, so this adds only seconds to the build
#      and needs no system package. Force it with -DLLMCORE_FETCH_CJSON=ON, or
#      pin a tag with -DLLMCORE_CJSON_TAG=vX.Y.Z.
cmake_minimum_required(VERSION 3.16)
project(llmcore_client_c C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LLMCORE_FETCH_CJSON "Always build cJSON from source via FetchContent" OFF)
set(LLMCORE_CJSON_TAG "v1.7.18" CACHE STRING "cJSON git tag when fetched from source")

find_package(CURL REQUIRED)

# cJSON: prefer an installed CMake config, else pkg-config (libcjson), else fetch.
set(_CJSON_VIA "")
if(NOT LLMCORE_FETCH_CJSON)
  find_package(cJSON QUIET CONFIG)
  if(cJSON_FOUND)
    set(_CJSON_VIA "config")
  else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(CJSON IMPORTED_TARGET libcjson)
      if(CJSON_FOUND)
        set(_CJSON_VIA "pkgconfig")
      endif()
    endif()
  endif()
endif()

if(NOT _CJSON_VIA)
  message(STATUS "llmcore: cJSON not found — fetching ${LLMCORE_CJSON_TAG} via FetchContent")
  include(FetchContent)
  set(ENABLE_CJSON_TEST OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(CJSON_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(ENABLE_CJSON_UNINSTALL OFF CACHE BOOL "" FORCE)
  # cJSON still declares `cmake_minimum_required(VERSION 3.0)`, which CMake >= 4
  # rejects; this shim lets the old project configure.
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
  FetchContent_Declare(cjson
    GIT_REPOSITORY https://github.com/DaveGamble/cJSON
    GIT_TAG ${LLMCORE_CJSON_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(cjson)
  set(_CJSON_VIA "fetched")
  # cJSON's in-tree target doesn't export its header dir; add it so consumers
  # resolve `#include <cJSON.h>` (the fallback the client uses when there is no
  # <cjson/cJSON.h> system layout).
  target_include_directories(cjson INTERFACE $<BUILD_INTERFACE:${cjson_SOURCE_DIR}>)
endif()

add_library(llmcore_c STATIC src/llmcore_client.c)
target_include_directories(llmcore_c PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(llmcore_c PUBLIC CURL::libcurl)

if(_CJSON_VIA STREQUAL "config" OR _CJSON_VIA STREQUAL "fetched")
  target_link_libraries(llmcore_c PUBLIC cjson)
elseif(_CJSON_VIA STREQUAL "pkgconfig")
  target_link_libraries(llmcore_c PUBLIC PkgConfig::CJSON)
else()
  message(WARNING "cJSON not found via config/pkg-config; assuming -lcjson on the default path")
  target_link_libraries(llmcore_c PUBLIC cjson)
endif()

add_executable(quickstart examples/quickstart.c)
target_link_libraries(quickstart PRIVATE llmcore_c)

enable_testing()
add_executable(e2e tests/e2e.c)
target_link_libraries(e2e PRIVATE llmcore_c)
add_test(NAME e2e COMMAND e2e)
if(DEFINED ENV{LLMCORE_BRIDGE_PYTHON})
  set_tests_properties(e2e PROPERTIES
    ENVIRONMENT "LLMCORE_BRIDGE_PYTHON=$ENV{LLMCORE_BRIDGE_PYTHON}")
endif()
