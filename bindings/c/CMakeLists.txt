# llmcore C client — CMake build.
#
# Requires libcurl + cJSON development packages, e.g.:
#   apt install libcurl4-openssl-dev libcjson-dev
cmake_minimum_required(VERSION 3.16)
project(llmcore_client_c C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CURL REQUIRED)

# cJSON: prefer its CMake config, else pkg-config (libcjson), else assume -lcjson.
set(_CJSON_VIA "")
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

add_library(llmcore_c STATIC src/llmcore_client.c)
target_include_directories(llmcore_c PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(llmcore_c PUBLIC CURL::libcurl)

if(_CJSON_VIA STREQUAL "config")
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
