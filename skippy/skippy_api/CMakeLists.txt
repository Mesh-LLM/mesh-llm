cmake_minimum_required(VERSION 3.27)

project(skippy_api LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SKIPPY_LINK_COMPILE_COMMANDS
  "Symlink compile_commands.json into the source tree"
  ON)

if(SKIPPY_LINK_COMPILE_COMMANDS
   AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND NOT CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
  file(CREATE_LINK
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
    SYMBOLIC)
endif()

set(SKIPPY_SANITIZERS "" CACHE STRING
  "Semicolon list of sanitizers, e.g. 'address;undefined' or 'thread'")

if(SKIPPY_SANITIZERS)
  if("thread" IN_LIST SKIPPY_SANITIZERS AND
     ("address" IN_LIST SKIPPY_SANITIZERS OR "memory" IN_LIST SKIPPY_SANITIZERS))
    message(FATAL_ERROR "thread sanitizer cannot be combined with address or memory")
  endif()

  if("memory" IN_LIST SKIPPY_SANITIZERS AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "memory sanitizer requires Clang")
  endif()

  list(JOIN SKIPPY_SANITIZERS "," _skippy_sanitize)

  # Restrict to C/C++ so llama.cpp's GPU backends (CUDA etc.) don't get the flags.
  set(_skippy_c_cxx "$<COMPILE_LANGUAGE:C,CXX>")
  add_compile_options(
    "$<${_skippy_c_cxx}:-fsanitize=${_skippy_sanitize}>"
    "$<${_skippy_c_cxx}:-fno-omit-frame-pointer>"
    "$<${_skippy_c_cxx}:-fno-sanitize-recover=all>"
    "$<${_skippy_c_cxx}:-g>"
  )
  if("memory" IN_LIST SKIPPY_SANITIZERS)
    add_compile_options("$<${_skippy_c_cxx}:-fsanitize-memory-track-origins=2>")
  endif()
  add_link_options("-fsanitize=${_skippy_sanitize}")

  message(STATUS "skippy: sanitizers enabled: ${_skippy_sanitize}")
endif()
option(SKIPPY_ENABLE_CLANG_TIDY "Run clang-tidy while compiling skippy targets" OFF)
if(SKIPPY_ENABLE_CLANG_TIDY)
  find_program(SKIPPY_CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
endif()

function(skippy_enable_tidy target)
  if(SKIPPY_ENABLE_CLANG_TIDY)
    set_target_properties(${target} PROPERTIES
      CXX_CLANG_TIDY "${SKIPPY_CLANG_TIDY_EXE}")
  endif()
endfunction()

if(SKIPPY_ENABLE_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY
          "$<$<STREQUAL:$<TARGET_PROPERTY:SOURCE_DIR>,${CMAKE_CURRENT_SOURCE_DIR}>:${SKIPPY_CLANG_TIDY_EXE}>")
endif()


add_library(skippy_core STATIC
    src/runtime.cpp
    src/normalize.cpp
    src/stage_plan.cpp
    src/execution_contract.cpp
    src/family.cpp
    src/stage_load.cpp
)
add_library(Skippy::Core ALIAS skippy_core)

target_compile_features(skippy_core PUBLIC cxx_std_23)
target_compile_options(skippy_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_include_directories(skippy_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)


option(SKIPPY_BUILD_LLAMA "Build the llama.cpp backend (needs SKIPPY_LLAMA_SOURCE_DIR)" ON)

if(SKIPPY_BUILD_LLAMA)
  #TODO, lets figure this out properly, aso I can just add_subdir
  set(SKIPPY_LLAMA_SOURCE_DIR "" CACHE PATH
        "Path to the prepared llama.cpp source checkout")

  if(NOT TARGET llama)
    if(NOT EXISTS "${SKIPPY_LLAMA_SOURCE_DIR}/include/llama.h")
      message(FATAL_ERROR
                "Set SKIPPY_LLAMA_SOURCE_DIR to the llama.cpp source checkout, "
                "or configure with -DSKIPPY_BUILD_LLAMA=OFF for core + tests only")
    endif()

    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

    add_subdirectory(
            "${SKIPPY_LLAMA_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}/llama"
            EXCLUDE_FROM_ALL
        )
  endif()

  add_library(skippy_llama STATIC
    src/llama/backend.cpp
    src/llama/model_inventory.cpp
    src/llama/batch.cpp
    src/llama/stage_planner.cpp
    src/llama/staged.cpp
    src/llama/stage_chain.cpp
    )
  add_library(Skippy::Llama ALIAS skippy_llama)
  target_compile_options(skippy_llama PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(skippy_llama PUBLIC Skippy::Core llama)

  # What the Rust side links
  # Core, plus the llama backend
  add_library(skippy_api INTERFACE)
  add_library(Skippy::Api ALIAS skippy_api)
  target_link_libraries(skippy_api INTERFACE Skippy::Core Skippy::Llama)
endif()

option(SKIPPY_API_BUILD_TESTS "Build Skippy API tests" OFF)

if(SKIPPY_API_BUILD_TESTS)
  include(FetchContent)
  enable_testing()

  if(NOT TARGET GTest::gmock_main)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.18.0
            GIT_SHALLOW TRUE
        )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(skippy_api_tests
      tests/runtime_test.cpp
      tests/pooling_test.cpp
      tests/selector_test.cpp
      tests/batch_test.cpp
      tests/normalize_test.cpp
      tests/workload_test.cpp
      tests/rerank_template_test.cpp
      tests/rerank_prompt_test.cpp
      tests/rerank_test.cpp
      tests/embed_test.cpp
      tests/encode_test.cpp
      tests/types_test.cpp
      tests/stage_plan_test.cpp
      tests/execution_contract_test.cpp
    tests/family_test.cpp
    tests/stage_chain_test.cpp
      tests/stage_load_test.cpp
      tests/workload_profile_test.cpp
      tests/borrow_stress_test.cpp
      tests/inventory_parser_test.cpp
  )
  target_compile_options(skippy_api_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(skippy_api_tests
        PRIVATE
            Skippy::Core
            GTest::gmock_main
    )
  include(GoogleTest)
  gtest_discover_tests(skippy_api_tests)
endif()

option(SKIPPY_API_BUILD_BENCHMARKS "Build Skippy API benchmarks" OFF)

if(SKIPPY_API_BUILD_BENCHMARKS)
  include(FetchContent)

  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
        GIT_SHALLOW TRUE
    )
  FetchContent_MakeAvailable(googlebenchmark)

  add_executable(skippy_api_benchmarks
        benchmarks/embedding_benchmark.cpp
    )
  target_link_libraries(skippy_api_benchmarks
        PRIVATE Skippy::Core benchmark::benchmark_main
    )

  # Old C ABI (skippy.h) against the facade, in one process on one llama build.
  # Needs the backend, since the C ABI lives inside the llama target.
  option(SKIPPY_API_BUILD_PLANNER_PROBE
  "Build the stage planner probe (needs SKIPPY_BUILD_LLAMA)" OFF)
option(SKIPPY_API_BUILD_FFI_BENCHMARKS
    "Build the old C ABI vs skippy_api benchmark (requires the llama backend)" OFF)

  if(SKIPPY_API_BUILD_FFI_BENCHMARKS)
    if(NOT SKIPPY_BUILD_LLAMA)
      message(FATAL_ERROR
        "SKIPPY_API_BUILD_FFI_BENCHMARKS needs SKIPPY_BUILD_LLAMA=ON")
    endif()

    add_executable(skippy_api_ffi_benchmarks
        benchmarks/ffi_vs_api_benchmark.cpp
        benchmarks/stage_plan_benchmark.cpp
      )
    target_compile_options(skippy_api_ffi_benchmarks PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_include_directories(skippy_api_ffi_benchmarks PRIVATE
        "${SKIPPY_LLAMA_SOURCE_DIR}/include"
        "${SKIPPY_LLAMA_SOURCE_DIR}/common"
      )
    target_link_libraries(skippy_api_ffi_benchmarks
        PRIVATE Skippy::Api benchmark::benchmark
      )
  endif()
endif()

if(SKIPPY_API_BUILD_PLANNER_PROBE)
  if(NOT SKIPPY_BUILD_LLAMA)
    message(FATAL_ERROR "SKIPPY_API_BUILD_PLANNER_PROBE needs SKIPPY_BUILD_LLAMA=ON")
  endif()
  add_executable(skippy_api_planner_probe tools/gguf_planner_probe.cpp)
  target_link_libraries(skippy_api_planner_probe PRIVATE Skippy::Api)
  target_compile_options(skippy_api_planner_probe PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_executable(skippy_api_stage_pipeline_probe tools/stage_pipeline_probe.cpp)
  add_executable(skippy_api_staged_probe tools/staged_model_probe.cpp)
  add_executable(skippy_api_stage_chain_probe tools/stage_chain_probe.cpp)
  target_compile_options(skippy_api_stage_chain_probe PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(skippy_api_stage_chain_probe PRIVATE Skippy::Api)
  target_compile_options(skippy_api_staged_probe PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(skippy_api_staged_probe PRIVATE Skippy::Api)
  target_link_libraries(skippy_api_stage_pipeline_probe PRIVATE Skippy::Api)
  target_compile_options(skippy_api_stage_pipeline_probe PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

find_program(SKIPPY_CLANG_FORMAT_EXE NAMES clang-format)

if(PROJECT_IS_TOP_LEVEL AND SKIPPY_CLANG_FORMAT_EXE)
  file(GLOB_RECURSE SKIPPY_FORMAT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp"
  )

  add_custom_target(format
    COMMAND "${SKIPPY_CLANG_FORMAT_EXE}" -i ${SKIPPY_FORMAT_SOURCES}
    COMMENT "clang-format: rewriting sources"
    VERBATIM)

  add_custom_target(format-check
    COMMAND "${SKIPPY_CLANG_FORMAT_EXE}" --dry-run --Werror ${SKIPPY_FORMAT_SOURCES}
    COMMENT "clang-format: checking sources"
    VERBATIM)
endif()
