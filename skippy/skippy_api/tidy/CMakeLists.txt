cmake_minimum_required(VERSION 3.27)
project(skippy_tidy_checks LANGUAGES C CXX)

# Out-of-tree clang-tidy module, loaded with `clang-tidy --load=<this library>`.
# It must be built against the exact LLVM/Clang that owns the clang-tidy binary:
#   cmake -S skippy_api/tidy -B target/skippy-tidy \
#         -DClang_DIR=/opt/homebrew/opt/llvm/lib/cmake/clang   (Homebrew)
#         -DClang_DIR=/usr/lib/llvm-20/lib/cmake/clang          (Debian/Ubuntu)
find_package(Clang REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG HINTS "${Clang_DIR}/../llvm")

add_library(SkippyTidyChecks MODULE
    SkippyTidyModule.cpp
    checks/BoundaryIsolationCheck.cpp
    checks/NoExpectedVoidCheck.cpp
    checks/NoFailureOptionalCheck.cpp
    checks/NoMutexCheck.cpp
    checks/NoRawLoopsCheck.cpp
    checks/OptionalErrorPolarityCheck.cpp
    checks/PrecededByCheck.cpp
    checks/NoAllocInOomHandlerCheck.cpp
    checks/BridgeBarrierCheck.cpp
    checks/SingleCallSiteCheck.cpp
    checks/NoLockAcrossNativeCallCheck.cpp
    checks/NoVoidSilencingCheck.cpp
    checks/NoProcessWideStateCheck.cpp
    checks/NoRawPointerCheck.cpp
    checks/NoThrowingAccessorCheck.cpp
    checks/RustStringLossyCheck.cpp
    checks/ExplicitNarrowingCheck.cpp
    checks/ConstNativeMutationCheck.cpp
)

target_include_directories(SkippyTidyChecks SYSTEM PRIVATE ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_include_directories(SkippyTidyChecks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(SkippyTidyChecks PRIVATE ${LLVM_DEFINITIONS_LIST})
target_compile_features(SkippyTidyChecks PRIVATE cxx_std_17)
set_target_properties(SkippyTidyChecks PROPERTIES PREFIX "" CXX_EXTENSIONS OFF)

if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(SkippyTidyChecks PRIVATE -fno-rtti)
endif()

# The plugin resolves clang symbols from the clang-tidy process at load time;
# linking the clang libraries into it would duplicate every registry.
if(APPLE)
    target_link_options(SkippyTidyChecks PRIVATE "LINKER:-undefined,dynamic_lookup")
endif()

# Lit-free regression tests: each fixture carries its own FileCheck lines.
find_program(SKIPPY_CLANG_TIDY_EXE NAMES clang-tidy HINTS "${LLVM_TOOLS_BINARY_DIR}")
find_program(SKIPPY_FILECHECK_EXE NAMES FileCheck HINTS "${LLVM_TOOLS_BINARY_DIR}")
if(SKIPPY_CLANG_TIDY_EXE AND SKIPPY_FILECHECK_EXE)
    enable_testing()
    file(GLOB SKIPPY_TIDY_FIXTURES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")
    foreach(fixture IN LISTS SKIPPY_TIDY_FIXTURES)
        get_filename_component(name "${fixture}" NAME_WE)
        add_test(NAME tidy.${name}
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test/run.sh"
                    "${SKIPPY_CLANG_TIDY_EXE}" "${SKIPPY_FILECHECK_EXE}"
                    "$<TARGET_FILE:SkippyTidyChecks>" "${fixture}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/test/stubs")
    endforeach()
endif()
