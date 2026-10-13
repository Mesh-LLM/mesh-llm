cmake_minimum_required(VERSION 3.20)

project(skippy_stage_rewriter LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

add_executable(skippy-stage-rewriter
    main.cpp)

target_compile_features(skippy-stage-rewriter PRIVATE cxx_std_17)
target_include_directories(skippy-stage-rewriter SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS})
target_compile_definitions(skippy-stage-rewriter PRIVATE
    ${LLVM_DEFINITIONS})

target_link_libraries(skippy-stage-rewriter PRIVATE
    clang-cpp
    LLVM)

include(CTest)
if (BUILD_TESTING)
    add_executable(skippy-stage-rewriter-fixture tests/rewriter_fixture.cpp)
    target_compile_features(skippy-stage-rewriter-fixture PRIVATE cxx_std_17)
    target_include_directories(skippy-stage-rewriter-fixture SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_libraries(skippy-stage-rewriter-fixture PRIVATE LLVM)
    add_test(
        NAME skippy-stage-rewriter-fixture-native
        COMMAND
            $<TARGET_FILE:skippy-stage-rewriter-fixture>
            $<TARGET_FILE:skippy-stage-rewriter>
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/source)
endif()
