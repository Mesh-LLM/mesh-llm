set(TARGET llama-freivalds-check)
add_executable(${TARGET} freivalds-check.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
