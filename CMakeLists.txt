cmake_minimum_required(VERSION 3.16)

project(tile_ml_runtime VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -pedantic)
endif()

find_package(OpenMP QUIET)

set(TILE_CORE_SOURCES
  src/runtime_placeholder.cpp
  src/tensor.cpp
  src/gemm_naive.cpp
  src/gemm_tiled.cpp
  src/gemm_parallel.cpp
  src/gemm_avx.cpp
  src/gemm_avx512.cpp
  src/gemm_parallel_simd.cpp
)

# std::experimental::simd is only available on GCC (libstdc++)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  list(APPEND TILE_CORE_SOURCES src/gemm_simd.cpp)
endif()

add_library(tile_runtime_core STATIC ${TILE_CORE_SOURCES})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_definitions(tile_runtime_core PUBLIC TILE_HAS_STD_SIMD)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/gemm_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
  set_source_files_properties(src/gemm_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512vl -mfma")
  set_source_files_properties(src/gemm_parallel_simd.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set_source_files_properties(src/gemm_simd.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
  endif()
elseif(MSVC)
  set_source_files_properties(src/gemm_avx.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
  set_source_files_properties(src/gemm_avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
  set_source_files_properties(src/gemm_parallel_simd.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
endif()

target_include_directories(tile_runtime_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(tile_runtime_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(benchmark_gemm
  benchmarks/benchmark_gemm.cpp
)
target_link_libraries(benchmark_gemm PRIVATE tile_runtime_core)

add_executable(test_runtime
  tests/test_runtime.cpp
)
target_link_libraries(test_runtime PRIVATE tile_runtime_core)

add_executable(test_tensor
  tests/test_tensor.cpp
)
target_link_libraries(test_tensor PRIVATE tile_runtime_core)

add_executable(test_gemm
  tests/test_gemm.cpp
)
target_link_libraries(test_gemm PRIVATE tile_runtime_core)

enable_testing()
add_test(NAME runtime_smoke_test COMMAND test_runtime)
add_test(NAME tensor_tests COMMAND test_tensor)
add_test(NAME gemm_tests COMMAND test_gemm)
