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
  src/gemm.cpp
  src/tile_driver.cpp
  src/tile_kernel_scalar.cpp
  src/tile_kernel_avx2.cpp
  src/tile_kernel_avx512.cpp
)

# std::experimental::simd is only available on GCC (libstdc++)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  list(APPEND TILE_CORE_SOURCES src/tile_kernel_stdsimd.cpp)
endif()

add_library(tile_runtime_objects OBJECT ${TILE_CORE_SOURCES})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_definitions(tile_runtime_objects PRIVATE TILE_HAS_STD_SIMD)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/tile_kernel_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
  set_source_files_properties(src/tile_kernel_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512vl -mfma")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set_source_files_properties(src/tile_kernel_stdsimd.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
  endif()
elseif(MSVC)
  set_source_files_properties(src/tile_kernel_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
  set_source_files_properties(src/tile_kernel_avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
endif()

target_include_directories(tile_runtime_objects
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(tile_runtime_objects PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(tile_runtime_core STATIC
  $<TARGET_OBJECTS:tile_runtime_objects>
  src/cpu_features.cpp
)
target_include_directories(tile_runtime_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
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

add_executable(test_gemm_no_simd
  tests/test_gemm.cpp
  tests/cpu_features_none.cpp
  $<TARGET_OBJECTS:tile_runtime_objects>
)
target_include_directories(test_gemm_no_simd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(test_gemm_no_simd PRIVATE OpenMP::OpenMP_CXX)
endif()

enable_testing()
add_test(NAME runtime_smoke_test COMMAND test_runtime)
add_test(NAME tensor_tests COMMAND test_tensor)
add_test(NAME gemm_tests COMMAND test_gemm)
add_test(NAME gemm_tests_no_simd COMMAND test_gemm_no_simd)
