# Tile Based ML Kernel Runtime

A C++17 systems project that implements and benchmarks tile-based matrix kernels
with simple runtime-style scheduling, inspired by ML kernel runtimes such as
Graphcore Poplibs.

## Project Goal

Build a clean, testable runtime scaffold that evolves from:

1. baseline matrix operations
2. tiled and parallel GEMM kernels
3. task partitioning and scheduler dispatch
4. reproducible benchmarking and performance analysis

## Phase 1 Status

This repository currently contains the Phase 1 setup:

- CMake build configuration
- core library target scaffold
- benchmark executable scaffold
- test executable scaffold
- project directory structure for future phases

## Repository Layout

```text
.
├── benchmarks/
├── docs/
├── include/
├── scripts/
├── src/
├── tests/
├── CMakeLists.txt
└── tile_runtime.plan.md
```

## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

## Run

From the `build` directory:

```bash
./benchmark_gemm
ctest --output-on-failure
```

## Notes

- OpenMP is detected and linked automatically when available.
- Kernel implementations and API headers are introduced in later phases.
