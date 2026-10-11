# Skew-Aware Adaptive Parallel Near-Duplicate Detection on Multicore CPUs

## Project Overview

This project investigates CPU-based near-duplicate document detection using SimHash locality-sensitive fingerprinting and OpenMP shared-memory parallelism. The primary research question is whether an adaptive, workload-aware task scheduling strategy can improve load balance and execution speed over standard OpenMP scheduling when candidate document comparisons are unevenly distributed across compute threads.

---

## Current Status: Phase 0 Completed

**Phase 0 establishes the base development environment, project build configuration, and validates C++17 and OpenMP runtime execution.**

No detection algorithms or benchmark pipelines are implemented in this phase.

### Phase 0 Implementation Scope
- Enforces strict **C++17** compilation without compiler-specific extensions.
- Requires and links **OpenMP** (`OpenMP::OpenMP_CXX`) via CMake, failing with an explicit fatal error if OpenMP support is missing.
- Implements a minimal test executable (`src/main.cpp`) that:
  1. Identifies the active C++ standard via `__cplusplus`.
  2. Discovers hardware processor availability (`omp_get_num_procs()`).
  3. Executes an OpenMP parallel team region (`#pragma omp parallel`).
  4. Dynamically reports the actual thread team size (`omp_get_num_threads()`).
  5. Exits cleanly with return code 0.

---

## Toolchain & System Requirements

- **Operating System:** Windows 11 (64-bit)
- **Compiler:** GCC 13+ (tested with GCC 16.2.0 via MSYS2 UCRT64, 64-bit)
- **Build System:** CMake $\ge 3.16$ (tested with CMake 4.4.4) and Ninja build generator
- **Runtime Libraries:** OpenMP runtime (`libgomp`, `libwinpthread`)
- **Target CPU Profile:** Intel Core i5-1135G7 (4 physical cores, 8 logical processors)

---

## Build & Run Instructions (PowerShell)

### 1. Ensure UCRT64 Tools are in PATH
To use the verified 64-bit toolchain installed in `C:\msys64\ucrt64\bin`, set the session PATH in PowerShell:
```powershell
$env:PATH = "C:\msys64\ucrt64\bin;C:\msys64\usr\bin;$env:PATH"
```

### 2. Configure the Project with CMake

#### Debug Configuration:
```powershell
cmake -B build -G "Ninja" -DCMAKE_CXX_COMPILER="C:/msys64/ucrt64/bin/g++.exe" -DCMAKE_BUILD_TYPE=Debug
```

#### Release Configuration:
```powershell
cmake -B build-release -G "Ninja" -DCMAKE_CXX_COMPILER="C:/msys64/ucrt64/bin/g++.exe" -DCMAKE_BUILD_TYPE=Release
```

### 3. Build the Executable
```powershell
# For Debug
cmake --build build

# For Release
cmake --build build-release
```

### 4. Run the Test Program
```powershell
# For Debug
.\build\hpc_near_duplicate.exe

# For Release
.\build-release\hpc_near_duplicate.exe
```

---

## Expected Output

When running either the Debug or Release build, the output should appear as follows:

```text
===============================================================
 Skew-Aware Adaptive Parallel Near-Duplicate Detection
 Phase 0: Environment & OpenMP Verification
===============================================================

[C++ Standard Check]
  __cplusplus macro value : 201703
  Active language standard: C++17

[System Resource Discovery]
  Logical processors detected : 8
  Default max threads pool    : 8

[OpenMP Team Verification]
  Entered OpenMP parallel region.
  Threads in active parallel team: 8

[Status] Phase 0 verification completed successfully.
```

---

## Environment Limitations & Verified Scope

- **Verified:** C++17 compilation, OpenMP 5.2 parallel region execution, multi-threaded team formation (8 threads verified), and both Debug and Release CMake builds.
- **Environment Note:** The legacy 32-bit compiler located at `C:\MinGW\bin\g++.exe` is outdated and lacks POSIX thread support for OpenMP. Builds must target the modern 64-bit toolchain installed at `C:\msys64\ucrt64\bin`.
