# Skew-Aware Adaptive Parallel Near-Duplicate Detection on Multicore CPUs

## Project Overview

This project investigates CPU-based near-duplicate document detection using SimHash locality-sensitive fingerprinting and OpenMP shared-memory parallelism. The primary research question is whether an adaptive, workload-aware task scheduling strategy can improve load balance and execution speed over standard OpenMP scheduling when candidate document comparisons are unevenly distributed across compute threads.

---

## Current Status: Phase 1 Completed

The project has completed:
- **Phase 0:** Environment inspection, C++17 validation, and OpenMP runtime verification.
- **Phase 1:** In-memory document data model, ground-truth duplicate cluster semantics, and a deterministic synthetic dataset generator with balanced and skewed workload profiles.

*(Subsequent components—SimHash fingerprinting, candidate band indexing, and parallel OpenMP schedulers—are part of future phases and are not yet implemented.)*

---

## Document Representation & Ground-Truth Semantics

Each document is modeled by the `Document` structure (`include/document.hpp`):
- `id`: Unique, stable 64-bit integer identifier.
- `group_id`: Ground-truth duplicate cluster identifier.
- `kind`: Categorical classification (`DocumentKind`):
  - `Original`: Primary seed document that defines a new duplicate cluster.
  - `ExactCopy`: Verbatim duplicate of an existing original document.
  - `ModifiedCopy`: Controlled variant of an original document with perturbed tokens.
  - `Unrelated`: Standalone document that does not belong to any duplicate group (singleton cluster of size 1).
- `source_id`: The ID of the original seed document for copies (`-1` for originals and unrelated documents).
- `text`: Readable, space-separated sequence of tokens generated from a curated vocabulary.

### Ground-Truth Evaluation Role
Documents sharing the same `group_id` are ground-truth duplicates by construction. This provides an exact evaluation oracle for later phases to measure true positives, false positives, precision, and recall without needing to run any detection algorithm.

---

## Synthetic Dataset Generator

The generator (`include/dataset_generator.hpp`, `src/dataset_generator.cpp`) produces reproducible document corpuses configured via `DatasetConfig`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `total_documents` | `size_t` | `100` | Total number of documents to produce |
| `min_words` / `max_words` | `size_t` | `40` / `80` | Range of token length per document |
| `exact_duplicate_ratio` | `double` | `0.20` | Fraction of documents that are exact copies |
| `modified_copy_ratio` | `double` | `0.20` | Fraction of documents that are modified copies |
| `unrelated_ratio` | `double` | `0.10` | Fraction of documents that are unrelated singletons |
| `modification_rate` | `double` | `0.15` | Fraction of tokens altered in a modified copy |
| `random_seed` | `uint64_t` | `42` | Seed for deterministic pseudorandom generation (`std::mt19937_64`) |
| `profile` | `WorkloadProfile` | `Balanced` | Duplicate cluster size distribution (`Balanced` or `Skewed`) |

### Workload Profiles: Balanced vs. Skewed
- **`Balanced` Profile:** Distributes exact and modified copies uniformly across original documents (round-robin). All duplicate clusters have similar, small group sizes (e.g., 2–3 documents per group).
- **`Skewed` Profile:** Distributes copies according to a Zipfian power-law weighting ($w_i = 1 / (i + 1)^2$). A small number of "heavy-hitter" original documents accumulate the vast majority of copies, producing a severe workload skew (e.g., one group containing 30+ documents while others contain 1 or 2).

> [!NOTE]
> **Important Distinction on Workload Scope:**  
> In Phase 1, the `Balanced` and `Skewed` profiles govern strictly the distribution and sizes of **ground-truth duplicate groups**. They do **not** dictate the downstream SimHash candidate bucket distribution. The actual bucket workloads and pairwise candidate comparison counts will be experimentally measured once SimHash fingerprinting and candidate band generation are implemented in subsequent phases.

### Deterministic Reproducibility
The generator uses `std::mt19937_64` initialized with `random_seed`. For identical configurations and seeds, it produces bitwise-identical documents, texts, IDs, and ground-truth group assignments.

---

## Toolchain & System Requirements

- **Operating System:** Windows 11 (64-bit)
- **Compiler:** GCC 13+ (verified with GCC 16.2.0 via MSYS2 UCRT64, 64-bit)
- **Build System:** CMake $\ge 3.16$ (verified with CMake 4.4.4) and Ninja generator
- **Runtime Libraries:** OpenMP runtime (`libgomp`, `libwinpthread`)
- **Target CPU Profile:** Intel Core i5-1135G7 (4 physical cores, 8 logical processors)

---

## Build, Test & Run Instructions (PowerShell)

### 1. Ensure UCRT64 Tools are in PATH
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

### 3. Build All Targets
```powershell
# For Debug
cmake --build build

# For Release
cmake --build build-release
```

### 4. Run Automated Tests
```powershell
# Run the test executable directly
.\build\test_dataset_generator.exe

# Or via CTest
ctest --test-dir build --output-on-failure
```

### 5. Run the Demonstration Program
```powershell
# For Debug
.\build\hpc_near_duplicate.exe

# For Release
.\build-release\hpc_near_duplicate.exe
```

---

## Sample Program Output

```text
===============================================================
 Skew-Aware Adaptive Parallel Near-Duplicate Detection
===============================================================

[Phase 0 Verification]
  __cplusplus macro value : 201703
  Active language standard: C++17
  Logical processors      : 8
  Max OpenMP thread pool  : 8
  OpenMP active team size : 8 threads
  Phase 0 status          : PASS

[Phase 1 Demonstration: Dataset Generation]
---------------------------------------------------------
 Synthetic Dataset Summary
---------------------------------------------------------
  Total Documents          : 100
  Original Documents       : 40
  Exact Copies             : 25
  Modified Copies          : 25
  Unrelated Singletons     : 10
  Total Ground-Truth Groups: 50
  Duplicate Groups (size>1): 40
  Max Group Size           : 3
  Min Group Size           : 1
  Workload Profile         : Balanced
  Random Seed              : 42
---------------------------------------------------------
---------------------------------------------------------
 Synthetic Dataset Summary
---------------------------------------------------------
  Total Documents          : 100
  Original Documents       : 40
  Exact Copies             : 25
  Modified Copies          : 25
  Unrelated Singletons     : 10
  Total Ground-Truth Groups: 50
  Duplicate Groups (size>1): 9
  Max Group Size           : 31
  Min Group Size           : 1
  Workload Profile         : Skewed
  Random Seed              : 42
---------------------------------------------------------

[Status] Phase 1 execution completed successfully.
```
