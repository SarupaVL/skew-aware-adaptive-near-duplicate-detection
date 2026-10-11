# Skew-Aware Adaptive Parallel Near-Duplicate Detection on Multicore CPUs

## Project Overview

This project investigates CPU-based near-duplicate document detection using SimHash locality-sensitive fingerprinting and OpenMP shared-memory parallelism. The primary research question is whether an adaptive, workload-aware task scheduling strategy can improve load balance and execution speed over standard OpenMP scheduling when candidate document comparisons are unevenly distributed across compute threads.

---

## Current Status: Phase 2 Completed

The project has completed:
- **Phase 0:** Environment inspection, C++17 validation, and OpenMP runtime verification.
- **Phase 1:** In-memory document data model, ground-truth duplicate cluster semantics, and deterministic synthetic dataset generator.
- **Phase 2:** 64-bit SimHash locality-sensitive fingerprinting, deterministic FNV-1a hashing, token shingling, and bitwise Hamming distance computation.

*(Subsequent components—candidate band indexing, duplicate pair verification, and OpenMP parallel loop schedulers—are part of future phases and are not yet implemented.)*

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
> In Phase 1 and 2, the `Balanced` and `Skewed` profiles govern strictly the distribution and sizes of **ground-truth duplicate groups**. They do **not** dictate the downstream SimHash candidate bucket distribution. The actual bucket workloads and pairwise candidate comparison counts will be experimentally measured once candidate band generation is implemented in Phase 3.

---

## SimHash Fingerprinting & Hamming Distance (Phase 2)

The SimHash module (`include/simhash.hpp`, `src/simhash.cpp`) transforms variable-length document text into a compact 64-bit fingerprint that preserves cosine similarity in Hamming space.

### 1. Tokenization & Normalization
- Text is tokenized by separating contiguous non-whitespace character sequences.
- Leading, trailing, and repeated whitespace (spaces, tabs, newlines, carriage returns) are cleanly ignored.

### 2. Token Shingling (Word $k$-grams)
- Consecutive tokens are grouped into overlapping windows of size $k$ (default: `kDefaultShingleSize = 3`).
- For example, tokens `[w0, w1, w2, w3]` produce shingles `["w0 w1 w2", "w1 w2 w3"]`.
- **Short-document handling:** If a document contains fewer tokens than $k$, all available tokens are joined into a single fallback shingle so the document still receives a valid fingerprint.

### 3. Deterministic 64-bit Hashing (FNV-1a)
- Each shingle is hashed into a 64-bit unsigned integer using the **FNV-1a** algorithm (`fnv1a_hash64`):
  $$\text{hash} = (\text{hash} \oplus \text{byte}) \times 1099511628211ULL$$
  starting from offset basis `14695981039346656037ULL`.
- Unlike `std::hash` (which is implementation-defined and varies across standard libraries), FNV-1a is strictly platform-independent and deterministic across compilers and runs.

### 4. Weighted Bit Voting
- An array of 64 signed accumulators `V[0..63]` is initialized to zero.
- For each shingle hash $H$:
  - If bit $i$ of $H$ is 1, $V[i] \mathrel{+}= 1$ (positive vote).
  - If bit $i$ of $H$ is 0, $V[i] \mathrel{-}= 1$ (negative vote).
- The final 64-bit fingerprint sets bit $i = 1$ if $V[i] > 0$, else $0$.
- **Empty-document handling:** Empty or whitespace-only documents deterministically return `0ULL`.

### 5. Bitwise Hamming Distance
- `hamming_distance(a, b)` computes the number of differing bit positions between two 64-bit fingerprints using bitwise XOR and hardware population count:
  $$\text{dist}(a, b) = \text{popcount}(a \oplus b)$$
- Output ranges from `0` (identical) to `64` (completely inverted).

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
# Run via CTest
ctest --test-dir build --output-on-failure

# Or run individual test suites directly
.\build\test_dataset_generator.exe
.\build\test_simhash.exe
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
  Total Documents          : 20
  Original Documents       : 10
  Exact Copies             : 4
  Modified Copies          : 4
  Unrelated Singletons     : 2
  Total Ground-Truth Groups: 12
  Duplicate Groups (size>1): 8
  Max Group Size           : 2
  Min Group Size           : 1
  Workload Profile         : Balanced
  Random Seed              : 42
---------------------------------------------------------

[Phase 2 Demonstration: SimHash Fingerprints]
Fingerprinting sample documents from generated dataset:
----------------------------------------------------------------------------------
 ID | Group | Kind          | 64-bit Hex Fingerprint | First 5 Words
----------------------------------------------------------------------------------
  0 |     1 | Original      | 0xda1259ecf93661af | pipeline resource cost strategy compiler...
  1 |     2 | Original      | 0x31b02534f0417760 | metric iteration pool library locality...
  2 |     3 | Original      | 0xa12f079bef003d1a | overhead multithreaded parallel buffer integer...
  3 |     4 | Original      | 0x7cf573ba9985b85a | load capacity speedup segment strategy...
  4 |     5 | Original      | 0x8fe3078803e5b1a5 | pipeline buffer multicore index response...
  5 |     6 | Original      | 0xfcb106c6bb73dad4 | deterministic database deadlock database contention...
  6 |     7 | Original      | 0xbe0eaf93a427e173 | library performance structure random flow...
  7 |     8 | Original      | 0xc624cb293bb30460 | random data timer computation data...
----------------------------------------------------------------------------------

Sample Hamming distance between Document 0 (Original) and Document 10 (ExactCopy): 0 bits differing

[Status] Execution completed successfully.
```
