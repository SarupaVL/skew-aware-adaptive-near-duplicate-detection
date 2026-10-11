# Skew-Aware Adaptive Parallel Near-Duplicate Detection on Multicore CPUs

## Project Overview

This project investigates CPU-based near-duplicate document detection using SimHash locality-sensitive fingerprinting and OpenMP shared-memory parallelism. The primary research question is whether an adaptive, workload-aware task scheduling strategy can improve load balance and execution speed over standard OpenMP scheduling when candidate document comparisons are unevenly distributed across compute threads.

---

## Current Status: Phase 3 Completed

The project has completed:
- **Phase 0:** Environment inspection, C++17 validation, and OpenMP runtime verification.
- **Phase 1:** In-memory document data model, ground-truth duplicate cluster semantics, and deterministic synthetic dataset generator.
- **Phase 2:** 64-bit SimHash locality-sensitive fingerprinting, deterministic FNV-1a hashing, token shingling, and bitwise Hamming distance computation.
- **Phase 3:** Deterministic sequential near-duplicate detection pipeline: 8-band / 8-bit candidate generation, pair deduplication, Hamming-distance verification, bucket workload statistics, and stage timings.

*(Parallel execution—OpenMP static, dynamic, guided, and adaptive scheduling—is part of subsequent phases and is not yet implemented. The pipeline at this stage is strictly sequential to establish the correctness baseline.)*

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
> The `Balanced` and `Skewed` profiles govern strictly the distribution and sizes of **ground-truth duplicate groups**. They do **not** dictate the downstream SimHash candidate bucket distribution. The actual bucket workloads and pairwise candidate comparison counts are explicitly measured by Phase 3's workload statistics.

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

## Sequential Candidate Generation & Verification (Phase 3)

The candidate-generation and verification pipeline (`include/candidate_gen.hpp`, `src/candidate_gen.cpp`) provides a deterministic, single-threaded baseline that prunes comparisons using fingerprint banding and verifies candidate pairs against an exact Hamming-distance threshold.

### 1. 8-Band, 8-Bit Fingerprint Partitioning
- Each 64-bit fingerprint is partitioned into **8 non-overlapping bands of 8 bits each**:
  - Band 0: bits 0..7
  - Band 1: bits 8..15
  - ...
  - Band 7: bits 56..63
- A candidate bucket is uniquely keyed by **`(band_index, band_value)`** where `band_index` $\in [0, 7]$ and `band_value` $\in [0, 255]$ (256 buckets per band; 2048 total possible buckets across all bands).
- Two documents are mapped into the same bucket if and only if they share the exact same 8-bit value at that specific band index. Identical 8-bit values occurring in different band indices are strictly separate buckets.

### 2. Candidate Pair Generation & Deduplication
- Within each bucket containing $k \ge 2$ documents, all unordered pairs $(u, v)$ are generated.
- Each pair is normalized such that the smaller document ID comes first ($u < v$).
- Because near-duplicate documents often match across multiple bands, candidate pairs are collected across all bands, sorted lexicographically, and deduplicated.
- **Deterministic ordering:** Deduplicated pairs are sorted so that candidate output is bitwise reproducible.

### 3. Exact Hamming-Distance Verification
- Every unique candidate pair $(u, v)$ is verified by computing $\text{dist}(\text{fp}_u, \text{fp}_v)$.
- A candidate pair is retained as a detected near-duplicate if:
  $$\text{dist}(\text{fp}_u, \text{fp}_v) \le \text{max\_hamming\_threshold}$$
- **Configurable threshold:** Default is **3 bits** (`kDefaultHammingThreshold = 3`).
- Candidate pairs exceeding the threshold are discarded as false positives.

### 4. Essential Correctness Guarantee (Recall Guarantee for Distance $\le 7$)
> **Mathematical Property (Pigeonhole Principle):**  
> With 8 bands of 8 bits across 64 bits, any two fingerprints differing by at most 7 bits ($\text{dist} \le 7$) must contain at least one band with 0 differing bits.  
> Consequently, **every document pair with Hamming distance $\le 7$ is guaranteed to share at least one identical band and appear in the candidate set (100% candidate recall).**

This property is verified by an automated test comparing pipeline candidates against exhaustive all-pairs ($O(N^2)$) checks. For any threshold $\le 7$ (including the default threshold of 3), the candidate-verification pipeline produces the exact same near-duplicate pairs as exhaustive all-pairs verification.

### 5. Workload Statistics Definitions
To support subsequent OpenMP scheduling analysis, the pipeline records:
- **Occupied Buckets:** Count of buckets containing $k_i \ge 1$ documents across all 8 bands.
- **Buckets with Pair Work:** Count of buckets containing $k_i \ge 2$ documents.
- **Estimated Pair Work ($W_i$):** $W_i = \frac{k_i(k_i - 1)}{2}$ pairwise comparisons for bucket $i$.
- **Mean Bucket Size:** $\frac{\sum k_i}{\text{Occupied Buckets}}$.
- **Mean Estimated Pair Work:** Mean of $W_i$ across buckets with $k_i \ge 2$.
- **Coefficient of Variation (CV):** $\text{CV} = \frac{\sigma}{\mu}$, where $\sigma$ is the population standard deviation of $W_i$ and $\mu$ is the mean of $W_i$ across buckets with $k_i \ge 2$. If no buckets have $k_i \ge 2$, $\text{CV} = 0.0$.

### 6. Stage Timings
Stage runtimes are measured in milliseconds using `std::chrono::high_resolution_clock`:
1. **Fingerprinting:** SimHash calculation across all documents.
2. **Bucket Construction:** Band extraction and bucket assignment.
3. **Candidate Gen & Deduplication:** Pair expansion within buckets and unique sorting.
4. **Hamming Distance Verification:** Pairwise XOR population count on candidate pairs.
5. **Total Pipeline Runtime:** End-to-end wall-clock time.

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
# Run all tests via CTest
ctest --test-dir build --output-on-failure

# Or run individual test suites directly
.\build\test_dataset_generator.exe
.\build\test_simhash.exe
.\build\test_candidate_gen.exe
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
  Original Documents       : 50
  Exact Copies             : 20
  Modified Copies          : 20
  Unrelated Singletons     : 10
  Total Ground-Truth Groups: 60
  Duplicate Groups (size>1): 40
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
  0 |     1 | Original      | 0xda965b2df926c1ef | pipeline resource cost strategy compiler...
  1 |     2 | Original      | 0x21a8e5b0e0217776 | lock topology overhead partition operation...
  2 |     3 | Original      | 0x4127038eff417d0b | scheduling dynamic compiler static memory...
  3 |     4 | Original      | 0x7ee5533a1895b04e | size array reproducibility cache deterministic...
  4 |     5 | Original      | 0xfdf306ccb767d2d5 | fault cache performance operating decomposition...
  5 |     6 | Original      | 0xac04ab3984a7c422 | random flow decomposition efficiency instruction...
----------------------------------------------------------------------------------

[Phase 3 Demonstration: Sequential Pipeline Execution]
Pipeline Summary:
  Input Documents Processed : 100
  Total Candidate Pairs     : 179
  Verified Near-Duplicate Pairs : 20

---------------------------------------------------------
 SimHash Bucket Workload Statistics (8 bands x 8 bits)
---------------------------------------------------------
  Occupied Buckets (k >= 1)         : 547
  Buckets with Pair Work (k >= 2)   : 202
  Max Bucket Size                   : 6
  Mean Bucket Size (Occupied)       : 1.46
  Max Estimated Bucket Pair Work    : 15
  Mean Estimated Pair Work (k >= 2) : 1.59
  Coefficient of Variation (CV)     : 1.02
---------------------------------------------------------
---------------------------------------------------------
 Sequential Pipeline Stage Timings
---------------------------------------------------------
  1. Fingerprinting                 : 1.349 ms
  2. Bucket Construction            : 0.133 ms
  3. Candidate Gen & Deduplication  : 0.028 ms
  4. Hamming Distance Verification  : 0.005 ms
  -------------------------------------------------------
  Total Pipeline Runtime            : 1.516 ms
---------------------------------------------------------

Sample Verified Near-Duplicates (Threshold <= 3):
  Pair (Doc 0, Doc 50) -> Hamming Distance: 0 bits
  Pair (Doc 1, Doc 51) -> Hamming Distance: 0 bits
  Pair (Doc 2, Doc 52) -> Hamming Distance: 0 bits
  Pair (Doc 3, Doc 53) -> Hamming Distance: 0 bits
  Pair (Doc 4, Doc 54) -> Hamming Distance: 0 bits

[Status] Execution completed successfully.
```
