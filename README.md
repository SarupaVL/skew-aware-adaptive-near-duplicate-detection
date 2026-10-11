# Skew-Aware Adaptive Parallel Near-Duplicate Detection on Multicore CPUs

## Project Overview

This project investigates CPU-based near-duplicate document detection using SimHash locality-sensitive fingerprinting and OpenMP shared-memory parallelism. The primary research question is whether an adaptive, workload-aware task scheduling strategy can improve load balance and execution speed over standard OpenMP scheduling when candidate document comparisons are unevenly distributed across compute threads.

---

## Current Status: Phase 5 Completed

The project has completed:
- **Phase 0:** Environment inspection, C++17 validation, and OpenMP runtime verification.
- **Phase 1:** In-memory document data model, ground-truth duplicate cluster semantics, and deterministic synthetic dataset generator.
- **Phase 2:** 64-bit SimHash locality-sensitive fingerprinting, deterministic FNV-1a hashing, token shingling, and bitwise Hamming distance computation.
- **Phase 3:** Deterministic sequential near-duplicate detection pipeline: 8-band / 8-bit candidate generation, pair deduplication, Hamming-distance verification, bucket workload statistics, and stage timings.
- **Phase 4:** OpenMP parallelization and fixed scheduling baselines (`static`, `dynamic`, `guided`), race-free per-thread pair accumulation, deterministic output matching sequential ground truth, and configurable thread scaling.
- **Phase 5:** Workload-aware adaptive scheduling and heavy-bucket splitting into bounded pair-work chunks (`4,096` pairs default), adaptive diagnostics, non-overlapping stage timings, and correctness verification.

*(Systematic performance benchmarking experiments across dataset scales, skew levels, and thread counts are reserved for the next phase.)*

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
> The `Balanced` and `Skewed` profiles govern strictly the distribution and sizes of **ground-truth duplicate groups**. They do **not** dictate the downstream SimHash candidate bucket distribution. The actual bucket workloads and pairwise candidate comparison counts are explicitly measured by Phase 3's and Phase 4's workload statistics.

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

## Candidate Generation, Verification & Workload Statistics (Phase 3)

The candidate-generation and verification pipeline (`include/candidate_gen.hpp`, `src/candidate_gen.cpp`) provides a deterministic, single-threaded reference baseline that prunes comparisons using fingerprint banding and verifies candidate pairs against an exact Hamming-distance threshold.

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

---

## OpenMP Parallelization & Scheduling Baselines (Phase 4)

The OpenMP parallel pipeline (`include/parallel_pipeline.hpp`, `src/parallel_pipeline.cpp`) extends candidate detection across multicore CPU threads using standard loop-scheduling baselines while guaranteeing bitwise-identical output to the sequential reference implementation.

### 1. Parallel Pipeline Interface
```cpp
// include/parallel_pipeline.hpp
enum class SchedulePolicy { Static, Dynamic, Guided };

struct ParallelConfig {
    PipelineConfig base_config{};                           // Shingle size & Hamming threshold
    int thread_count{4};                                    // OpenMP thread count (> 0)
    SchedulePolicy schedule_policy{SchedulePolicy::Static}; // Static, Dynamic, or Guided
    int chunk_size{0};                                      // Chunk size (0 = OpenMP default)

    void validate() const;
};

PipelineResult run_parallel_pipeline(const std::vector<Document>& documents,
                                     const ParallelConfig& config = ParallelConfig{});
```

### 2. Parallelized vs. Sequential Stages
- **Fingerprinting (Parallel):** Preallocated vector storage `result.fingerprints[num_docs]`. Each OpenMP thread computes SimHash for its assigned document partition and writes directly to its disjoint output index without synchronization locks.
- **Bucket Construction & Workload Statistics (Sequential):** Kept sequential. Since there are only $8 \times 256 = 2048$ bucket slots, sequential table grouping runs in $< 0.2$ ms and completely avoids hash table lock contention or atomic insertion overhead.
- **Candidate Generation (Parallel):** Parallel loop iterates over active buckets ($k_i \ge 2$). To avoid critical section contention, each thread accumulates candidate pairs into dedicated per-thread buffer slots (`per_thread_candidates[tid]`) in a heap-allocated `std::vector` (indexed by thread ID, avoiding C++ `thread_local` storage duration overhead). After the parallel region, per-thread vectors are merged, sorted lexicographically, and deduplicated via `std::unique`.
- **Hamming-Distance Verification (Parallel):** Parallel loop distributes deduplicated candidate pairs across threads. Matching pairs are recorded in per-thread buffer slots (`per_thread_verified[tid]`), merged, and sorted lexicographically after the parallel region.

### 3. OpenMP Scheduling Policies Supported
- **`Static` Scheduling:** Work iterations divided evenly into contiguous chunks across threads before execution. Minimizes scheduling overhead, but sensitive to load imbalance under skewed bucket workloads.
- **`Dynamic` Scheduling:** Work divided into chunks of size `chunk_size` (0 = OpenMP default of 1) and assigned to idle threads dynamically from a shared work queue. Mitigates load imbalance at the cost of runtime synchronization.
- **`Guided` Scheduling:** Chunks start large and decrease exponentially down to `chunk_size` (0 = OpenMP default of 1), balancing runtime synchronization overhead with dynamic load balancing.
- **RAII Schedule Safety:** An RAII guard (`OmpScheduleGuard`) records prior OpenMP runtime schedule settings via `omp_get_schedule()`, applies the requested policy and chunk size via `omp_set_schedule()`, and restores the prior configuration on both normal return and exception unwinding, completely preventing scheduler state leaks.

### 4. Deterministic Output Guarantee
Because per-thread accumulation orders depend on dynamic thread scheduling, both candidate pairs and verified pairs are sorted lexicographically (`std::sort`) after parallel accumulation. **As a result, parallel execution produces bitwise-identical fingerprints, candidate pairs, verified pairs, and workload statistics to the sequential reference implementation across all thread counts and scheduling policies.**

> [!NOTE]
> **Performance Note on Small Workloads:**  
> Parallel speedup is not guaranteed for small document collections ($N < 500$) due to thread creation and synchronization overhead. Measurable speedup and load-balance trade-offs must be evaluated experimentally on larger workloads.

---

## Workload-Aware Adaptive Scheduling & Heavy-Bucket Splitting (Phase 5)

Phase 5 (`include/parallel_pipeline.hpp`, `src/parallel_pipeline.cpp`) introduces an explicit adaptive pipeline (`run_adaptive_pipeline`) alongside the existing sequential reference (`run_sequential_pipeline`) and fixed OpenMP scheduling baselines (`run_parallel_pipeline` with `Static`, `Dynamic`, or `Guided`).

### 1. Fixed Baselines vs. Adaptive Scheduling
- **Fixed Baselines (`run_parallel_pipeline`):** Apply a single predetermined OpenMP loop schedule (`Static`, `Dynamic`, or `Guided`) across all active buckets as indivisible loop iterations.
- **Adaptive Scheduling (`run_adaptive_pipeline`):** Inspects the actual SimHash bucket distribution at runtime (never relying on synthetic generator labels) to:
  1. Select `Static` vs. `Dynamic` scheduling for ordinary buckets based on the measured Coefficient of Variation ($\text{CV}$).
  2. Partition oversized ("heavy") buckets into bounded pair-work chunks so no single bucket monopolizes a thread.

### 2. Workload Measurement & Adaptive Decision Rule
For each occupied bucket $i$ containing $k_i$ documents, the estimated pairwise comparison work is:

$$W_i = \frac{k_i(k_i - 1)}{2}$$

Across all buckets with positive pair work ($k_i \ge 2$, $W_i > 0$), the pipeline computes the mean positive pair work $\overline{W}$ and the population standard deviation $\sigma$, yielding the **Coefficient of Variation**:

$$\text{CV} = \frac{\sigma}{\overline{W}}$$

(If no buckets have $k_i \ge 2$, $\overline{W} = 0.0$ and $\text{CV} = 0.0$.)

The baseline schedule for **ordinary buckets** is chosen by `select_ordinary_schedule_policy()`:
- If $\text{CV} \le \text{cv\_threshold}$ (default `1.0`): select **`SchedulePolicy::Static`**.
- If $\text{CV} > \text{cv\_threshold}$ (default `1.0`): select **`SchedulePolicy::Dynamic`**.
- `Guided` scheduling is never silently substituted by the adaptive rule; it remains strictly a fixed baseline for comparison.

### 3. Oversized-Bucket Identification & Heavy-Bucket Splitting
A positive-work bucket is classified as **oversized** when its pair work strictly exceeds a configurable multiple of the mean positive pair work:

$$W_i > \text{oversized\_bucket\_multiplier} \times \overline{W}$$

where `oversized_bucket_multiplier` defaults to **`4.0`** (`kDefaultOversizedMultiplier = 4.0`). If there are no positive-work buckets, no buckets are classified as oversized.

Each positive-work bucket is processed either as an **ordinary bucket** ($W_i \le 4.0 \times \overline{W}$) or as **split heavy-bucket work** ($W_i > 4.0 \times \overline{W}$), never both:
- **Ordinary buckets** are processed in parallel under the selected baseline schedule (`Static` or `Dynamic`).
- **Oversized buckets** are split by `split_bucket_into_chunks()` into compact `HeavyBucketChunk` descriptors containing at most **`max_pair_task_size` = 4,096** pair positions (`kDefaultMaxPairTaskSize = 4096`). For example, a bucket with $10,000$ pair positions and a limit of $4,096$ produces 3 chunks of sizes $4,096$, $4,096$, and $1,808$.
- All generated heavy-bucket chunks are processed in parallel using **OpenMP `Dynamic` scheduling** (`chunk_size = 1`).

### 4. Compact Pair-Position Mapping (`unrank_pair_position`)
To avoid materializing a large intermediate array of all pairs before scheduling, each `HeavyBucketChunk` stores only `(band_index, band_value, start_pair_pos, end_pair_pos)`.

Within a bucket of size $k$, unordered index pairs $(i, j)$ with $0 \le i < j < k$ are ordered lexicographically:
- Row $i \in [0, k - 2]$ contains $(k - 1 - i)$ pairs and starts at linear pair position:
  $$\text{row\_start}(i, k) = \sum_{r=0}^{i-1} (k - 1 - r) = \frac{i(2k - 1 - i)}{2}$$
- At the beginning of a chunk `[start_pair_pos, end_pair_pos)`, `unrank_pair_position(k, start_pair_pos)` uses integer binary search over $i \in [0, k - 2]$ to find the largest row with $\text{row\_start}(i, k) \le \text{start\_pair\_pos}$ and computes $j = (i + 1) + (\text{start\_pair\_pos} - \text{row\_start}(i, k))$.
- Subsequent pair positions within the chunk advance $(i, j)$ in $O(1)$ time (`++j; if (j >= k) { ++i; j = i + 1; }`).

### 5. Adaptive Configuration, Diagnostics & Non-Overlapping Timings
```cpp
// include/parallel_pipeline.hpp
struct AdaptiveConfig {
    PipelineConfig base_config{};                        // Shingle size & Hamming threshold
    int thread_count{4};                                 // Requested OpenMP threads (> 0)
    double cv_threshold{1.0};                            // Static (<=) vs Dynamic (>) threshold (>= 0, finite)
    double oversized_bucket_multiplier{4.0};             // Heavy-bucket multiplier (> 0, finite)
    uint64_t max_pair_task_size{4096};                   // Max pair positions per heavy chunk (> 0)
    int ordinary_chunk_size{0};                          // OpenMP chunk size for ordinary buckets (>= 0)

    void validate() const;
};

AdaptivePipelineResult run_adaptive_pipeline(const std::vector<Document>& documents,
                                             const AdaptiveConfig& config = AdaptiveConfig{});
```
- **Diagnostics (`AdaptiveDiagnostics`):** Reports `configured_thread_count` vs. measured `actual_thread_count`, `observed_cv`, `selected_ordinary_policy`, `decision_reason`, `occupied_buckets`, `positive_work_buckets`, `ordinary_buckets`, `oversized_buckets`, `oversized_pair_work`, `heavy_bucket_chunks`, and `max_generated_chunk_size`.
- **Stage Timings (`AdaptiveStageTimings`):** Records strictly non-overlapping wall-clock durations for (1) `fingerprinting_ms`, (2) `bucket_construction_stats_ms`, (3a) `adaptive_decision_ms`, (3b) `ordinary_candidate_gen_ms`, (3c) `heavy_chunk_generation_ms`, (3d) `heavy_pair_processing_ms`, (3e) `merge_sort_dedup_ms`, (4) `verification_ms`, and `total_pipeline_ms`.

> [!IMPORTANT]
> **Experimental Evaluation Scope:**  
> Implementing adaptive scheduling and heavy-bucket splitting establishes the mechanism and correctness parity with the sequential reference. Actual runtime and load-balance benefits compared with fixed `static`, `dynamic`, and `guided` baselines remain to be evaluated experimentally across dataset sizes, skew profiles, and thread counts in the next phase.

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
# Run all 5 test suites via CTest
ctest --test-dir build --output-on-failure
ctest --test-dir build-release --output-on-failure

# Or run individual test suites directly
.\build\test_dataset_generator.exe
.\build\test_simhash.exe
.\build\test_candidate_gen.exe
.\build\test_parallel_pipeline.exe
.\build\test_adaptive_pipeline.exe
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
  Total Documents          : 200
  Original Documents       : 80
  Exact Copies             : 50
  Modified Copies          : 50
  Unrelated Singletons     : 20
  Total Ground-Truth Groups: 100
  Duplicate Groups (size>1): 80
  Max Group Size           : 3
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
----------------------------------------------------------------------------------

[Phase 3, 4 & 5 Demonstration: Sequential, Fixed Baselines & Adaptive Scheduling]

>>> Running Sequential Reference Pipeline...
  Candidate Pairs Generated : 702
  Verified Near-Duplicates  : 50
  Sequential Total Runtime  : 3.249 ms

>>> Running OpenMP Parallel Pipelines (4 threads)...
--------------------------------------------------------------------------------------------
 Schedule Policy | Chunk | Fingerprint | Candidate Gen | Verification | Total Time | Parity 
--------------------------------------------------------------------------------------------
 Static          | 0     |     1.355 ms |       0.206 ms |      0.112 ms |    1.871 ms | EXACT MATCH
 Dynamic         | 2     |     1.016 ms |       0.281 ms |      0.144 ms |    1.680 ms | EXACT MATCH
 Guided          | 2     |     1.087 ms |       0.272 ms |      0.119 ms |    1.732 ms | EXACT MATCH
 Adaptive        | 4096  |     1.025 ms |       0.398 ms |      0.125 ms |    1.760 ms | EXACT MATCH
--------------------------------------------------------------------------------------------

Workload Statistics (Identical across Sequential, Fixed, and Adaptive):
---------------------------------------------------------
 SimHash Bucket Workload Statistics (8 bands x 8 bits)
---------------------------------------------------------
  Occupied Buckets (k >= 1)         : 893
  Buckets with Pair Work (k >= 2)   : 464
  Max Bucket Size                   : 7
  Mean Bucket Size (Occupied)       : 1.79
  Max Estimated Bucket Pair Work    : 21
  Mean Estimated Pair Work (k >= 2) : 2.33
  Coefficient of Variation (CV)     : 1.18
---------------------------------------------------------

[Phase 5 Adaptive Diagnostics & Stage Timings]
---------------------------------------------------------
 Adaptive Scheduling Diagnostics
---------------------------------------------------------
  Configured Thread Count           : 4
  Actual OpenMP Team Size           : 4
  Observed Bucket Pair-Work CV      : 1.1800 (Threshold: 1.0000)
  Selected Ordinary-Bucket Policy   : Dynamic
  Decision Reason                   : CV (1.1800) > threshold (1.0000) -> Dynamic schedule for ordinary buckets
  Occupied Buckets (k >= 1)         : 893
  Positive-Work Buckets (k >= 2)    : 464
  Mean Positive Pair Work (W_bar)   : 2.33
  Oversized Threshold (4.00 x W_bar) : 9.32
  Ordinary Positive-Work Buckets    : 440 (Pair Work: 798)
  Oversized Buckets                 : 24 (Pair Work: 283)
  Task-Size Limit (Pairs / Chunk)   : 4096
  Generated Heavy-Bucket Chunks     : 24
  Max Generated Chunk Size          : 21
---------------------------------------------------------
---------------------------------------------------------
 Adaptive Pipeline Stage Timings (Non-Overlapping)
---------------------------------------------------------
  1. Fingerprinting                 : 1.025 ms
  2. Bucket Construction & Stats    : 0.194 ms
  3a. Adaptive Decision & Classify  : 0.054 ms
  3b. Ordinary-Bucket Candidate Gen : 0.143 ms
  3c. Heavy-Bucket Chunk Generation : 0.004 ms
  3d. Heavy-Bucket Pair Processing  : 0.116 ms
  3e. Merge, Sort & Deduplication   : 0.080 ms
  4. Hamming Distance Verification  : 0.125 ms
  -------------------------------------------------------
  Total Adaptive Pipeline Runtime   : 1.760 ms
---------------------------------------------------------

[Status] Execution completed successfully.
```
