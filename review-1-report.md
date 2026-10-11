# Review 1: High Performance Computing Course Project

---

## Group Details

- **Group Number:** 29  
- **Group Members:**
  - Lakshmi Sarupa Venkadesh (`23BCE0463`)

---

## Project Title

**Skew-Aware Adaptive Parallel Near-Duplicate Detection on Multicore CPUs**

---

## Abstract

Near-duplicate detection is used to identify documents that contain the same or nearly the same content. Comparing every document against every other document becomes computationally prohibitive as the size of the collection increases. Fingerprinting methods can reduce the number of pairwise comparisons, but the resulting candidate groups often exhibit severe variance in workload. This leads to substantial load imbalance when the verification phase is parallelized naively across CPU cores.

This project proposes a CPU-parallel near-duplicate detection system using SimHash locality-sensitive fingerprinting and OpenMP. The system generates candidate document pairs using fingerprint banding and verifies them using Hamming distance. A sequential implementation serves as the baseline. Standard OpenMP parallel loop-scheduling configurations (static, dynamic, and guided) are compared against a proposed workload-aware adaptive strategy that estimates the computational cost of candidate groups and dynamically splits exceptionally expensive groups.

The implementation is developed in C++17 with OpenMP using synthetic, skew-controlled datasets. Experiments systematically evaluate dataset size, workload distribution skew, thread count, and scheduling configurations. Measured metrics include execution time, speedup, parallel efficiency, worker load balance, memory footprint, and detection accuracy. The investigation aims to determine whether workload-aware scheduling provides measurable performance advantages under uneven candidate-comparison workloads on a commodity multicore CPU.

---

## Methodology

The pipeline is implemented in C++17 using OpenMP for shared-memory parallelism.

```text
Input Documents ──> SimHash Fingerprinting ──> Band Partitioning & Candidate Generation ──> Skew-Aware Adaptive Parallel Verification (OpenMP) ──> Final Duplicate Pairs
```

### 1. Dataset Generation
A synthetic document generator creates collections consisting of original documents, exact duplicates, modified copies (with varying perturbation levels), and unrelated text. The generator supports tunable document lengths and controlled levels of workload skew (e.g., uniform vs. power-law/Zipfian distribution of candidate cluster sizes). A fixed pseudorandom seed is maintained to ensure strict experimental reproducibility.

### 2. Document Fingerprinting
Each document is converted into a compact 64-bit SimHash fingerprint using token shingles (n-grams) and weighted bit voting. These compact locality-sensitive fingerprints preserve cosine similarity in Hamming space.

### 3. Candidate Generation
To avoid exhaustive $O(N^2)$ pairwise comparisons, each 64-bit fingerprint is partitioned into eight 8-bit bands. Documents sharing an identical band index are mapped to common hash buckets and designated as candidate matches. Duplicate candidate pairs generated across multiple bands are deduplicated prior to verification.

The candidate-generation stage is calibrated against exhaustive all-pairs comparison on small control datasets to verify that candidate recall is preserved.

### 4. Similarity Verification
Candidate pairs are verified by computing the exact Hamming distance between their 64-bit fingerprints using bitwise XOR combined with hardware-accelerated population-count (`POPCNT`) instructions. Candidate pairs with a Hamming distance at or below a predefined threshold $k$ are classified as near-duplicates.

### 5. Sequential Baseline
A single-threaded, sequential verification pipeline is implemented first, serving as the ground-truth reference for both algorithmic correctness and baseline execution time.

### 6. Standard Parallel Implementation (OpenMP)
OpenMP is employed to parallelize candidate-pair verification across multicore CPU threads. Three standard loop-scheduling configurations are benchmarked:
- **Static Scheduling**: Fixed chunk distribution; minimal runtime scheduling overhead, but highly susceptible to severe thread load imbalance under skewed candidate clusters.
- **Dynamic Scheduling**: Work distributed on-demand in chunks; reduces idle time at the cost of runtime synchronization and task queue overhead.
- **Guided Scheduling**: Exponentially decreasing chunk sizes; aims to balance scheduling overhead with dynamic work allocation.

All scheduling modes execute across identical candidate pairs and verification logic to ensure that observed performance variations arise strictly from parallel execution and scheduling dynamics.

### 7. Proposed Workload-Aware Adaptive Scheduling
For a candidate bucket containing $k$ documents, the internal pairwise comparison workload is quadratic:
$$\text{Cost}(k) = \frac{k(k - 1)}{2}$$

When candidate bucket sizes follow a skewed distribution, a small number of oversized buckets dominate total compute time. The proposed adaptive framework:
1. **Cost Estimation**: Rapidly computes workload estimates across candidate groups based on bucket cardinalities prior to execution.
2. **Workload Partitioning**: Categorizes candidate groups into light, moderate, and heavyweight tasks.
3. **Adaptive Decomposition**: Decomposes exceptionally large candidate clusters into balanced pairwise sub-tasks rather than treating each bucket as an atomic work unit.
4. **Dynamic Assignment**: Assigns decomposed tasks to OpenMP threads to maintain balanced core utilization throughout execution.

Its performance is assessed strictly through experimental validation against standard OpenMP policies.

### 8. Experimental Evaluation
The experimental testbed varies:
- **Dataset Size ($N$)**: Number of documents ($10^3$ to $10^5+$).
- **Workload Skew**: Skewness factor across candidate-group sizes.
- **Core / Thread Scaling**: Execution across 1, 2, 4, 8, and maximum available CPU threads.
- **Scheduling Strategies**: Sequential, OpenMP Static, OpenMP Dynamic, OpenMP Guided, and Proposed Adaptive.
- **Chunk Sizes**: Systematic sweep across power-of-two chunk sizes.

#### Performance Metrics:
- **Execution Time ($T_p$)**: Wall-clock time averaged across repeated runs.
- **Speedup ($S_p$)**: $S_p = \frac{T_1}{T_p}$
- **Parallel Efficiency ($E_p$)**: $E_p = \frac{S_p}{p}$
- **Load Imbalance Factor**: Variation in per-thread compute time and idle latency.
- **Memory Footprint**: Peak resident set size (RSS).
- **Quality Metrics**: Candidate-generation recall, final precision, and final recall.

---

## Literature Survey

### 1. On the Resemblance and Containment of Documents
> **Broder [1]** introduced the mathematical foundation for document resemblance and containment using min-wise independent permutations (MinHash) and random sampling. This seminal work established the paradigm of identifying similar documents without performing exhaustive all-pairs comparisons.

### 2. Similarity Estimation Techniques from Rounding Algorithms
> **Charikar [2]** formulated locality-sensitive hashing techniques based on random hyperplane rounding, creating **SimHash**. This theoretical foundation proves that the angle between two high-dimensional vectors is proportional to the Hamming distance between their compact bitwise fingerprints.

### 3. LSH Forest: Self-Tuning Indexes for Similarity Search
> **Bawa, Condie, and Ganesan [3]** proposed the LSH Forest indexing scheme, demonstrating how locality-sensitive hashing scales to variable similarity thresholds without manual tuning, and highlighting the performance impact of skewed data distributions on similarity lookups.

### 4. Probabilistic Near-Duplicate Detection Using SimHash
> **Sood and Loguinov [4]** conducted a rigorous performance and probabilistic analysis of SimHash for multi-billion-page web corpora. Their study demonstrated that while SimHash provides robust candidate pruning, downstream candidate verification is computationally intensive.

### 5. Near-Duplicate Detection in an Academic Digital Library
> **Williams and Giles [5]** evaluated SimHash alongside shingle-based deduplication across the CiteSeerX digital library, examining trade-offs between fingerprint compactness, candidate filtering efficiency, and resulting F-score accuracy.

### 6. A Scalable Parallel Deduplication Algorithm
> **Santos et al. [6]** developed parallel deduplication algorithms using probabilistic record linkage on multicore architectures. Their work demonstrated that irregular candidate cluster sizes introduce thread imbalance, highlighting the necessity of workload partitioning.

### 7. Efficient String Similarity Join in Multi-Core and Distributed Systems
> **Yan et al. [7]** analyzed parallel string-similarity joins across multicore and distributed architectures, showing that partitioning input spaces according to frequency and interval distributions alleviates severe workload skew in parallel join pipelines.

### 8. Parallel Loop Scheduling for High Performance Computers
> **Yue and Lilja [8]** evaluated scheduling overhead and load-balancing trade-offs among static, dynamic, and self-scheduling loop strategies in high-performance computing, emphasizing that no single fixed scheduling policy remains optimal across varying workload distributions.

### 9. Accelerating Content-Defined-Chunking-Based Data Deduplication by Exploiting Parallelism
> **Xia et al. [9]** developed P-Dedupe, a parallel deduplication pipeline demonstrating that fingerprinting and chunk verification bottlenecks require decoupled stage parallelism and dynamic task distribution to achieve high throughput on multicore systems.

### 10. Parallel Near-Duplicate Document Detection Using General-Purpose GPU
> **Peshevski, Zdraveski, and Ristov [10]** implemented a GPU-accelerated MinHash near-duplicate document detection pipeline, establishing the effectiveness of fine-grained data-parallel acceleration for fingerprint-based similarity search.

### 11. Automatic Near-Duplicate Content Document Detection in a Cancer Registry
> **Niemi et al. [11]** combined SimHash fingerprinting with Smith-Waterman sequence alignment to detect near-duplicate clinical records within real-world cancer registries, validating that compact fingerprint indexing combined with rigorous pairwise verification provides high fidelity in domain-specific text corpora.

---

## Novelty in the Proposed Work

Existing literature has extensively addressed document fingerprinting algorithms, similarity indexing structures, and general parallel loop scheduling. However, these techniques typically address either algorithmic filtering or parallel scheduling in isolation:

1. **Workload-Aware Group Decomposition**: Standard OpenMP scheduling schedules loop iterations blindly without inspecting the internal work complexity $\frac{k(k-1)}{2}$ of each candidate group.
2. **Mitigation of Extreme Skew on Multicore CPUs**: In near-duplicate detection, candidate clusters naturally exhibit high variance—a few large clusters contain tens of thousands of comparisons while most contain fewer than five. Naive loop scheduling leads to thread stalling and delayed barrier synchronization.
3. **Adaptive Scheduling Framework**: This project connects candidate-bucket structural properties directly with OpenMP scheduling decisions, decomposing high-cost buckets while assigning light buckets efficiently.
4. **Empirical Performance Boundary Mapping**: The project will experimentally delineate the exact skew thresholds and corpus sizes where adaptive workload-aware scheduling outperforms standard OpenMP policies versus where its overhead outweighs its benefit.

---

## Deliverables & Expected Outcomes

- **Course Deliverable**: A comprehensive, reproducible research paper prepared in **IEEE conference format**, reporting methodology, mathematical formulations, hardware configuration, speedup, efficiency, and load balance analysis.
- **Codebase**: A clean, modular C++17 / OpenMP codebase containing synthetic skew data generation, SimHash indexing, multiple parallel scheduler implementations, and automated benchmarking scripts.
- **Publication Potential**: If the adaptive scheduler demonstrates statistically significant gains in parallel efficiency and load balance under high skew, the findings will be prepared for submission to an HPC or applied computing conference.

---

## References

- **[1]** A. Z. Broder, “On the resemblance and containment of documents,” in *Proc. Compression and Complexity of Sequences*, Positano, Italy, 1997, pp. 21–29, doi: [10.1109/SEQUEN.1997.666900](https://doi.org/10.1109/SEQUEN.1997.666900).
- **[2]** M. S. Charikar, “Similarity estimation techniques from rounding algorithms,” in *Proc. 34th Annu. ACM Symp. Theory of Computing (STOC)*, Montreal, QC, Canada, 2002, pp. 380–388, doi: [10.1145/509907.509965](https://doi.org/10.1145/509907.509965).
- **[3]** M. Bawa, T. Condie, and P. Ganesan, “LSH forest: Self-tuning indexes for similarity search,” in *Proc. 14th Int. Conf. World Wide Web (WWW)*, Chiba, Japan, 2005, pp. 651–660, doi: [10.1145/1060745.1060840](https://doi.org/10.1145/1060745.1060840).
- **[4]** S. Sood and D. Loguinov, “Probabilistic near-duplicate detection using simhash,” in *Proc. 20th ACM Int. Conf. Information and Knowledge Management (CIKM)*, Glasgow, UK, 2011, pp. 1117–1126, doi: [10.1145/2063576.2063737](https://doi.org/10.1145/2063576.2063737).
- **[5]** K. Williams and C. L. Giles, “Near duplicate detection in an academic digital library,” in *Proc. ACM Symp. Document Engineering (DocEng)*, Florence, Italy, 2013, pp. 91–94, doi: [10.1145/2494266.2494312](https://doi.org/10.1145/2494266.2494312).
- **[6]** W. Santos, T. Teixeira, C. Machado, W. Meira Jr., A. S. da Silva, R. Ferreira, and D. Guedes, “A Scalable Parallel Deduplication Algorithm,” in *Proc. 19th Int. Symp. Computer Architecture and High Performance Computing (SBAC-PAD)*, Gramado, Brazil, 2007, pp. 79–86, doi: [10.1109/SBAC-PAD.2007.32](https://doi.org/10.1109/SBAC-PAD.2007.32).
- **[7]** C. Yan, X. Zhao, Q. Zhang, and Y. Huang, “Efficient string similarity join in multi-core and distributed systems,” *PLoS ONE*, vol. 12, no. 3, Art. no. e0172526, Mar. 2017, doi: [10.1371/journal.pone.0172526](https://doi.org/10.1371/journal.pone.0172526).
- **[8]** K. K. Yue and D. J. Lilja, “Parallel loop scheduling for high performance computers,” *Advances in Parallel Computing*, Elsevier, vol. 10, pp. 243–264, 1995, doi: [10.1016/S0927-5452(06)80016-X](https://doi.org/10.1016/S0927-5452(06)80016-X).
- **[9]** W. Xia, D. Feng, H. Jiang, Y. Zhang, V. Chang, and X. Zou, “Accelerating content-defined-chunking based data deduplication by exploiting parallelism,” *Future Generation Computer Systems*, vol. 98, pp. 406–418, Sep. 2019, doi: [10.1016/j.future.2019.02.008](https://doi.org/10.1016/j.future.2019.02.008).
- **[10]** D. Peshevski, V. Zdraveski, and S. Ristov, “Parallel near-duplicate document detection using general-purpose GPU,” *Computing and Informatics*, vol. 43, no. 3, pp. 583–610, 2024, doi: [10.31577/cai_2024_3_583](https://doi.org/10.31577/cai_2024_3_583).
- **[11]** T. Niemi, J. P. Ghobril, G. Defossez, S. Germann, E. Martin, and J.-L. Bulliard, “Automatic (near-) duplicate content document detection in a cancer registry,” *International Journal of Medical Informatics*, vol. 195, Art. no. 105799, Mar. 2025, doi: [10.1016/j.ijmedinf.2025.105799](https://doi.org/10.1016/j.ijmedinf.2025.105799).
