# Skew-Aware Adaptive Parallel Near-Duplicate Detection on Multicore CPUs

## Overview

A high-performance C++ application designed to detect near-duplicate documents at scale and evaluate whether a workload-aware adaptive parallel execution strategy improves throughput and load balance over standard OpenMP scheduling when computational workloads are heavily skewed.

The system addresses the computational challenge where candidate document groups exhibit high variance in size: a naive parallel distribution across candidate buckets causes severe thread load imbalance, idle cores, and synchronization overheads.

---

## Key Components

1. **Near-Duplicate Detection Engine**:
   - **SimHash Fingerprinting**: Maps variable-length document tokens into 64-bit compact locality-sensitive fingerprints.
   - **Fingerprint Banding (Candidate Generation)**: Partitions fingerprints into sub-bands/tables to rapidly index and retrieve candidate matches without requiring all-pairs comparisons ($O(N^2)$).
   - **Verification**: Evaluates candidate pairs by computing exact bitwise Hamming distance against a configurable distance threshold.

2. **Adaptive Parallel Execution Engine**:
   - Distributes candidate pair comparisons across multicore CPU threads.
   - Estimates workload and computational cost across candidate buckets.
   - Dynamically adapts scheduling strategies (e.g., splitting high-density buckets and delegating small buckets) to alleviate tail latency and load skew.

---

## HPC Contribution & Research Focus

Standard OpenMP scheduling policies (`static`, `dynamic`, `guided`) make scheduling decisions at the loop or iteration chunk level without domain knowledge of workload skew within each iteration. 

In near-duplicate detection, candidate bucket sizes often follow power-law or highly skewed distributions (a few massive buckets containing hundreds or thousands of pairwise checks, alongside many tiny buckets). 

This project benchmarks and compares:
- **Serial Baseline**
- **OpenMP Static Scheduling**
- **OpenMP Dynamic Scheduling**
- **OpenMP Guided Scheduling**
- **Proposed Adaptive Workload-Aware Scheduling Policy** (cost-estimation, hierarchical distribution, and large-bucket splitting)

### Research Hypothesis
> *An adaptive, workload-aware scheduling policy can achieve superior core utilization, reduced tail latency, and higher parallel efficiency compared to conventional OpenMP loop scheduling under unevenly distributed candidate workloads.*

---

## Technology Stack

| Component | Specification |
| :--- | :--- |
| **Language** | C++17 |
| **Parallelization** | OpenMP |
| **Fingerprinting** | SimHash (64-bit) |
| **Candidate Verification** | Bitwise Hamming Distance (POPCNT / Bit manipulation) |
| **Dataset** | Synthetic & locally generated document corpuses with controlled skew |
| **Evaluation Metrics** | Execution time, Speedup, Parallel Efficiency, Load Balance / Core Utilization |
| **Target Platform** | Multicore CPU |

---

## Project Structure (Planned)

```text
├── .gitignore
├── README.md
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── simhash.hpp / .cpp
│   ├── candidate_gen.hpp / .cpp
│   ├── verification.hpp / .cpp
│   └── schedulers/
│       ├── openmp_schedulers.hpp
│       └── adaptive_scheduler.hpp
├── data/
│   └── generator/
├── benchmarks/
└── docs/
```

---

## Getting Started

*(Code implementation in progress. Build instructions and benchmark scripts will be added as modules are introduced.)*
