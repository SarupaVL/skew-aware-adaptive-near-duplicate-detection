#include "dataset_generator.hpp"
#include <iostream>
#include <omp.h>

// Returns a human-readable description of the C++ standard
// based on the standard __cplusplus preprocessor macro.
const char* get_cpp_standard_name(long standard_value) {
    if (standard_value >= 202002L) {
        return "C++20 or later";
    }
    if (standard_value >= 201703L) {
        return "C++17";
    }
    if (standard_value >= 201402L) {
        return "C++14";
    }
    if (standard_value >= 201103L) {
        return "C++11";
    }
    return "Pre-C++11";
}

int main() {
    std::cout << "===============================================================\n";
    std::cout << " Skew-Aware Adaptive Parallel Near-Duplicate Detection\n";
    std::cout << "===============================================================\n\n";

    // -------------------------------------------------------------
    // Phase 0 Checks: Compiler, C++ Standard & OpenMP Runtime
    // -------------------------------------------------------------
    std::cout << "[Phase 0 Verification]\n";
    std::cout << "  __cplusplus macro value : " << __cplusplus << "\n";
    std::cout << "  Active language standard: " << get_cpp_standard_name(__cplusplus) << "\n";
    std::cout << "  Logical processors      : " << omp_get_num_procs() << "\n";
    std::cout << "  Max OpenMP thread pool  : " << omp_get_max_threads() << "\n";

    int actual_team_size = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            actual_team_size = omp_get_num_threads();
            std::cout << "  OpenMP active team size : " << actual_team_size << " threads\n";
        }
    }
    std::cout << "  Phase 0 status          : PASS\n\n";

    // -------------------------------------------------------------
    // Phase 1 Demonstration: Synthetic Dataset Generation
    // -------------------------------------------------------------
    std::cout << "[Phase 1 Demonstration: Dataset Generation]\n";

    // 1. Generate a Balanced Dataset
    DatasetConfig balanced_cfg;
    balanced_cfg.total_documents = 100;
    balanced_cfg.min_words = 30;
    balanced_cfg.max_words = 60;
    balanced_cfg.exact_duplicate_ratio = 0.25;
    balanced_cfg.modified_copy_ratio = 0.25;
    balanced_cfg.unrelated_ratio = 0.10;
    balanced_cfg.modification_rate = 0.15;
    balanced_cfg.random_seed = 42;
    balanced_cfg.profile = WorkloadProfile::Balanced;

    std::vector<Document> balanced_dataset = generate_dataset(balanced_cfg);
    DatasetSummary balanced_summary = summarize_dataset(balanced_dataset, balanced_cfg);
    balanced_summary.print(std::cout);

    // 2. Generate a Skewed Dataset (same seed, skewed distribution)
    DatasetConfig skewed_cfg = balanced_cfg;
    skewed_cfg.profile = WorkloadProfile::Skewed;

    std::vector<Document> skewed_dataset = generate_dataset(skewed_cfg);
    DatasetSummary skewed_summary = summarize_dataset(skewed_dataset, skewed_cfg);
    skewed_summary.print(std::cout);

    std::cout << "\n[Status] Phase 1 execution completed successfully.\n";
    return 0;
}
