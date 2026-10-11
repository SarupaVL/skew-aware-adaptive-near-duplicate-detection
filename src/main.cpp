#include "candidate_gen.hpp"
#include "dataset_generator.hpp"
#include "simhash.hpp"
#include <iomanip>
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

    DatasetConfig balanced_cfg;
    balanced_cfg.total_documents = 100;
    balanced_cfg.min_words = 30;
    balanced_cfg.max_words = 60;
    balanced_cfg.exact_duplicate_ratio = 0.20;
    balanced_cfg.modified_copy_ratio = 0.20;
    balanced_cfg.unrelated_ratio = 0.10;
    balanced_cfg.modification_rate = 0.10;
    balanced_cfg.random_seed = 42;
    balanced_cfg.profile = WorkloadProfile::Balanced;

    std::vector<Document> dataset = generate_dataset(balanced_cfg);
    DatasetSummary summary = summarize_dataset(dataset, balanced_cfg);
    summary.print(std::cout);

    // -------------------------------------------------------------
    // Phase 2 Demonstration: SimHash Fingerprints Preview
    // -------------------------------------------------------------
    std::cout << "\n[Phase 2 Demonstration: SimHash Fingerprints]\n";
    std::cout << "Fingerprinting sample documents from generated dataset:\n";
    std::cout << "----------------------------------------------------------------------------------\n";
    std::cout << " ID | Group | Kind          | 64-bit Hex Fingerprint | First 5 Words\n";
    std::cout << "----------------------------------------------------------------------------------\n";

    size_t preview_count = std::min<size_t>(dataset.size(), 6);
    for (size_t i = 0; i < preview_count; ++i) {
        const auto& doc = dataset[i];
        uint64_t fp = compute_simhash(doc.text);

        auto tokens = tokenize_text(doc.text);
        std::string preview;
        for (size_t t = 0; t < std::min<size_t>(tokens.size(), 5); ++t) {
            if (t > 0) preview += ' ';
            preview += tokens[t];
        }
        if (tokens.size() > 5) preview += "...";

        std::cout << " " << std::setw(2) << doc.id << " | "
                  << std::setw(5) << doc.group_id << " | "
                  << std::setw(13) << std::left << to_string(doc.kind) << std::right << " | "
                  << "0x" << std::hex << std::setw(16) << std::setfill('0') << fp
                  << std::dec << std::setfill(' ') << " | "
                  << preview << "\n";
    }
    std::cout << "----------------------------------------------------------------------------------\n\n";

    // -------------------------------------------------------------
    // Phase 3 Demonstration: Sequential Candidate Gen & Verification
    // -------------------------------------------------------------
    std::cout << "[Phase 3 Demonstration: Sequential Pipeline Execution]\n";

    PipelineConfig pipe_cfg;
    pipe_cfg.shingle_size = 3;
    pipe_cfg.max_hamming_threshold = 3;

    PipelineResult pipe_result = run_sequential_pipeline(dataset, pipe_cfg);

    std::cout << "Pipeline Summary:\n";
    std::cout << "  Input Documents Processed : " << pipe_result.document_ids.size() << "\n";
    std::cout << "  Total Candidate Pairs     : " << pipe_result.candidate_pairs.size() << "\n";
    std::cout << "  Verified Near-Duplicate Pairs : " << pipe_result.verified_pairs.size() << "\n\n";

    pipe_result.workload_stats.print(std::cout);
    pipe_result.timings.print(std::cout);

    std::cout << "\nSample Verified Near-Duplicates (Threshold <= " << pipe_cfg.max_hamming_threshold << "):\n";
    size_t sample_show = std::min<size_t>(pipe_result.verified_pairs.size(), 5);
    for (size_t i = 0; i < sample_show; ++i) {
        const auto& vp = pipe_result.verified_pairs[i];
        std::cout << "  Pair (Doc " << vp.first << ", Doc " << vp.second
                  << ") -> Hamming Distance: " << vp.distance << " bits\n";
    }

    std::cout << "\n[Status] Execution completed successfully.\n";
    return 0;
}
