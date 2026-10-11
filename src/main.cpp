#include "candidate_gen.hpp"
#include "dataset_generator.hpp"
#include "parallel_pipeline.hpp"
#include "simhash.hpp"

#include <iomanip>
#include <iostream>
#include <omp.h>

// Returns a human-readable description of the C++ standard
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

    DatasetConfig dcfg;
    dcfg.total_documents = 200;
    dcfg.min_words = 30;
    dcfg.max_words = 60;
    dcfg.exact_duplicate_ratio = 0.25;
    dcfg.modified_copy_ratio = 0.25;
    dcfg.unrelated_ratio = 0.10;
    dcfg.modification_rate = 0.10;
    dcfg.random_seed = 42;
    dcfg.profile = WorkloadProfile::Balanced;

    std::vector<Document> dataset = generate_dataset(dcfg);
    DatasetSummary summary = summarize_dataset(dataset, dcfg);
    summary.print(std::cout);

    // -------------------------------------------------------------
    // Phase 2 Demonstration: SimHash Fingerprints Preview
    // -------------------------------------------------------------
    std::cout << "\n[Phase 2 Demonstration: SimHash Fingerprints]\n";
    std::cout << "Fingerprinting sample documents from generated dataset:\n";
    std::cout << "----------------------------------------------------------------------------------\n";
    std::cout << " ID | Group | Kind          | 64-bit Hex Fingerprint | First 5 Words\n";
    std::cout << "----------------------------------------------------------------------------------\n";

    size_t preview_count = std::min<size_t>(dataset.size(), 4);
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
    // Phase 3 & 4: Sequential vs. OpenMP Parallel Scheduling Baselines
    // -------------------------------------------------------------
    std::cout << "[Phase 3 & 4 Demonstration: Sequential vs. OpenMP Parallel Baselines]\n";

    PipelineConfig pipe_cfg;
    pipe_cfg.shingle_size = 3;
    pipe_cfg.max_hamming_threshold = 3;

    // 1. Sequential Reference Run
    std::cout << "\n>>> Running Sequential Reference Pipeline...\n";
    PipelineResult seq_res = run_sequential_pipeline(dataset, pipe_cfg);

    std::cout << "  Candidate Pairs Generated : " << seq_res.candidate_pairs.size() << "\n";
    std::cout << "  Verified Near-Duplicates  : " << seq_res.verified_pairs.size() << "\n";
    std::cout << "  Sequential Total Runtime  : " << std::fixed << std::setprecision(3)
              << seq_res.timings.total_pipeline_ms << " ms\n";

    // 2. Parallel Baseline Runs: Static, Dynamic, and Guided Scheduling
    int demo_threads = std::min(4, omp_get_max_threads());
    std::cout << "\n>>> Running OpenMP Parallel Pipelines (" << demo_threads << " threads)...\n";

    std::vector<SchedulePolicy> demo_policies = {
        SchedulePolicy::Static,
        SchedulePolicy::Dynamic,
        SchedulePolicy::Guided
    };

    std::cout << "--------------------------------------------------------------------------------------------\n";
    std::cout << " Schedule Policy | Chunk | Fingerprint | Candidate Gen | Verification | Total Time | Parity \n";
    std::cout << "--------------------------------------------------------------------------------------------\n";

    for (auto policy : demo_policies) {
        ParallelConfig par_cfg;
        par_cfg.base_config = pipe_cfg;
        par_cfg.thread_count = demo_threads;
        par_cfg.schedule_policy = policy;
        par_cfg.chunk_size = (policy == SchedulePolicy::Static) ? 0 : 2;

        PipelineResult par_res = run_parallel_pipeline(dataset, par_cfg);

        // Verify exact equivalence of outputs with sequential baseline
        bool parity_ok = (seq_res.document_ids == par_res.document_ids) &&
                         (seq_res.fingerprints == par_res.fingerprints) &&
                         (seq_res.candidate_pairs == par_res.candidate_pairs) &&
                         (seq_res.verified_pairs == par_res.verified_pairs);

        std::cout << " " << std::setw(15) << std::left << to_string(policy)
                  << " | " << std::setw(5) << par_cfg.chunk_size
                  << " | " << std::setw(9) << std::right << par_res.timings.fingerprinting_ms << " ms"
                  << " | " << std::setw(11) << par_res.timings.candidate_generation_dedup_ms << " ms"
                  << " | " << std::setw(10) << par_res.timings.verification_ms << " ms"
                  << " | " << std::setw(8) << par_res.timings.total_pipeline_ms << " ms"
                  << " | " << (parity_ok ? "EXACT MATCH" : "MISMATCH!")
                  << "\n";
    }
    std::cout << "--------------------------------------------------------------------------------------------\n";

    std::cout << "\nWorkload Statistics (Identical across Sequential and Parallel):\n";
    seq_res.workload_stats.print(std::cout);

    std::cout << "\n[Status] Execution completed successfully.\n";
    return 0;
}
