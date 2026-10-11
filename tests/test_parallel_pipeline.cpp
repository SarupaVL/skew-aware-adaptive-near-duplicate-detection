#include "candidate_gen.hpp"
#include "dataset_generator.hpp"
#include "parallel_pipeline.hpp"

#include <cmath>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

void check(bool condition, const std::string& test_name, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << test_name << ": " << message << "\n";
        std::exit(1);
    }
    std::cout << "[PASS] " << test_name << "\n";
}

// Helper to assert bitwise equivalence between sequential and parallel results
void assert_pipeline_results_equal(const PipelineResult& seq,
                                  const PipelineResult& par,
                                  const std::string& context) {
    // 1. Document IDs
    check(seq.document_ids == par.document_ids, context + "_doc_ids",
          "Document IDs mismatch between sequential and parallel.");

    // 2. Fingerprints
    check(seq.fingerprints == par.fingerprints, context + "_fingerprints",
          "Fingerprints mismatch between sequential and parallel.");

    // 3. Candidate Pairs (Normalized, Deduplicated, Sorted)
    check(seq.candidate_pairs.size() == par.candidate_pairs.size(), context + "_candidate_size",
          "Candidate pair count mismatch: seq=" + std::to_string(seq.candidate_pairs.size()) +
          " par=" + std::to_string(par.candidate_pairs.size()));
    check(seq.candidate_pairs == par.candidate_pairs, context + "_candidate_pairs",
          "Candidate pairs mismatch between sequential and parallel.");

    // 4. Verified Pairs (Distance <= Threshold, Sorted)
    check(seq.verified_pairs.size() == par.verified_pairs.size(), context + "_verified_size",
          "Verified pair count mismatch: seq=" + std::to_string(seq.verified_pairs.size()) +
          " par=" + std::to_string(par.verified_pairs.size()));
    check(seq.verified_pairs == par.verified_pairs, context + "_verified_pairs",
          "Verified pairs mismatch between sequential and parallel.");

    // 5. Workload Statistics
    const auto& s_ws = seq.workload_stats;
    const auto& p_ws = par.workload_stats;
    check(s_ws.occupied_buckets == p_ws.occupied_buckets, context + "_ws_occ",
          "Occupied buckets count mismatch.");
    check(s_ws.positive_work_buckets == p_ws.positive_work_buckets, context + "_ws_pos_work",
          "Positive work buckets count mismatch.");
    check(s_ws.max_bucket_size == p_ws.max_bucket_size, context + "_ws_max_size",
          "Max bucket size mismatch.");
    check(s_ws.max_estimated_pair_work == p_ws.max_estimated_pair_work, context + "_ws_max_work",
          "Max estimated pair work mismatch.");
    check(std::abs(s_ws.mean_bucket_size - p_ws.mean_bucket_size) < 1e-6, context + "_ws_mean_size",
          "Mean bucket size mismatch.");
    check(std::abs(s_ws.mean_estimated_pair_work - p_ws.mean_estimated_pair_work) < 1e-6, context + "_ws_mean_work",
          "Mean estimated pair work mismatch.");
    check(std::abs(s_ws.cv_estimated_pair_work - p_ws.cv_estimated_pair_work) < 1e-6, context + "_ws_cv",
          "CV of estimated pair work mismatch.");
}

void test_empty_and_single_document() {
    ParallelConfig pcfg;
    pcfg.thread_count = 4;

    // 1. Empty input
    PipelineResult seq_empty = run_sequential_pipeline({});
    PipelineResult par_empty = run_parallel_pipeline({}, pcfg);
    assert_pipeline_results_equal(seq_empty, par_empty, "test_empty_input");

    // 2. Single document
    Document doc{42, 1, DocumentKind::Original, -1, "multicore cpu openmp scheduling"};
    PipelineResult seq_single = run_sequential_pipeline({doc});
    PipelineResult par_single = run_parallel_pipeline({doc}, pcfg);
    assert_pipeline_results_equal(seq_single, par_single, "test_single_document");
}

void test_balanced_and_skewed_equivalence() {
    // Test dataset with Balanced profile
    DatasetConfig dcfg_bal;
    dcfg_bal.total_documents = 60;
    dcfg_bal.min_words = 20;
    dcfg_bal.max_words = 50;
    dcfg_bal.exact_duplicate_ratio = 0.25;
    dcfg_bal.modified_copy_ratio = 0.25;
    dcfg_bal.unrelated_ratio = 0.10;
    dcfg_bal.random_seed = 101;
    dcfg_bal.profile = WorkloadProfile::Balanced;
    auto docs_bal = generate_dataset(dcfg_bal);

    PipelineResult seq_bal = run_sequential_pipeline(docs_bal);
    ParallelConfig pcfg_bal;
    pcfg_bal.thread_count = 4;
    pcfg_bal.schedule_policy = SchedulePolicy::Static;
    PipelineResult par_bal = run_parallel_pipeline(docs_bal, pcfg_bal);
    assert_pipeline_results_equal(seq_bal, par_bal, "test_balanced_dataset");

    // Test dataset with Skewed profile
    DatasetConfig dcfg_skew = dcfg_bal;
    dcfg_skew.profile = WorkloadProfile::Skewed;
    dcfg_skew.random_seed = 202;
    auto docs_skew = generate_dataset(dcfg_skew);

    PipelineResult seq_skew = run_sequential_pipeline(docs_skew);
    ParallelConfig pcfg_skew;
    pcfg_skew.thread_count = 4;
    pcfg_skew.schedule_policy = SchedulePolicy::Dynamic;
    PipelineResult par_skew = run_parallel_pipeline(docs_skew, pcfg_skew);
    assert_pipeline_results_equal(seq_skew, par_skew, "test_skewed_dataset");
}

void test_non_contiguous_and_reordered_ids() {
    Document doc1{50000, 1, DocumentKind::Original, -1, "concurrency thread scheduling load balance"};
    Document doc2{10, 1, DocumentKind::ExactCopy, 50000, "concurrency thread scheduling load balance"};
    Document doc3{999999, 2, DocumentKind::Unrelated, -1, "network database transaction isolation level"};

    std::vector<Document> docs_orig = {doc1, doc2, doc3};
    std::vector<Document> docs_perm = {doc3, doc1, doc2};

    ParallelConfig pcfg;
    pcfg.thread_count = 2;

    PipelineResult seq_orig = run_sequential_pipeline(docs_orig);
    PipelineResult par_orig = run_parallel_pipeline(docs_orig, pcfg);
    assert_pipeline_results_equal(seq_orig, par_orig, "test_non_contig_orig");

    PipelineResult par_perm = run_parallel_pipeline(docs_perm, pcfg);
    check(par_orig.candidate_pairs == par_perm.candidate_pairs, "test_par_reorder_cand",
          "Reordered input must yield identical candidate pairs in parallel pipeline.");
    check(par_orig.verified_pairs == par_perm.verified_pairs, "test_par_reorder_ver",
          "Reordered input must yield identical verified pairs in parallel pipeline.");
}

void test_duplicate_document_ids_rejected() {
    Document doc1{100, 1, DocumentKind::Original, -1, "parallel loop"};
    Document doc2{100, 1, DocumentKind::ExactCopy, 100, "parallel loop"}; // Duplicate ID: 100

    bool caught = false;
    try {
        ParallelConfig pcfg;
        pcfg.thread_count = 2;
        run_parallel_pipeline({doc1, doc2}, pcfg);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    check(caught, "test_par_duplicate_ids_rejected",
          "Parallel pipeline must reject duplicate document IDs with std::invalid_argument.");
}

void test_multiple_candidates_and_threshold_boundaries() {
    // Generate dataset with varied modification levels
    DatasetConfig dcfg;
    dcfg.total_documents = 50;
    dcfg.exact_duplicate_ratio = 0.30;
    dcfg.modified_copy_ratio = 0.30;
    dcfg.modification_rate = 0.08;
    dcfg.random_seed = 303;
    auto docs = generate_dataset(dcfg);

    // Test across different threshold boundaries (0, 1, 3, 5)
    for (int threshold : {0, 1, 3, 5}) {
        PipelineConfig pipe_cfg;
        pipe_cfg.max_hamming_threshold = threshold;

        PipelineResult seq = run_sequential_pipeline(docs, pipe_cfg);

        ParallelConfig par_cfg;
        par_cfg.base_config = pipe_cfg;
        par_cfg.thread_count = 4;
        PipelineResult par = run_parallel_pipeline(docs, par_cfg);

        assert_pipeline_results_equal(seq, par, "test_threshold_" + std::to_string(threshold));
    }
}

void test_scheduling_policies_and_thread_counts() {
    DatasetConfig dcfg;
    dcfg.total_documents = 80;
    dcfg.exact_duplicate_ratio = 0.25;
    dcfg.modified_copy_ratio = 0.25;
    dcfg.random_seed = 404;
    dcfg.profile = WorkloadProfile::Skewed; // High workload variation across buckets
    auto docs = generate_dataset(dcfg);

    PipelineResult seq_ref = run_sequential_pipeline(docs);

    std::vector<SchedulePolicy> policies = {
        SchedulePolicy::Static,
        SchedulePolicy::Dynamic,
        SchedulePolicy::Guided
    };

    std::vector<int> thread_counts = {1, 2, 4};
    int max_avail = omp_get_max_threads();
    if (max_avail >= 8) {
        thread_counts.push_back(8);
    }

    for (auto policy : policies) {
        for (int threads : thread_counts) {
            ParallelConfig pcfg;
            pcfg.thread_count = threads;
            pcfg.schedule_policy = policy;
            pcfg.chunk_size = (policy == SchedulePolicy::Static) ? 0 : 2;

            PipelineResult par = run_parallel_pipeline(docs, pcfg);
            std::string label = std::string("test_") + to_string(policy) + "_t" + std::to_string(threads);
            assert_pipeline_results_equal(seq_ref, par, label);
        }
    }
}

void test_repeated_runs_determinism() {
    DatasetConfig dcfg;
    dcfg.total_documents = 50;
    dcfg.exact_duplicate_ratio = 0.25;
    dcfg.modified_copy_ratio = 0.25;
    dcfg.random_seed = 505;
    auto docs = generate_dataset(dcfg);

    ParallelConfig pcfg;
    pcfg.thread_count = 4;
    pcfg.schedule_policy = SchedulePolicy::Dynamic;
    pcfg.chunk_size = 1;

    PipelineResult first_run = run_parallel_pipeline(docs, pcfg);

    // Repeat 5 times to detect race conditions or nondeterministic thread ordering
    for (int trial = 1; trial <= 5; ++trial) {
        PipelineResult next_run = run_parallel_pipeline(docs, pcfg);
        check(first_run.candidate_pairs == next_run.candidate_pairs,
              "test_repeat_candidates_trial_" + std::to_string(trial),
              "Parallel candidate pairs must be identical across repeated executions.");
        check(first_run.verified_pairs == next_run.verified_pairs,
              "test_repeat_verified_trial_" + std::to_string(trial),
              "Parallel verified pairs must be identical across repeated executions.");
    }
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << " Running Phase 4 Test Suite: OpenMP Parallel Baselines\n";
    std::cout << "=========================================================\n";

    test_empty_and_single_document();
    test_balanced_and_skewed_equivalence();
    test_non_contiguous_and_reordered_ids();
    test_duplicate_document_ids_rejected();
    test_multiple_candidates_and_threshold_boundaries();
    test_scheduling_policies_and_thread_counts();
    test_repeated_runs_determinism();

    std::cout << "\n=========================================================\n";
    std::cout << " All Phase 4 Parallel Pipeline Tests Passed Successfully!\n";
    std::cout << "=========================================================\n";
    return 0;
}
