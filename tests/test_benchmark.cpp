#include "benchmark.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& test_name, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << test_name << ": " << message << "\n";
        std::exit(1);
    }
    std::cout << "[PASS] " << test_name << "\n";
}

void test_statistical_helpers() {
    // 1. Median (odd and even counts, unsorted input)
    std::vector<double> odd_vals = {9.0, 1.0, 5.0, 3.0, 7.0};
    check(std::abs(compute_median(odd_vals) - 5.0) < 1e-9,
          "test_median_odd", "Median of {1,3,5,7,9} must be 5.0.");

    std::vector<double> even_vals = {10.0, 2.0, 8.0, 4.0};
    check(std::abs(compute_median(even_vals) - 6.0) < 1e-9,
          "test_median_even", "Median of {2,4,8,10} must be 6.0.");

    check(std::abs(compute_median({42.5}) - 42.5) < 1e-9,
          "test_median_single", "Median of single element must be itself.");

    // 2. Mean and Sample SD
    // Sample {2, 4, 4, 4, 5, 5, 7, 9}: mean = 5.0, sum sq diffs = 32, sample var = 32/7, sample SD = sqrt(32/7)
    std::vector<double> dist_vals = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    check(std::abs(compute_mean(dist_vals) - 5.0) < 1e-9,
          "test_mean", "Mean of sample must be 5.0.");
    double expected_sd = std::sqrt(32.0 / 7.0);
    check(std::abs(compute_sample_sd(dist_vals) - expected_sd) < 1e-9,
          "test_sample_sd", "Sample SD mismatch.");
    check(std::abs(compute_sample_sd({10.0}) - 0.0) < 1e-9,
          "test_sample_sd_single", "Sample SD of 1 element must be 0.0.");

    // 3. Interquartile Range (IQR)
    // Odd 5 elements: {10, 20, 30, 40, 50} -> lower half {10, 20} (Q1=15), upper half {40, 50} (Q3=45) -> IQR = 30
    std::vector<double> iqr_odd = {50.0, 10.0, 30.0, 20.0, 40.0};
    check(std::abs(compute_iqr(iqr_odd) - 30.0) < 1e-9,
          "test_iqr_odd", "IQR of {10,20,30,40,50} must be 30.0.");
    check(std::abs(compute_iqr({7.0}) - 0.0) < 1e-9,
          "test_iqr_single", "IQR of 1 element must be 0.0.");

    // 4. Speedup and Efficiency
    double sp = compute_speedup(100.0, 25.0);
    check(std::abs(sp - 4.0) < 1e-9, "test_speedup", "Speedup 100/25 must be 4.0.");
    double eff = compute_efficiency(sp, 8);
    check(std::abs(eff - 0.5) < 1e-9, "test_efficiency", "Efficiency 4.0/8 must be 0.5.");

    // 5. Validation exceptions
    bool threw = false;
    try { compute_median({}); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "test_median_empty_throws", "Empty vector must throw.");

    threw = false;
    try { compute_speedup(10.0, 0.0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "test_speedup_zero_throws", "Zero parallel_ms must throw.");

    threw = false;
    try { compute_efficiency(2.0, 0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "test_efficiency_zero_threads_throws", "Zero thread_count must throw.");
}

void test_work_balance_metrics() {
    // 1. Ideal balance across 4 threads
    WorkBalanceMetrics wb_ideal = compute_work_balance_metrics(
        {250, 250, 250, 250}, {5, 5, 5, 5}, 4, 4
    );
    check(wb_ideal.participating_threads == 4, "test_wb_ideal_part", "All 4 threads participated.");
    check(wb_ideal.total_pair_work == 1000, "test_wb_ideal_total", "Total work must be 1000.");
    check(std::abs(wb_ideal.mean_thread_pair_work - 250.0) < 1e-9, "test_wb_ideal_mean", "Mean work must be 250.");
    check(std::abs(wb_ideal.work_imbalance_ratio - 1.0) < 1e-9, "test_wb_ideal_ratio", "Ideal imbalance I must be 1.0.");
    check(wb_ideal.per_thread_work_distribution == "250;250;250;250",
          "test_wb_ideal_dist", "Work distribution string mismatch.");

    // 2. Extreme skew where 1 thread gets all work and 7 requested threads get 0 work
    WorkBalanceMetrics wb_skew = compute_work_balance_metrics(
        {800, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, 8, 8
    );
    check(wb_skew.participating_threads == 1, "test_wb_skew_part", "Only 1 thread participated.");
    check(wb_skew.max_thread_pair_work == 800, "test_wb_skew_max", "Max work must be 800.");
    check(wb_skew.min_thread_pair_work == 0, "test_wb_skew_min", "Min work must be 0.");
    check(std::abs(wb_skew.mean_thread_pair_work - 100.0) < 1e-9,
          "test_wb_skew_mean", "Mean work across all 8 configured threads must be 100.0.");
    check(std::abs(wb_skew.work_imbalance_ratio - 8.0) < 1e-9,
          "test_wb_skew_ratio", "Imbalance I on 1/8 active threads must be 8.0.");

    // 3. Zero total work edge case
    WorkBalanceMetrics wb_zero = compute_work_balance_metrics(
        {0, 0, 0, 0}, {0, 0, 0, 0}, 4, 4
    );
    check(wb_zero.participating_threads == 0, "test_wb_zero_part", "0 threads participated.");
    check(std::abs(wb_zero.work_imbalance_ratio - 1.0) < 1e-9,
          "test_wb_zero_ratio", "Zero total work must report I = 1.0.");
}

void test_parity_verification_and_instrumentation() {
    DatasetConfig dcfg;
    dcfg.total_documents = 120;
    dcfg.min_words = 25;
    dcfg.max_words = 50;
    dcfg.exact_duplicate_ratio = 0.25;
    dcfg.modified_copy_ratio = 0.25;
    dcfg.unrelated_ratio = 0.10;
    dcfg.modification_rate = 0.10;
    dcfg.random_seed = 42;
    dcfg.profile = WorkloadProfile::Skewed;

    std::vector<Document> docs = generate_dataset(dcfg);
    PipelineConfig pcfg;

    PipelineResult seq_res = run_sequential_pipeline(docs, pcfg);

    ParallelConfig par_cfg;
    par_cfg.base_config = pcfg;
    par_cfg.thread_count = 4;
    par_cfg.schedule_policy = SchedulePolicy::Static;
    PipelineResult par_res = run_parallel_pipeline(docs, par_cfg);

    AdaptiveConfig adap_cfg;
    adap_cfg.base_config = pcfg;
    adap_cfg.thread_count = 4;
    AdaptivePipelineResult adap_res = run_adaptive_pipeline(docs, adap_cfg);

    ParityCheckResult p1 = verify_pipeline_parity(seq_res, par_res);
    check(p1.passed, "test_parity_static_passes", "Static pipeline must pass parity: " + p1.failure_reason);

    ParityCheckResult p2 = verify_pipeline_parity(seq_res, adap_res.pipeline_result);
    check(p2.passed, "test_parity_adaptive_passes", "Adaptive pipeline must pass parity: " + p2.failure_reason);

    // Mutate candidate_res to verify failure detection
    PipelineResult mutated = par_res;
    mutated.fingerprints[0] ^= 0x1ULL;
    check(!verify_pipeline_parity(seq_res, mutated).passed,
          "test_parity_detects_fp_mismatch", "Parity checker must detect fingerprint mutation.");

    mutated = par_res;
    mutated.per_thread_pair_work[0] += 1ULL;
    check(!verify_pipeline_parity(seq_res, mutated).passed,
          "test_parity_detects_work_sum_mismatch", "Parity checker must detect non-conserved thread work.");
}

void test_mini_benchmark_suite_and_csv_export() {
    BenchmarkSuiteConfig suite_cfg;
    suite_cfg.dataset_sizes = {60};
    suite_cfg.profiles = {WorkloadProfile::Balanced, WorkloadProfile::Skewed};
    suite_cfg.thread_counts = {1, 2, 4};
    suite_cfg.warmup_runs = 1;
    suite_cfg.measured_runs = 2;

    BenchmarkSuiteResult res = run_benchmark_suite(suite_cfg, nullptr);

    // Per (size, profile): 1 Sequential + 4 parallel policies * 3 thread counts = 13 configs
    // Across 2 profiles: 26 summary rows, 26 * 2 = 52 raw sample rows
    check(res.summary_rows.size() == 26, "test_suite_summary_row_count",
          "Expected 26 summary rows, got " + std::to_string(res.summary_rows.size()));
    check(res.raw_samples.size() == 52, "test_suite_raw_row_count",
          "Expected 52 raw sample rows, got " + std::to_string(res.raw_samples.size()));

    for (const auto& s : res.summary_rows) {
        check(s.parity_all_passed, "test_suite_all_parity_passed", "Every summary row must pass parity.");
        check(s.median_total_ms > 0.0, "test_suite_positive_runtime", "Median total time must be positive.");
        check(s.work_imbalance_ratio >= 1.0 - 1e-9, "test_suite_imbalance_ge_1",
              "Work imbalance ratio I must be >= 1.0.");
    }

    std::ostringstream raw_csv;
    write_raw_csv(raw_csv, res.raw_samples);
    check(raw_csv.str().find("dataset_size,profile,random_seed") == 0,
          "test_raw_csv_header", "Raw CSV header missing or malformed.");

    std::ostringstream sum_csv;
    write_summary_csv(sum_csv, res.summary_rows);
    check(sum_csv.str().find("speedup_total,efficiency_total") != std::string::npos,
          "test_summary_csv_header", "Summary CSV header missing speedup/efficiency columns.");
}

} // namespace

int main() {
    std::cout << "========================================================\n";
    std::cout << " Running Phase 6 Benchmark Unit Tests\n";
    std::cout << "========================================================\n";

    test_statistical_helpers();
    test_work_balance_metrics();
    test_parity_verification_and_instrumentation();
    test_mini_benchmark_suite_and_csv_export();

    std::cout << "========================================================\n";
    std::cout << " All Phase 6 Benchmark Tests PASSED!\n";
    std::cout << "========================================================\n";
    return 0;
}
