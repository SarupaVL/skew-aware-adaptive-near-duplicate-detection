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
    suite_cfg.order_seed = 1337ULL;

    BenchmarkSuiteResult res = run_benchmark_suite(suite_cfg, nullptr);

    // Per (size, profile): 1 Sequential + 4 parallel policies * 3 thread counts = 13 configs
    // Across 2 profiles: 26 summary rows, 26 * 2 = 52 raw sample rows
    check(res.summary_rows.size() == 26, "test_suite_summary_row_count",
          "Expected 26 summary rows, got " + std::to_string(res.summary_rows.size()));
    check(res.raw_samples.size() == 52, "test_suite_raw_row_count",
          "Expected 52 raw sample rows, got " + std::to_string(res.raw_samples.size()));

    for (const auto& s : res.summary_rows) {
        check(s.parity_all_passed, "test_suite_all_parity_passed", "Every summary row must pass parity.");
        check(s.order_seed == 1337ULL, "test_suite_summary_order_seed", "Summary row must record order_seed.");
        check(std::isfinite(s.median_total_ms) && s.median_total_ms > 0.0,
              "test_suite_positive_runtime", "Median total time must be finite and positive.");
        check(s.work_imbalance_ratio >= 1.0 - 1e-9, "test_suite_imbalance_ge_1",
              "Work imbalance ratio I must be >= 1.0.");
    }

    std::ostringstream raw_csv;
    write_raw_csv(raw_csv, res.raw_samples);
    const std::string raw_str = raw_csv.str();
    check(raw_str.find("dataset_size,profile,random_seed") == 0,
          "test_raw_csv_header", "Raw CSV header missing or malformed.");
    check(raw_str.find("order_seed,execution_order_index,round_execution_rank,round_execution_order") != std::string::npos,
          "test_raw_csv_order_metadata_columns", "Raw CSV header must include Phase 7 order metadata columns.");

    std::ostringstream sum_csv;
    write_summary_csv(sum_csv, res.summary_rows);
    const std::string sum_str = sum_csv.str();
    check(sum_str.find("speedup_total,efficiency_total") != std::string::npos,
          "test_summary_csv_header", "Summary CSV header missing speedup/efficiency columns.");
    check(sum_str.find("warmup_runs,measured_runs,order_seed,parity_all_passed") != std::string::npos,
          "test_summary_csv_order_seed_column", "Summary CSV header must include order_seed column.");
}

void test_phase7_execution_order_and_timing_reliability() {
    const std::vector<int> threads = {1, 2, 4, 8};
    const std::vector<BenchmarkConfigSpec> base_specs = build_dataset_config_specs(threads);

    // 1 Sequential + 4 parallel policies * 4 thread counts = 17 configurations
    check(base_specs.size() == 17, "test_base_specs_count",
          "Expected 17 canonical configs for thread_counts={1,2,4,8}.");

    // 1. Deterministic order for a fixed seed
    auto order_a1 = generate_shuffled_config_order(
        base_specs, 1337ULL, 500, WorkloadProfile::Skewed, 0
    );
    auto order_a2 = generate_shuffled_config_order(
        base_specs, 1337ULL, 500, WorkloadProfile::Skewed, 0
    );
    check(order_a1 == order_a2, "test_shuffle_deterministic_same_seed",
          "Identical seed and round parameters must produce identical policy execution order.");

    // 2. Different seeds and different rounds produce different orders
    auto order_diff_seed = generate_shuffled_config_order(
        base_specs, 99999ULL, 500, WorkloadProfile::Skewed, 0
    );
    auto order_diff_round = generate_shuffled_config_order(
        base_specs, 1337ULL, 500, WorkloadProfile::Skewed, 1
    );
    check(order_a1 != order_diff_seed, "test_shuffle_different_seed",
          "Different order_seed must produce a different policy execution order.");
    check(order_a1 != order_diff_round, "test_shuffle_different_round",
          "Different repetition round index must produce a different policy execution order.");
    check(order_a1 != base_specs, "test_shuffle_not_canonical",
          "Shuffled order must differ from unshuffled canonical order.");

    // 3. Every randomized execution order includes all expected policies/configs exactly once
    for (const auto& candidate_order : {order_a1, order_diff_seed, order_diff_round}) {
        check(candidate_order.size() == base_specs.size(),
              "test_shuffle_preserves_size", "Shuffled order must have same size as base_specs.");
        for (const auto& expected_spec : base_specs) {
            size_t occurrences = 0;
            for (const auto& item : candidate_order) {
                if (item == expected_spec) {
                    ++occurrences;
                }
            }
            check(occurrences == 1, "test_shuffle_bijection",
                  "Every expected (policy, thread_count) configuration must appear exactly once per round.");
        }
    }

    // 4. End-to-end suite runs under two different order seeds on the same dataset
    BenchmarkSuiteConfig cfg_seed1;
    cfg_seed1.dataset_sizes = {50};
    cfg_seed1.profiles = {WorkloadProfile::Balanced, WorkloadProfile::Skewed};
    cfg_seed1.thread_counts = {1, 2, 4};
    cfg_seed1.warmup_runs = 1;
    cfg_seed1.measured_runs = 2;
    cfg_seed1.base_seed = 42ULL;
    cfg_seed1.order_seed = 101ULL;

    BenchmarkSuiteConfig cfg_seed2 = cfg_seed1;
    cfg_seed2.order_seed = 202ULL;

    BenchmarkSuiteResult res1 = run_benchmark_suite(cfg_seed1, nullptr);
    BenchmarkSuiteResult res2 = run_benchmark_suite(cfg_seed2, nullptr);

    check(res1.raw_samples.size() == res2.raw_samples.size(),
          "test_order_seed_raw_count_match", "Raw sample counts must match across order seeds.");

    // Verify global chronological execution_order_index and round_execution_rank metadata
    const size_t configs_per_round = 1 + 4 * cfg_seed1.thread_counts.size(); // 13
    for (size_t i = 0; i < res1.raw_samples.size(); ++i) {
        const auto& s = res1.raw_samples[i];
        check(s.execution_order_index == i, "test_chronological_exec_index",
              "execution_order_index must increment monotonically from 0.");
        check(s.round_execution_rank == (i % configs_per_round), "test_round_execution_rank",
              "round_execution_rank must match position within the repetition round.");
        check(!s.round_execution_order.empty(), "test_round_execution_order_nonempty",
              "round_execution_order string must be populated.");
        check(s.parity_passed, "test_raw_sample_parity",
              "Every raw sample must pass parity regardless of execution order.");

        // 5. Verify all stage and total timings are finite and non-negative
        const double timings_to_check[] = {
            s.fingerprinting_ms,
            s.bucket_construction_ms,
            s.candidate_gen_total_ms,
            s.adaptive_decision_ms,
            s.ordinary_candidate_gen_ms,
            s.heavy_chunk_gen_ms,
            s.heavy_pair_proc_ms,
            s.merge_sort_dedup_ms,
            s.verification_ms,
            s.total_pipeline_ms
        };
        for (double t_val : timings_to_check) {
            check(std::isfinite(t_val) && t_val >= 0.0,
                  "test_sample_timings_finite_nonnegative",
                  "All stage and total timings must be finite and non-negative.");
        }
        check(s.total_pipeline_ms > 0.0, "test_sample_total_timing_positive",
              "Total pipeline runtime must be strictly positive.");
    }

    // 6. Confirm that the two different order seeds actually executed configurations in a different order,
    //    while evaluating the exact same dataset and producing identical deterministic detection outputs.
    bool saw_different_chronological_policy = false;
    for (size_t i = 0; i < res1.raw_samples.size(); ++i) {
        if (res1.raw_samples[i].policy != res2.raw_samples[i].policy ||
            res1.raw_samples[i].configured_threads != res2.raw_samples[i].configured_threads) {
            saw_different_chronological_policy = true;
            break;
        }
    }
    check(saw_different_chronological_policy, "test_different_order_seeds_change_schedule",
          "Different order_seed values must change the actual chronological execution order.");

    check(res1.summary_rows.size() == res2.summary_rows.size(),
          "test_summary_rows_count_equal", "Summary row counts must match across order seeds.");
    for (size_t i = 0; i < res1.summary_rows.size(); ++i) {
        const auto& sum1 = res1.summary_rows[i];
        const auto& sum2 = res2.summary_rows[i];
        check(sum1.policy == sum2.policy && sum1.configured_threads == sum2.configured_threads,
              "test_summary_canonical_ordering", "Summary rows must preserve canonical (policy, threads) ordering.");
        check(sum1.candidate_pairs_count == sum2.candidate_pairs_count &&
              sum1.verified_pairs_count == sum2.verified_pairs_count &&
              sum1.occupied_buckets == sum2.occupied_buckets &&
              sum1.positive_work_buckets == sum2.positive_work_buckets &&
              sum1.max_bucket_size == sum2.max_bucket_size &&
              sum1.total_estimated_pair_work == sum2.total_estimated_pair_work,
              "test_identical_dataset_and_outputs_across_orders",
              "All policies must evaluate the exact same dataset and produce identical detection counts regardless of execution order.");
    }
}

void test_phase10_band_suppression_benchmark_comparison() {
    BenchmarkSuiteConfig base_cfg;
    base_cfg.dataset_sizes = {80};
    base_cfg.profiles = {WorkloadProfile::Balanced, WorkloadProfile::Skewed};
    base_cfg.thread_counts = {1, 2, 4};
    base_cfg.warmup_runs = 1;
    base_cfg.measured_runs = 2;
    base_cfg.base_seed = 42ULL;
    base_cfg.order_seed = 1337ULL;
    base_cfg.enable_band_suppression = false;

    BenchmarkSuiteConfig supp_cfg = base_cfg;
    supp_cfg.enable_band_suppression = true;
    supp_cfg.pipeline_config.enable_band_suppression = true;

    BenchmarkSuiteResult base_res = run_benchmark_suite(base_cfg, nullptr);
    BenchmarkSuiteResult supp_res = run_benchmark_suite(supp_cfg, nullptr);

    check(base_res.summary_rows.size() == supp_res.summary_rows.size(),
          "test_p10_bench_summary_size", "Baseline and suppression suites must produce identical row counts.");

    for (size_t i = 0; i < base_res.summary_rows.size(); ++i) {
        const auto& b = base_res.summary_rows[i];
        const auto& s = supp_res.summary_rows[i];

        check(!b.band_suppression_enabled && s.band_suppression_enabled,
              "test_p10_bench_flag_recorded", "Summary rows must record band_suppression_enabled flag.");
        check(b.parity_all_passed && s.parity_all_passed,
              "test_p10_bench_parity_passed", "All baseline and suppression benchmark runs must pass parity.");
        check(b.candidate_pairs_count == s.candidate_pairs_count &&
              b.verified_pairs_count == s.verified_pairs_count &&
              b.raw_candidates_evaluated == s.raw_candidates_evaluated &&
              b.total_estimated_pair_work == s.raw_candidates_evaluated,
              "test_p10_bench_exact_count_parity",
              "Evaluated, unique candidate, and verified pair counts must match identically.");
        check(b.raw_candidates_emitted == b.raw_candidates_evaluated && b.suppressed_pair_count == 0,
              "test_p10_bench_base_emits_all",
              "Baseline must emit all evaluated pairs with 0 suppressed.");
        check(s.raw_candidates_emitted == s.candidate_pairs_count &&
              s.raw_candidates_emitted + s.suppressed_pair_count == s.raw_candidates_evaluated &&
              s.suppressed_pair_count > 0,
              "test_p10_bench_supp_emits_unique_only",
              "Suppression must emit only unique pairs and conserve evaluated = emitted + suppressed.");
    }

    std::ostringstream raw_csv;
    write_raw_csv(raw_csv, supp_res.raw_samples);
    check(raw_csv.str().find("band_suppression_enabled,raw_candidates_evaluated,raw_candidates_emitted,suppressed_pair_count") != std::string::npos,
          "test_p10_raw_csv_columns", "Raw CSV must include Phase 10 suppression columns.");

    std::ostringstream sum_csv;
    write_summary_csv(sum_csv, supp_res.summary_rows);
    check(sum_csv.str().find("band_suppression_enabled,raw_candidates_evaluated,raw_candidates_emitted,suppressed_pair_count") != std::string::npos,
          "test_p10_summary_csv_columns", "Summary CSV must include Phase 10 suppression columns.");

    // Verify paired in-process comparison mode (generate_paired_config_order + compare_band_suppression)
    const std::vector<int> threads = {1, 2, 4};
    const std::vector<BenchmarkConfigSpec> base_specs = build_dataset_config_specs(threads); // 13 specs
    auto paired_r0 = generate_paired_config_order(base_specs, 1337ULL, 80, WorkloadProfile::Skewed, 0);
    auto paired_r1 = generate_paired_config_order(base_specs, 1337ULL, 80, WorkloadProfile::Skewed, 1);

    check(paired_r0.size() == 2 * base_specs.size(),
          "test_p10_paired_order_size", "Paired config order must have exactly 2x base_specs entries.");

    for (size_t k = 0; k < base_specs.size(); ++k) {
        const auto& first0 = paired_r0[2 * k];
        const auto& second0 = paired_r0[2 * k + 1];
        check(first0.policy == second0.policy && first0.thread_count == second0.thread_count &&
              first0.enable_band_suppression != second0.enable_band_suppression,
              "test_p10_paired_adjacent_variants",
              "Each adjacent pair in paired_r0 must have identical (policy, threads) and opposite suppression flags.");

        bool expected_first_supp_r0 = (((0 + k) & 1ULL) == 1ULL);
        bool expected_first_supp_r1 = (((1 + k) & 1ULL) == 1ULL);
        check(first0.enable_band_suppression == expected_first_supp_r0,
              "test_p10_paired_alternation_r0", "Round 0 rank alternation mismatch.");
        check(paired_r1[2 * k].enable_band_suppression == expected_first_supp_r1,
              "test_p10_paired_alternation_r1", "Round 1 rank alternation mismatch.");
    }

    BenchmarkSuiteConfig paired_cfg = base_cfg;
    paired_cfg.compare_band_suppression = true;
    BenchmarkSuiteResult paired_res = run_benchmark_suite(paired_cfg, nullptr);

    check(paired_res.summary_rows.size() == 2 * base_res.summary_rows.size(),
          "test_p10_paired_suite_summary_size", "Paired suite must produce 2x summary rows (Baseline + Suppression).");
    check(paired_res.raw_samples.size() == 2 * base_res.raw_samples.size(),
          "test_p10_paired_suite_raw_size", "Paired suite must produce 2x raw sample rows.");

    for (size_t k = 0; k < base_res.summary_rows.size(); ++k) {
        const auto& pb = paired_res.summary_rows[2 * k];
        const auto& ps = paired_res.summary_rows[2 * k + 1];
        check(!pb.band_suppression_enabled && ps.band_suppression_enabled,
              "test_p10_paired_summary_order", "Summary rows must alternate (Baseline, Suppression) per canonical spec.");
        check(pb.policy == ps.policy && pb.configured_threads == ps.configured_threads &&
              pb.dataset_size == ps.dataset_size && pb.profile == ps.profile,
              "test_p10_paired_summary_spec_match", "Paired summary rows must share dataset, profile, policy, and threads.");
        check(pb.parity_all_passed && ps.parity_all_passed &&
              pb.candidate_pairs_count == ps.candidate_pairs_count &&
              pb.verified_pairs_count == ps.verified_pairs_count &&
              pb.raw_candidates_evaluated == ps.raw_candidates_evaluated &&
              ps.raw_candidates_emitted == ps.candidate_pairs_count &&
              ps.raw_candidates_emitted + ps.suppressed_pair_count == ps.raw_candidates_evaluated,
              "test_p10_paired_summary_exact_parity",
              "Paired in-process runs must pass parity and conserve candidate counts.");
    }
}

} // namespace

int main() {
    std::cout << "========================================================\n";
    std::cout << " Running Phase 6, Phase 7 & Phase 10 Benchmark Unit Tests\n";
    std::cout << "========================================================\n";

    test_statistical_helpers();
    test_work_balance_metrics();
    test_parity_verification_and_instrumentation();
    test_mini_benchmark_suite_and_csv_export();
    test_phase7_execution_order_and_timing_reliability();
    test_phase10_band_suppression_benchmark_comparison();

    std::cout << "========================================================\n";
    std::cout << " All Phase 6, Phase 7 & Phase 10 Benchmark Tests PASSED!\n";
    std::cout << "========================================================\n";
    return 0;
}
