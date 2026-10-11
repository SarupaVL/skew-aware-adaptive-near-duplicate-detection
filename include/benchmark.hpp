#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include "candidate_gen.hpp"
#include "dataset_generator.hpp"
#include "parallel_pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

/**
 * @brief Execution policies evaluated in the Phase 6 benchmark suite.
 */
enum class BenchmarkPolicy {
    Sequential, ///< Single-threaded reference baseline (run_sequential_pipeline)
    Static,     ///< OpenMP fixed static scheduling (run_parallel_pipeline)
    Dynamic,    ///< OpenMP fixed dynamic scheduling (run_parallel_pipeline)
    Guided,     ///< OpenMP fixed guided scheduling (run_parallel_pipeline)
    Adaptive    ///< Workload-aware adaptive scheduling with heavy-bucket splitting (run_adaptive_pipeline)
};

/**
 * @brief Converts a BenchmarkPolicy enum value to its human-readable string label.
 */
inline const char* to_string(BenchmarkPolicy policy) noexcept {
    switch (policy) {
        case BenchmarkPolicy::Sequential: return "Sequential";
        case BenchmarkPolicy::Static:     return "Static";
        case BenchmarkPolicy::Dynamic:    return "Dynamic";
        case BenchmarkPolicy::Guided:     return "Guided";
        case BenchmarkPolicy::Adaptive:   return "Adaptive";
        default:                          return "Unknown";
    }
}

/**
 * @brief Per-thread estimated candidate-generation pair-work load balance metrics.
 *
 * Convention:
 * - Let P = max(configured_threads, actual_threads, per_thread_pair_work.size()) be the number
 *   of worker slots in the team. Any requested worker thread that receives 0 pair work is
 *   included with W_t = 0 when computing mean_thread_pair_work and min_thread_pair_work.
 * - participating_threads counts how many worker threads received W_t > 0.
 * - work_imbalance_ratio I = max_t(W_t) / mean_t(W_t) across all P worker slots
 *   (or 1.0 when total_pair_work == 0 or P <= 1).
 * - Note: I measures estimated pair-work imbalance across threads, NOT direct thread
 *   wall-clock execution-time imbalance.
 */
struct WorkBalanceMetrics {
    int configured_threads{1};
    int actual_threads{1};
    int participating_threads{0};
    uint64_t total_pair_work{0};
    uint64_t max_thread_pair_work{0};
    uint64_t min_thread_pair_work{0};
    double mean_thread_pair_work{0.0};
    double work_imbalance_ratio{1.0};
    size_t total_units_processed{0};
    std::string per_thread_work_distribution; ///< Semicolon-delimited "W0;W1;..."
};

/**
 * @brief Result of verifying exact output parity against the sequential reference run.
 */
struct ParityCheckResult {
    bool passed{true};
    std::string failure_reason;
};

/**
 * @brief Computes the median of a non-empty vector of numbers.
 * @throws std::invalid_argument if values is empty.
 */
double compute_median(std::vector<double> values);

/**
 * @brief Computes the arithmetic mean of a non-empty vector of numbers.
 * @throws std::invalid_argument if values is empty.
 */
double compute_mean(const std::vector<double>& values);

/**
 * @brief Computes the sample standard deviation (Bessel-corrected, N-1 denominator; 0.0 if N < 2).
 * @throws std::invalid_argument if values is empty.
 */
double compute_sample_sd(const std::vector<double>& values);

/**
 * @brief Computes the interquartile range (IQR = Q3 - Q1) using Tukey's median-of-halves method
 *        (or 0.0 if values has fewer than 2 elements).
 * @throws std::invalid_argument if values is empty.
 */
double compute_iqr(std::vector<double> values);

/**
 * @brief Computes parallel speedup S_p = T_seq / T_p.
 * @throws std::invalid_argument if sequential_ms < 0 or parallel_ms <= 0.
 */
double compute_speedup(double sequential_ms, double parallel_ms);

/**
 * @brief Computes parallel efficiency E_p = S_p / p.
 * @throws std::invalid_argument if thread_count <= 0 or speedup < 0.
 */
double compute_efficiency(double speedup, int thread_count);

/**
 * @brief Computes per-thread estimated pair-work balance metrics from per-thread work counts.
 */
WorkBalanceMetrics compute_work_balance_metrics(
    const std::vector<uint64_t>& per_thread_pair_work,
    const std::vector<size_t>& per_thread_units_processed,
    int configured_threads,
    int actual_threads
);

/**
 * @brief Identifies one benchmark configuration (policy + thread count) to execute on a dataset.
 */
struct BenchmarkConfigSpec {
    BenchmarkPolicy policy{BenchmarkPolicy::Sequential};
    int thread_count{1};

    bool operator==(const BenchmarkConfigSpec& other) const noexcept {
        return policy == other.policy && thread_count == other.thread_count;
    }

    bool operator!=(const BenchmarkConfigSpec& other) const noexcept {
        return !(*this == other);
    }
};

/**
 * @brief Builds the canonical list of configurations evaluated for a dataset:
 *        (Sequential, 1) followed by (Static, t), (Dynamic, t), (Guided, t), (Adaptive, t)
 *        for each thread count t in thread_counts.
 */
std::vector<BenchmarkConfigSpec> build_dataset_config_specs(
    const std::vector<int>& thread_counts
);

/**
 * @brief Produces a deterministic, reproducible permutation of base_specs for a given
 *        (order_seed, dataset_size, profile, round_index) using SplitMix64 seed mixing,
 *        std::mt19937_64, and Fisher-Yates shuffling.
 *
 * Rationale (Phase 7):
 * Running all repetitions of one policy consecutively before moving to the next policy
 * introduces execution-order bias from CPU turbo/thermal drift, cache state, and transient
 * background system load. Shuffling the configuration order within each repetition round
 * interleaves policies across the timeline while remaining 100% reproducible from order_seed.
 */
std::vector<BenchmarkConfigSpec> generate_shuffled_config_order(
    const std::vector<BenchmarkConfigSpec>& base_specs,
    uint64_t order_seed,
    size_t dataset_size,
    WorkloadProfile profile,
    size_t round_index
);

/**
 * @brief Formats a sequence of BenchmarkConfigSpec items into a semicolon-delimited
 *        schedule string (e.g., "Guided:4;Sequential:1;Adaptive:2;...").
 */
std::string format_config_order(const std::vector<BenchmarkConfigSpec>& specs);

/**
 * @brief Verifies exact correctness parity between a sequential reference PipelineResult
 *        and a candidate PipelineResult (parallel fixed or adaptive).
 */
ParityCheckResult verify_pipeline_parity(
    const PipelineResult& reference_seq,
    const PipelineResult& candidate_res
);

/**
 * @brief One raw measured repetition record in the benchmark suite.
 */
struct BenchmarkSampleRow {
    // Dataset configuration & parameters
    size_t dataset_size{0};
    WorkloadProfile profile{WorkloadProfile::Balanced};
    uint64_t random_seed{0};
    size_t min_words{0};
    size_t max_words{0};
    double exact_dup_ratio{0.0};
    double modified_copy_ratio{0.0};
    double unrelated_ratio{0.0};
    double modification_rate{0.0};
    size_t shingle_size{kDefaultShingleSize};
    int hamming_threshold{kDefaultHammingThreshold};

    // Execution configuration & deterministic order metadata (Phase 7)
    BenchmarkPolicy policy{BenchmarkPolicy::Sequential};
    int configured_threads{1};
    int actual_threads{1};
    int chunk_size{0};
    size_t repetition_index{0};
    uint64_t order_seed{0};
    size_t execution_order_index{0};   ///< Global 0-based chronological execution index across measured runs
    size_t round_execution_rank{0};    ///< 0-based execution rank within the repetition round for this dataset
    std::string round_execution_order; ///< Semicolon-delimited schedule for this round ("Policy:Threads;...")

    // Correctness & output counts
    bool parity_passed{true};
    size_t candidate_pairs_count{0};
    size_t verified_pairs_count{0};

    // Bucket workload statistics
    size_t occupied_buckets{0};
    size_t positive_work_buckets{0};
    size_t max_bucket_size{0};
    double mean_bucket_size{0.0};
    uint64_t total_estimated_pair_work{0};
    uint64_t max_estimated_pair_work{0};
    double mean_estimated_pair_work{0.0};
    double bucket_work_cv{0.0};

    // Adaptive diagnostics (0 / N/A for fixed policies)
    std::string selected_ordinary_policy;
    size_t ordinary_buckets{0};
    uint64_t ordinary_pair_work{0};
    size_t oversized_buckets{0};
    uint64_t oversized_pair_work{0};
    size_t heavy_bucket_chunks{0};
    uint64_t max_generated_chunk_size{0};

    // Per-thread estimated pair-work load balance
    int participating_threads{1};
    uint64_t max_thread_pair_work{0};
    uint64_t min_thread_pair_work{0};
    double mean_thread_pair_work{0.0};
    double work_imbalance_ratio{1.0};
    size_t total_units_processed{0};
    std::string per_thread_work_distribution;

    // Stage & total timings (milliseconds)
    double fingerprinting_ms{0.0};
    double bucket_construction_ms{0.0};
    double candidate_gen_total_ms{0.0};
    double adaptive_decision_ms{0.0};
    double ordinary_candidate_gen_ms{0.0};
    double heavy_chunk_gen_ms{0.0};
    double heavy_pair_proc_ms{0.0};
    double merge_sort_dedup_ms{0.0};
    double verification_ms{0.0};
    double total_pipeline_ms{0.0};
};

/**
 * @brief Aggregated summary record across all measured repetitions of a configuration.
 */
struct BenchmarkSummaryRow {
    // Dataset configuration & parameters
    size_t dataset_size{0};
    WorkloadProfile profile{WorkloadProfile::Balanced};
    uint64_t random_seed{0};
    size_t min_words{0};
    size_t max_words{0};
    double exact_dup_ratio{0.0};
    double modified_copy_ratio{0.0};
    double unrelated_ratio{0.0};
    double modification_rate{0.0};
    size_t shingle_size{kDefaultShingleSize};
    int hamming_threshold{kDefaultHammingThreshold};

    // Execution configuration
    BenchmarkPolicy policy{BenchmarkPolicy::Sequential};
    int configured_threads{1};
    int actual_threads{1};
    int chunk_size{0};
    size_t warmup_runs{0};
    size_t measured_runs{0};
    uint64_t order_seed{0};

    // Correctness & output counts
    bool parity_all_passed{true};
    size_t candidate_pairs_count{0};
    size_t verified_pairs_count{0};

    // Bucket workload statistics
    size_t occupied_buckets{0};
    size_t positive_work_buckets{0};
    size_t max_bucket_size{0};
    double mean_bucket_size{0.0};
    uint64_t total_estimated_pair_work{0};
    uint64_t max_estimated_pair_work{0};
    double mean_estimated_pair_work{0.0};
    double bucket_work_cv{0.0};

    // Adaptive diagnostics
    std::string selected_ordinary_policy;
    size_t ordinary_buckets{0};
    uint64_t ordinary_pair_work{0};
    size_t oversized_buckets{0};
    uint64_t oversized_pair_work{0};
    size_t heavy_bucket_chunks{0};
    uint64_t max_generated_chunk_size{0};

    // Per-thread estimated pair-work load balance (median/representative across repetitions)
    int participating_threads{1};
    uint64_t max_thread_pair_work{0};
    uint64_t min_thread_pair_work{0};
    double mean_thread_pair_work{0.0};
    double work_imbalance_ratio{1.0};
    size_t total_units_processed{0};
    std::string per_thread_work_distribution;

    // Aggregated stage median timings (ms)
    double median_fingerprinting_ms{0.0};
    double median_bucket_construction_ms{0.0};
    double median_candidate_gen_total_ms{0.0};
    double iqr_candidate_gen_total_ms{0.0};
    double sd_candidate_gen_total_ms{0.0};
    double median_adaptive_decision_ms{0.0};
    double median_ordinary_candidate_gen_ms{0.0};
    double median_heavy_chunk_gen_ms{0.0};
    double median_heavy_pair_proc_ms{0.0};
    double median_merge_sort_dedup_ms{0.0};
    double median_verification_ms{0.0};

    // Aggregated total pipeline timings (ms)
    double median_total_ms{0.0};
    double mean_total_ms{0.0};
    double iqr_total_ms{0.0};
    double sd_total_ms{0.0};
    double min_total_ms{0.0};
    double max_total_ms{0.0};

    // Speedup and parallel efficiency relative to Sequential median on the same dataset
    double speedup_total{1.0};
    double efficiency_total{1.0};
    double speedup_candidate_gen{1.0};
    double efficiency_candidate_gen{1.0};
};

/**
 * @brief Configuration for executing a reproducible benchmark suite.
 */
struct BenchmarkSuiteConfig {
    std::vector<size_t> dataset_sizes{500, 1000, 2000, 4000};
    std::vector<WorkloadProfile> profiles{WorkloadProfile::Balanced, WorkloadProfile::Skewed};
    std::vector<int> thread_counts{1, 2, 4, 8};

    // Dataset generation parameters
    uint64_t base_seed{42};
    size_t min_words{30};
    size_t max_words{60};
    double exact_duplicate_ratio{0.25};
    double modified_copy_ratio{0.25};
    double unrelated_ratio{0.10};
    double modification_rate{0.10};

    // Pipeline parameters
    PipelineConfig pipeline_config{};
    int fixed_chunk_size{0}; ///< 0 = OpenMP default chunk size

    // Adaptive pipeline parameters (kept at primary Phase 5 defaults)
    double cv_threshold{kDefaultCvThreshold};                         ///< 1.0
    double oversized_bucket_multiplier{kDefaultOversizedMultiplier};  ///< 4.0
    uint64_t max_pair_task_size{kDefaultMaxPairTaskSize};             ///< 4096
    int ordinary_chunk_size{0};

    // Repetition & execution-order parameters
    size_t warmup_runs{1};
    size_t measured_runs{5};
    uint64_t order_seed{1337ULL}; ///< Deterministic seed for policy execution-order randomization (Phase 7)
};

/**
 * @brief Complete output of a benchmark suite run.
 */
struct BenchmarkSuiteResult {
    std::vector<BenchmarkSampleRow> raw_samples;
    std::vector<BenchmarkSummaryRow> summary_rows;
};

/**
 * @brief Aggregates a non-empty list of BenchmarkSampleRow measurements for a single configuration
 *        into a BenchmarkSummaryRow, computing speedup and efficiency against seq_median_total_ms
 *        and seq_median_cand_ms.
 */
BenchmarkSummaryRow summarize_benchmark_samples(
    const std::vector<BenchmarkSampleRow>& samples,
    size_t warmup_runs,
    double seq_median_total_ms,
    double seq_median_cand_ms
);

/**
 * @brief Runs the complete benchmark suite according to suite_cfg.
 *        Throws std::runtime_error immediately if any run fails correctness parity against
 *        the sequential reference result.
 */
BenchmarkSuiteResult run_benchmark_suite(
    const BenchmarkSuiteConfig& suite_cfg,
    std::ostream* progress_os = nullptr
);

/**
 * @brief Writes raw per-repetition benchmark samples to a CSV stream.
 */
void write_raw_csv(std::ostream& os, const std::vector<BenchmarkSampleRow>& rows);

/**
 * @brief Writes aggregated benchmark summary rows to a CSV stream.
 */
void write_summary_csv(std::ostream& os, const std::vector<BenchmarkSummaryRow>& rows);

/**
 * @brief Prints a formatted human-readable benchmark summary table to an output stream.
 */
void print_summary_table(std::ostream& os, const std::vector<BenchmarkSummaryRow>& rows);

#endif // BENCHMARK_HPP
