#ifndef PARALLEL_PIPELINE_HPP
#define PARALLEL_PIPELINE_HPP

#include "candidate_gen.hpp"
#include <omp.h>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Default parameters for the workload-aware adaptive scheduling pipeline (Phase 5).
 */
inline constexpr double kDefaultCvThreshold = 1.0;          ///< CV <= 1.0 -> Static; CV > 1.0 -> Dynamic
inline constexpr double kDefaultOversizedMultiplier = 4.0;  ///< W_i > 4.0 * mean_positive_work -> Oversized
inline constexpr uint64_t kDefaultMaxPairTaskSize = 4096;   ///< Max candidate pair positions per heavy-bucket chunk

/**
 * @brief Supported OpenMP loop scheduling policies for baseline parallel execution.
 */
enum class SchedulePolicy {
    Static,
    Dynamic,
    Guided
};

/**
 * @brief Converts SchedulePolicy to human-readable string.
 */
inline const char* to_string(SchedulePolicy policy) {
    switch (policy) {
        case SchedulePolicy::Static:  return "Static";
        case SchedulePolicy::Dynamic: return "Dynamic";
        case SchedulePolicy::Guided:  return "Guided";
        default:                      return "Unknown";
    }
}

namespace detail {

/**
 * @brief RAII guard to safely set and restore OpenMP runtime loop schedule.
 *
 * When using `schedule(runtime)`, OpenMP reads the schedule set by `omp_set_schedule()`.
 * This guard records the prior schedule and restores it upon scope exit (normal return
 * or exception unwinding), preventing schedule configuration from leaking into unrelated work.
 */
struct OmpScheduleGuard {
    omp_sched_t orig_sched{omp_sched_static};
    int orig_modifier{0};

    OmpScheduleGuard(SchedulePolicy policy, int chunk_size);
    ~OmpScheduleGuard();

    OmpScheduleGuard(const OmpScheduleGuard&) = delete;
    OmpScheduleGuard& operator=(const OmpScheduleGuard&) = delete;
    OmpScheduleGuard(OmpScheduleGuard&&) = delete;
    OmpScheduleGuard& operator=(OmpScheduleGuard&&) = delete;
};

} // namespace detail

/**
 * @brief Configuration parameters for OpenMP parallel pipeline execution.
 */
struct ParallelConfig {
    PipelineConfig base_config{};                  ///< Shingle size and Hamming threshold.
    int thread_count{4};                           ///< Number of OpenMP threads to employ (must be > 0).
    SchedulePolicy schedule_policy{SchedulePolicy::Static}; ///< OpenMP loop schedule policy.
    int chunk_size{0};                             ///< Chunk size for schedule (0 = OpenMP default).

    /**
     * @brief Validates parallel configuration settings. Throws std::invalid_argument if invalid.
     */
    void validate() const;
};

/**
 * @brief Configuration parameters for workload-aware adaptive OpenMP pipeline execution (Phase 5).
 */
struct AdaptiveConfig {
    PipelineConfig base_config{};                                       ///< Shingle size and Hamming threshold.
    int thread_count{4};                                                ///< Requested OpenMP thread count (must be > 0).
    double cv_threshold{kDefaultCvThreshold};                           ///< CV threshold for Static (<=) vs Dynamic (>) (must be >= 0 and finite).
    double oversized_bucket_multiplier{kDefaultOversizedMultiplier};    ///< Multiplier over mean positive pair work (must be > 0 and finite).
    uint64_t max_pair_task_size{kDefaultMaxPairTaskSize};               ///< Max pair positions per heavy-bucket chunk (must be > 0).
    int ordinary_chunk_size{0};                                         ///< OpenMP chunk size for ordinary buckets (0 = OpenMP default, >= 0).

    /**
     * @brief Validates adaptive configuration settings. Throws std::invalid_argument if invalid.
     */
    void validate() const;
};

/**
 * @brief Compact descriptor representing a bounded range of unordered pair positions
 *        within an oversized SimHash bucket.
 *
 * Avoids materializing all k_i * (k_i - 1) / 2 pairs prior to parallel execution.
 * Each chunk covers the half-open linear pair-position interval [start_pair_pos, end_pair_pos)
 * within the bucket identified by (band_index, band_value).
 */
struct HeavyBucketChunk {
    size_t band_index{0};         ///< Band index in [0, kNumBands - 1].
    uint8_t band_value{0};        ///< 8-bit band value in [0, 255].
    uint64_t start_pair_pos{0};   ///< Inclusive starting linear pair position in [0, W_i).
    uint64_t end_pair_pos{0};     ///< Exclusive ending linear pair position in (start_pair_pos, W_i].

    uint64_t pair_count() const noexcept {
        return (end_pair_pos >= start_pair_pos) ? (end_pair_pos - start_pair_pos) : 0;
    }

    bool operator==(const HeavyBucketChunk& other) const noexcept {
        return band_index == other.band_index &&
               band_value == other.band_value &&
               start_pair_pos == other.start_pair_pos &&
               end_pair_pos == other.end_pair_pos;
    }
};

/**
 * @brief Diagnostic metrics recorded by the workload-aware adaptive scheduler.
 */
struct AdaptiveDiagnostics {
    double observed_cv{0.0};                                            ///< Measured CV of positive-work buckets.
    double cv_threshold{kDefaultCvThreshold};                           ///< Configured CV threshold.
    double oversized_bucket_multiplier{kDefaultOversizedMultiplier};    ///< Configured oversized-bucket multiplier.
    double mean_positive_pair_work{0.0};                                ///< Measured mean pair work across positive-work buckets.
    double oversized_pair_work_threshold{0.0};                          ///< Computed threshold (multiplier * mean_positive_pair_work).
    SchedulePolicy selected_ordinary_policy{SchedulePolicy::Static};    ///< Policy chosen for ordinary buckets (Static or Dynamic).
    std::string decision_reason{};                                      ///< Human-readable explanation of the scheduling decision.
    size_t occupied_buckets{0};                                         ///< Buckets with k_i >= 1.
    size_t positive_work_buckets{0};                                    ///< Buckets with k_i >= 2 (W_i > 0).
    size_t ordinary_buckets{0};                                         ///< Positive-work buckets with W_i <= oversized threshold.
    size_t oversized_buckets{0};                                        ///< Positive-work buckets with W_i > oversized threshold.
    uint64_t ordinary_pair_work{0};                                     ///< Total estimated pair work in ordinary buckets.
    uint64_t oversized_pair_work{0};                                    ///< Total estimated pair work in oversized buckets.
    size_t heavy_bucket_chunks{0};                                      ///< Total bounded chunks generated from oversized buckets.
    uint64_t max_generated_chunk_size{0};                               ///< Largest pair count among generated heavy-bucket chunks.
    int configured_thread_count{0};                                     ///< Thread count requested in AdaptiveConfig.
    int actual_thread_count{0};                                         ///< Actual OpenMP team size formed at runtime.
    uint64_t max_pair_task_size{kDefaultMaxPairTaskSize};               ///< Configured maximum pair positions per chunk.
    std::vector<uint64_t> per_thread_pair_work;                         ///< Estimated pair positions (W_t) assigned per worker thread.
    std::vector<size_t> per_thread_units_processed;                     ///< Total ordinary buckets + heavy chunks processed per worker thread.

    void print(std::ostream& os) const;
};

/**
 * @brief Non-overlapping wall-clock stage timings (in milliseconds) for the adaptive pipeline.
 *
 * Timing Definitions:
 * - fingerprinting_ms: Parallel SimHash computation across all input documents.
 * - bucket_construction_stats_ms: Sequential band table insertion and WorkloadStatistics computation.
 * - adaptive_decision_ms: Evaluating CV against cv_threshold and classifying positive-work buckets
 *   into ordinary vs. oversized buckets.
 * - ordinary_candidate_gen_ms: Parallel pair generation across ordinary buckets under the selected
 *   baseline policy (Static or Dynamic).
 * - heavy_chunk_generation_ms: Partitioning oversized buckets into compact HeavyBucketChunk descriptors.
 * - heavy_pair_processing_ms: Parallel pair generation across heavy-bucket chunks under OpenMP Dynamic scheduling.
 * - merge_sort_dedup_ms: Sequential concatenation of per-thread candidate buffers, lexicographical
 *   sorting, and std::unique deduplication.
 * - verification_ms: Parallel Hamming-distance verification, sequential merge, and deterministic sort.
 * - total_pipeline_ms: Complete end-to-end wall-clock runtime of run_adaptive_pipeline().
 */
struct AdaptiveStageTimings {
    double fingerprinting_ms{0.0};
    double bucket_construction_stats_ms{0.0};
    double adaptive_decision_ms{0.0};
    double ordinary_candidate_gen_ms{0.0};
    double heavy_chunk_generation_ms{0.0};
    double heavy_pair_processing_ms{0.0};
    double merge_sort_dedup_ms{0.0};
    double verification_ms{0.0};
    double total_pipeline_ms{0.0};

    void print(std::ostream& os) const;
};

/**
 * @brief Complete output of the workload-aware adaptive detection pipeline.
 */
struct AdaptivePipelineResult {
    PipelineResult pipeline_result{};               ///< Standard detection outputs (IDs, fingerprints, pairs, stats, timings).
    AdaptiveDiagnostics diagnostics{};              ///< Adaptive scheduling decisions and heavy-bucket metrics.
    AdaptiveStageTimings adaptive_timings{};        ///< Fine-grained non-overlapping stage timings.
    std::vector<HeavyBucketChunk> heavy_chunks{};   ///< Compact descriptors of generated heavy-bucket chunks.
};

/**
 * @brief Selects the baseline OpenMP schedule policy for ordinary buckets from observed CV.
 *
 * Decision Rule:
 * - If observed_cv <= cv_threshold: returns SchedulePolicy::Static.
 * - If observed_cv >  cv_threshold: returns SchedulePolicy::Dynamic.
 *
 * Guided scheduling is never silently substituted by the adaptive rule.
 */
SchedulePolicy select_ordinary_schedule_policy(double observed_cv,
                                               double cv_threshold = kDefaultCvThreshold,
                                               std::string* reason_out = nullptr);

/**
 * @brief Maps a 0-based linear pair position in [0, k*(k-1)/2) to its unique
 *        lexicographical document index pair (i, j) with 0 <= i < j < bucket_size.
 *
 * Throws std::invalid_argument if bucket_size < 2, or std::out_of_range if pair_pos >= k*(k-1)/2.
 */
std::pair<size_t, size_t> unrank_pair_position(size_t bucket_size, uint64_t pair_pos);

/**
 * @brief Splits an oversized bucket's unordered pair-generation work into compact chunks
 *        containing at most max_pair_task_size pair positions each.
 */
std::vector<HeavyBucketChunk> split_bucket_into_chunks(size_t band_index,
                                                       uint8_t band_value,
                                                       uint64_t estimated_pair_work,
                                                       uint64_t max_pair_task_size = kDefaultMaxPairTaskSize);

/**
 * @brief Executes the end-to-end fixed-schedule OpenMP-parallel near-duplicate detection pipeline.
 *
 * Parallelized Stages:
 * 1. Fingerprinting: Parallel loop over documents writing to disjoint preallocated indices.
 * 2. Candidate Generation: Parallel loop over candidate buckets using per-thread buffer slots
 *    in a std::vector (indexed by thread ID, avoiding TLS overhead), merged and lexicographically
 *    sorted/deduplicated after parallel region.
 * 3. Verification: Parallel loop over deduplicated candidate pairs evaluating bitwise Hamming
 *    distance using per-thread buffer slots, merged and sorted after parallel region.
 *
 * Sequential Stages:
 * - Bucket construction (cheap and race-free for 8 bands x 256 buckets).
 * - Workload statistics calculation (cheap statistical summary over 2048 buckets).
 *
 * Guaranteed Invariants:
 * - Deterministic, bitwise-identical output to run_sequential_pipeline() for identical inputs.
 * - Race-free execution across all thread counts.
 *
 * @param documents Input document collection.
 * @param config Parallel configuration (thread count, schedule policy, chunk size).
 * @return PipelineResult Complete pipeline results including stage timings and workload statistics.
 */
PipelineResult run_parallel_pipeline(const std::vector<Document>& documents,
                                     const ParallelConfig& config = ParallelConfig{});

/**
 * @brief Executes the workload-aware adaptive OpenMP near-duplicate detection pipeline (Phase 5).
 *
 * Adaptive Strategy:
 * 1. Computes SimHash fingerprints in parallel and constructs the 8x256 band buckets sequentially.
 * 2. Evaluates actual bucket workload statistics (mean positive pair work and CV).
 * 3. Selects Static (CV <= cv_threshold) or Dynamic (CV > cv_threshold) scheduling for ordinary buckets.
 * 4. Identifies oversized buckets where W_i > oversized_bucket_multiplier * mean_positive_pair_work.
 * 5. Processes ordinary buckets in parallel using the selected baseline schedule.
 * 6. Splits oversized buckets into bounded HeavyBucketChunk ranges (at most max_pair_task_size pairs)
 *    and processes them in parallel using OpenMP Dynamic scheduling.
 * 7. Merges, lexicographically sorts, and deduplicates candidate pairs after all parallel work finishes,
 *    then verifies pairs in parallel using bitwise Hamming distance.
 *
 * @param documents Input document collection.
 * @param config Adaptive configuration (thread count, CV threshold, oversized multiplier, max task size).
 * @return AdaptivePipelineResult Complete pipeline outputs, diagnostics, stage timings, and chunk descriptors.
 */
AdaptivePipelineResult run_adaptive_pipeline(const std::vector<Document>& documents,
                                             const AdaptiveConfig& config = AdaptiveConfig{});

#endif // PARALLEL_PIPELINE_HPP
