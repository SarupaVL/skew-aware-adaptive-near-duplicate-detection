#ifndef PARALLEL_PIPELINE_HPP
#define PARALLEL_PIPELINE_HPP

#include "candidate_gen.hpp"
#include <omp.h>
#include <cstddef>
#include <string>
#include <vector>

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
 * @brief Executes the end-to-end OpenMP-parallel near-duplicate detection pipeline.
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

#endif // PARALLEL_PIPELINE_HPP
