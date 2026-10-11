#include "parallel_pipeline.hpp"
#include "simhash.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

void ParallelConfig::validate() const {
    if (thread_count <= 0) {
        throw std::invalid_argument(
            "ParallelConfig::validate: thread_count must be positive (got " +
            std::to_string(thread_count) + ")."
        );
    }
    if (chunk_size < 0) {
        throw std::invalid_argument(
            "ParallelConfig::validate: chunk_size cannot be negative (got " +
            std::to_string(chunk_size) + ")."
        );
    }
    if (schedule_policy != SchedulePolicy::Static &&
        schedule_policy != SchedulePolicy::Dynamic &&
        schedule_policy != SchedulePolicy::Guided) {
        throw std::invalid_argument(
            "ParallelConfig::validate: unknown schedule_policy enum value (" +
            std::to_string(static_cast<int>(schedule_policy)) + ")."
        );
    }
}

void AdaptiveConfig::validate() const {
    if (thread_count <= 0) {
        throw std::invalid_argument(
            "AdaptiveConfig::validate: thread_count must be positive (got " +
            std::to_string(thread_count) + ")."
        );
    }
    if (!std::isfinite(cv_threshold) || cv_threshold < 0.0) {
        throw std::invalid_argument(
            "AdaptiveConfig::validate: cv_threshold must be finite and non-negative."
        );
    }
    if (!std::isfinite(oversized_bucket_multiplier) || oversized_bucket_multiplier <= 0.0) {
        throw std::invalid_argument(
            "AdaptiveConfig::validate: oversized_bucket_multiplier must be finite and strictly positive."
        );
    }
    if (max_pair_task_size == 0) {
        throw std::invalid_argument(
            "AdaptiveConfig::validate: max_pair_task_size must be positive (> 0)."
        );
    }
    if (ordinary_chunk_size < 0) {
        throw std::invalid_argument(
            "AdaptiveConfig::validate: ordinary_chunk_size cannot be negative (got " +
            std::to_string(ordinary_chunk_size) + ")."
        );
    }
    if (base_config.shingle_size == 0) {
        throw std::invalid_argument(
            "AdaptiveConfig::validate: base_config.shingle_size must be positive (> 0)."
        );
    }
    if (base_config.max_hamming_threshold < 0 || base_config.max_hamming_threshold > 64) {
        throw std::invalid_argument(
            "AdaptiveConfig::validate: base_config.max_hamming_threshold must be in [0, 64] (got " +
            std::to_string(base_config.max_hamming_threshold) + ")."
        );
    }
}

namespace detail {

OmpScheduleGuard::OmpScheduleGuard(SchedulePolicy policy, int chunk_size) {
    omp_get_schedule(&orig_sched, &orig_modifier);

    omp_sched_t target_sched = omp_sched_static;
    switch (policy) {
        case SchedulePolicy::Static:  target_sched = omp_sched_static;  break;
        case SchedulePolicy::Dynamic: target_sched = omp_sched_dynamic; break;
        case SchedulePolicy::Guided:  target_sched = omp_sched_guided;  break;
        default:
            throw std::invalid_argument("OmpScheduleGuard: unknown schedule policy enum value.");
    }

    omp_set_schedule(target_sched, chunk_size);
}

OmpScheduleGuard::~OmpScheduleGuard() {
    omp_set_schedule(orig_sched, orig_modifier);
}

} // namespace detail

PipelineResult run_parallel_pipeline(const std::vector<Document>& documents,
                                     const ParallelConfig& config) {
    config.validate();

    // RAII guard safely sets requested OpenMP runtime schedule and restores prior state on return/exception
    detail::OmpScheduleGuard schedule_guard(config.schedule_policy, config.chunk_size);

    auto pipeline_start = SteadyClock::now();

    PipelineResult result;
    size_t num_docs = documents.size();
    result.document_ids.resize(num_docs);
    result.fingerprints.resize(num_docs);

    if (documents.empty()) {
        result.actual_thread_count = 0;
        result.per_thread_pair_work.assign(static_cast<size_t>(config.thread_count), 0ULL);
        result.per_thread_units_processed.assign(static_cast<size_t>(config.thread_count), 0);
        auto pipeline_end = SteadyClock::now();
        result.timings.total_pipeline_ms = elapsed_ms(pipeline_start, pipeline_end);
        return result;
    }

    // -------------------------------------------------------------
    // Pre-Stage Validation & ID Mapping
    // -------------------------------------------------------------
    // Map each document ID to its position in the arrays to guarantee unique IDs
    // and enable fast O(1) fingerprint lookup during candidate verification.
    // Note: This sequential validation step runs before t1_start so that
    // fingerprinting_ms measures strictly the OpenMP parallel fingerprinting loop.
    std::unordered_map<uint64_t, size_t> id_to_index;
    id_to_index.reserve(num_docs);

    for (size_t i = 0; i < num_docs; ++i) {
        const auto& doc = documents[i];
        auto insert_res = id_to_index.emplace(doc.id, i);
        if (!insert_res.second) {
            throw std::invalid_argument(
                "run_parallel_pipeline: Duplicate document ID detected (" +
                std::to_string(doc.id) + "). All document IDs must be unique."
            );
        }
        result.document_ids[i] = doc.id;
    }

    // -------------------------------------------------------------
    // Stage 1: Parallel Document Fingerprinting
    // -------------------------------------------------------------
    // Strategy: Loop over all documents in parallel.
    // Concurrency Safety:
    // - Preallocated vector `result.fingerprints` of exact size `num_docs`.
    // - Thread i writes strictly to its disjoint memory cell `result.fingerprints[i]`.
    // - No shared mutable state or critical sections required.
    auto t1_start = SteadyClock::now();

    #pragma omp parallel for num_threads(config.thread_count) schedule(runtime) default(none) \
        shared(documents, result, config, num_docs)
    for (size_t i = 0; i < num_docs; ++i) {
        result.fingerprints[i] = compute_simhash(documents[i].text, config.base_config.shingle_size);
    }

    auto t1_end = SteadyClock::now();
    result.timings.fingerprinting_ms = elapsed_ms(t1_start, t1_end);

    // -------------------------------------------------------------
    // Stage 2: Band Bucket Construction & Workload Statistics
    // -------------------------------------------------------------
    // Design Decision: Bucket construction is performed sequentially.
    // Rationale:
    // With 8 bands and 256 bucket slots per band, total bucket count is small (2048).
    // Sequential insertion is extremely fast (< 0.2 ms for 100K docs) and completely
    // eliminates concurrent write locks or multi-hash contention across threads.
    auto t2_start = SteadyClock::now();

    std::array<std::array<std::vector<uint64_t>, 256>, kNumBands> buckets_table;

    for (size_t i = 0; i < num_docs; ++i) {
        uint64_t doc_id = result.document_ids[i];
        uint64_t fp = result.fingerprints[i];

        for (size_t band = 0; band < kNumBands; ++band) {
            uint8_t val = extract_band(fp, band);
            buckets_table[band][val].push_back(doc_id);
        }
    }

    // Collect non-empty BucketInfo and calculate workload statistics
    size_t total_occupied_docs = 0;
    std::vector<uint64_t> positive_works;

    // Also collect pointers to buckets having >= 2 documents for parallel candidate generation
    std::vector<const std::vector<uint64_t>*> active_buckets;
    active_buckets.reserve(kNumBands * 256);

    for (size_t band = 0; band < kNumBands; ++band) {
        for (size_t val = 0; val < 256; ++val) {
            const auto& bucket_docs = buckets_table[band][val];
            if (!bucket_docs.empty()) {
                BucketInfo binfo;
                binfo.band_index = band;
                binfo.band_value = static_cast<uint8_t>(val);
                binfo.document_count = bucket_docs.size();
                binfo.estimated_pair_work = (binfo.document_count >= 2)
                    ? (static_cast<uint64_t>(binfo.document_count) * (binfo.document_count - 1)) / 2
                    : 0;

                result.workload_stats.occupied_buckets++;
                total_occupied_docs += binfo.document_count;

                if (binfo.document_count > result.workload_stats.max_bucket_size) {
                    result.workload_stats.max_bucket_size = binfo.document_count;
                }
                if (binfo.estimated_pair_work > result.workload_stats.max_estimated_pair_work) {
                    result.workload_stats.max_estimated_pair_work = binfo.estimated_pair_work;
                }

                if (binfo.document_count >= 2) {
                    result.workload_stats.positive_work_buckets++;
                    positive_works.push_back(binfo.estimated_pair_work);
                    active_buckets.push_back(&bucket_docs);
                }

                result.buckets.push_back(binfo);
            }
        }
    }

    if (result.workload_stats.occupied_buckets > 0) {
        result.workload_stats.mean_bucket_size = static_cast<double>(total_occupied_docs) /
            static_cast<double>(result.workload_stats.occupied_buckets);
    }

    if (result.workload_stats.positive_work_buckets > 0) {
        uint64_t sum_work = 0;
        for (uint64_t w : positive_works) {
            sum_work += w;
        }
        double mean_work = static_cast<double>(sum_work) /
            static_cast<double>(result.workload_stats.positive_work_buckets);
        result.workload_stats.mean_estimated_pair_work = mean_work;

        // Population variance and standard deviation
        double variance_sum = 0.0;
        for (uint64_t w : positive_works) {
            double diff = static_cast<double>(w) - mean_work;
            variance_sum += (diff * diff);
        }
        double pop_sd = std::sqrt(variance_sum / static_cast<double>(result.workload_stats.positive_work_buckets));

        if (mean_work > 0.0) {
            result.workload_stats.cv_estimated_pair_work = pop_sd / mean_work;
        } else {
            result.workload_stats.cv_estimated_pair_work = 0.0;
        }
    } else {
        result.workload_stats.mean_estimated_pair_work = 0.0;
        result.workload_stats.cv_estimated_pair_work = 0.0;
    }

    auto t2_end = SteadyClock::now();
    result.timings.bucket_construction_ms = elapsed_ms(t2_start, t2_end);

    // -------------------------------------------------------------
    // Stage 3: Parallel Candidate Generation & Deduplication
    // -------------------------------------------------------------
    // Strategy: Loop over all active buckets in parallel using OpenMP.
    // Concurrency Safety:
    // - Preallocated vector of per-thread buffer slots (per_thread_candidates[tid]).
    // - Note: We use heap-allocated per-thread vector slots indexed by thread ID (0 <= tid < thread_count),
    //   NOT C++ `thread_local` storage duration. This guarantees deterministic lifetimes and
    //   avoids TLS overhead, limits, or barrier synchronization.
    // - Thread `tid` appends pairs exclusively to its dedicated slot `per_thread_candidates[tid]`.
    // - Completely avoids mutexes, locks, or critical sections.
    // - After parallel loop, per-thread slots are sequentially concatenated.
    // - Final candidates are sorted and deduplicated via std::unique to guarantee
    //   bitwise-deterministic candidate ordering identical to sequential execution.
    auto t3_start = SteadyClock::now();

    size_t num_active_buckets = active_buckets.size();
    std::vector<std::vector<CandidatePair>> per_thread_candidates(config.thread_count);
    std::vector<size_t> per_thread_units(config.thread_count, 0);
    int observed_team_size = config.thread_count;

    #pragma omp parallel num_threads(config.thread_count) default(none) \
        shared(active_buckets, num_active_buckets, per_thread_candidates, per_thread_units, observed_team_size)
    {
        int tid = omp_get_thread_num();
        size_t local_units = 0;

        #pragma omp single
        {
            int actual_threads = omp_get_num_threads();
            observed_team_size = actual_threads;
            if (actual_threads > static_cast<int>(per_thread_candidates.size())) {
                per_thread_candidates.resize(actual_threads);
                per_thread_units.resize(actual_threads, 0);
            }
        }

        #pragma omp for schedule(runtime)
        for (size_t b = 0; b < num_active_buckets; ++b) {
            ++local_units;
            const auto& bucket_docs = *active_buckets[b];
            size_t bsize = bucket_docs.size();

            for (size_t i = 0; i < bsize; ++i) {
                for (size_t j = i + 1; j < bsize; ++j) {
                    uint64_t u = bucket_docs[i];
                    uint64_t v = bucket_docs[j];
                    if (u != v) {
                        CandidatePair pair = (u < v) ? CandidatePair{u, v} : CandidatePair{v, u};
                        per_thread_candidates[tid].push_back(pair);
                    }
                }
            }
        }

        per_thread_units[tid] = local_units;
    }

    result.actual_thread_count = observed_team_size;
    result.per_thread_pair_work.resize(per_thread_candidates.size(), 0ULL);
    for (size_t t = 0; t < per_thread_candidates.size(); ++t) {
        result.per_thread_pair_work[t] = static_cast<uint64_t>(per_thread_candidates[t].size());
    }
    result.per_thread_units_processed = std::move(per_thread_units);

    // Merge per-thread candidate lists sequentially
    size_t total_raw_candidates = 0;
    for (const auto& pt_vec : per_thread_candidates) {
        total_raw_candidates += pt_vec.size();
    }

    std::vector<CandidatePair> raw_candidates;
    raw_candidates.reserve(total_raw_candidates);
    for (auto& pt_vec : per_thread_candidates) {
        raw_candidates.insert(
            raw_candidates.end(),
            std::make_move_iterator(pt_vec.begin()),
            std::make_move_iterator(pt_vec.end())
        );
    }

    // Deterministic sort and deduplication
    std::sort(raw_candidates.begin(), raw_candidates.end());
    raw_candidates.erase(
        std::unique(raw_candidates.begin(), raw_candidates.end()),
        raw_candidates.end()
    );
    result.candidate_pairs = std::move(raw_candidates);

    auto t3_end = SteadyClock::now();
    result.timings.candidate_generation_dedup_ms = elapsed_ms(t3_start, t3_end);

    // -------------------------------------------------------------
    // Stage 4: Parallel Hamming Distance Verification
    // -------------------------------------------------------------
    // Strategy: Loop over all candidate pairs in parallel using OpenMP.
    // Concurrency Safety:
    // - Preallocated vector of per-thread buffer slots (per_thread_verified[tid]).
    // - Disjoint per-thread accumulation (vector of vectors, not C++ `thread_local`).
    // - Read-only lookup into `id_to_index` and `result.fingerprints`.
    // - After parallel loop, per-thread slots are sequentially concatenated and sorted
    //   lexicographically to preserve exact, bitwise-deterministic ordering matching
    //   the sequential reference pipeline.
    auto t4_start = SteadyClock::now();

    size_t num_candidates = result.candidate_pairs.size();
    std::vector<std::vector<VerifiedPair>> per_thread_verified(config.thread_count);

    #pragma omp parallel num_threads(config.thread_count) default(none) \
        shared(result, num_candidates, id_to_index, config, per_thread_verified)
    {
        int tid = omp_get_thread_num();

        #pragma omp single
        {
            int actual_threads = omp_get_num_threads();
            if (actual_threads > static_cast<int>(per_thread_verified.size())) {
                per_thread_verified.resize(actual_threads);
            }
        }

        #pragma omp for schedule(runtime)
        for (size_t c = 0; c < num_candidates; ++c) {
            const auto& candidate = result.candidate_pairs[c];
            size_t idx1 = id_to_index.at(candidate.first);
            size_t idx2 = id_to_index.at(candidate.second);

            uint64_t fp1 = result.fingerprints[idx1];
            uint64_t fp2 = result.fingerprints[idx2];

            int dist = hamming_distance(fp1, fp2);
            if (dist <= config.base_config.max_hamming_threshold) {
                per_thread_verified[tid].push_back(VerifiedPair{candidate.first, candidate.second, dist});
            }
        }
    }

    // Merge per-thread verified pairs sequentially
    size_t total_verified = 0;
    for (const auto& pt_vec : per_thread_verified) {
        total_verified += pt_vec.size();
    }

    result.verified_pairs.reserve(total_verified);
    for (auto& pt_vec : per_thread_verified) {
        result.verified_pairs.insert(
            result.verified_pairs.end(),
            std::make_move_iterator(pt_vec.begin()),
            std::make_move_iterator(pt_vec.end())
        );
    }

    // Sort to guarantee deterministic ordering matching sequential pipeline
    std::sort(result.verified_pairs.begin(), result.verified_pairs.end());

    auto t4_end = SteadyClock::now();
    result.timings.verification_ms = elapsed_ms(t4_start, t4_end);

    auto pipeline_end = SteadyClock::now();
    result.timings.total_pipeline_ms = elapsed_ms(pipeline_start, pipeline_end);

    return result;
}

// =============================================================
// Phase 5: Workload-Aware Adaptive Scheduling & Heavy-Bucket Splitting
// =============================================================

void AdaptiveDiagnostics::print(std::ostream& os) const {
    os << "---------------------------------------------------------\n";
    os << " Adaptive Scheduling Diagnostics\n";
    os << "---------------------------------------------------------\n";
    os << "  Configured Thread Count           : " << configured_thread_count << "\n";
    os << "  Actual OpenMP Team Size           : " << actual_thread_count << "\n";
    os << "  Observed Bucket Pair-Work CV      : " << std::fixed << std::setprecision(4)
       << observed_cv << " (Threshold: " << cv_threshold << ")\n";
    os << "  Selected Ordinary-Bucket Policy   : " << to_string(selected_ordinary_policy) << "\n";
    os << "  Decision Reason                   : " << decision_reason << "\n";
    os << "  Occupied Buckets (k >= 1)         : " << occupied_buckets << "\n";
    os << "  Positive-Work Buckets (k >= 2)    : " << positive_work_buckets << "\n";
    os << "  Mean Positive Pair Work (W_bar)   : " << std::setprecision(2)
       << mean_positive_pair_work << "\n";
    os << "  Oversized Threshold (" << oversized_bucket_multiplier << " x W_bar) : "
       << oversized_pair_work_threshold << "\n";
    os << "  Ordinary Positive-Work Buckets    : " << ordinary_buckets
       << " (Pair Work: " << ordinary_pair_work << ")\n";
    os << "  Oversized Buckets                 : " << oversized_buckets
       << " (Pair Work: " << oversized_pair_work << ")\n";
    os << "  Task-Size Limit (Pairs / Chunk)   : " << max_pair_task_size << "\n";
    os << "  Generated Heavy-Bucket Chunks     : " << heavy_bucket_chunks << "\n";
    os << "  Max Generated Chunk Size          : " << max_generated_chunk_size << "\n";
    os << "---------------------------------------------------------\n";
}

void AdaptiveStageTimings::print(std::ostream& os) const {
    os << "---------------------------------------------------------\n";
    os << " Adaptive Pipeline Stage Timings (Non-Overlapping)\n";
    os << "---------------------------------------------------------\n";
    os << "  1. Fingerprinting                 : " << std::fixed << std::setprecision(3)
       << fingerprinting_ms << " ms\n";
    os << "  2. Bucket Construction & Stats    : " << bucket_construction_stats_ms << " ms\n";
    os << "  3a. Adaptive Decision & Classify  : " << adaptive_decision_ms << " ms\n";
    os << "  3b. Ordinary-Bucket Candidate Gen : " << ordinary_candidate_gen_ms << " ms\n";
    os << "  3c. Heavy-Bucket Chunk Generation : " << heavy_chunk_generation_ms << " ms\n";
    os << "  3d. Heavy-Bucket Pair Processing  : " << heavy_pair_processing_ms << " ms\n";
    os << "  3e. Merge, Sort & Deduplication   : " << merge_sort_dedup_ms << " ms\n";
    os << "  4. Hamming Distance Verification  : " << verification_ms << " ms\n";
    os << "  -------------------------------------------------------\n";
    os << "  Total Adaptive Pipeline Runtime   : " << total_pipeline_ms << " ms\n";
    os << "---------------------------------------------------------\n";
}

SchedulePolicy select_ordinary_schedule_policy(double observed_cv,
                                               double cv_threshold,
                                               std::string* reason_out) {
    if (!std::isfinite(observed_cv) || observed_cv < 0.0) {
        throw std::invalid_argument(
            "select_ordinary_schedule_policy: observed_cv must be finite and non-negative."
        );
    }
    if (!std::isfinite(cv_threshold) || cv_threshold < 0.0) {
        throw std::invalid_argument(
            "select_ordinary_schedule_policy: cv_threshold must be finite and non-negative."
        );
    }

    if (observed_cv <= cv_threshold) {
        if (reason_out != nullptr) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4)
                << "CV (" << observed_cv << ") <= threshold (" << cv_threshold
                << ") -> Static schedule for ordinary buckets";
            *reason_out = oss.str();
        }
        return SchedulePolicy::Static;
    } else {
        if (reason_out != nullptr) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4)
                << "CV (" << observed_cv << ") > threshold (" << cv_threshold
                << ") -> Dynamic schedule for ordinary buckets";
            *reason_out = oss.str();
        }
        return SchedulePolicy::Dynamic;
    }
}

namespace {

/**
 * @brief Computes the starting 0-based linear pair position for row `i` in a bucket of size `k`.
 *
 * How Linear Pair Positions Map to Unordered Document Pairs:
 * Given a bucket with `k` documents indexed 0 .. k-1, there are W = k*(k-1)/2 unordered
 * index pairs (i, j) with 0 <= i < j < k.
 * Ordering pairs lexicographically by (i, j):
 *   Row i = 0 has (k - 1) pairs: (0, 1), (0, 2), ..., (0, k - 1) -> positions [0, k - 2]
 *   Row i = 1 has (k - 2) pairs: (1, 2), (1, 3), ..., (1, k - 1) -> positions [k - 1, 2k - 4]
 *   ...
 *   Row i has (k - 1 - i) pairs, starting at linear offset:
 *     row_start(i, k) = sum_{r=0}^{i-1} (k - 1 - r) = i * (2k - 1 - i) / 2
 *
 * Within row i, column j in [i + 1, k - 1] is at offset (j - i - 1), so:
 *   pair_pos(i, j) = row_start(i, k) + (j - i - 1).
 *
 * To avoid 64-bit integer overflow prior to division by 2, the even factor of
 * i and (2k - 1 - i) is divided by 2 first.
 */
inline uint64_t row_start_pair_pos(size_t i, size_t k) noexcept {
    uint64_t ui = static_cast<uint64_t>(i);
    uint64_t factor = 2ULL * static_cast<uint64_t>(k) - 1ULL - ui;
    if ((ui & 1ULL) == 0ULL) {
        return (ui / 2ULL) * factor;
    } else {
        return ui * (factor / 2ULL);
    }
}

} // namespace

std::pair<size_t, size_t> unrank_pair_position(size_t bucket_size, uint64_t pair_pos) {
    if (bucket_size < 2) {
        throw std::invalid_argument(
            "unrank_pair_position: bucket_size must be at least 2 (got " +
            std::to_string(bucket_size) + ")."
        );
    }

    uint64_t total_pairs = row_start_pair_pos(bucket_size - 1, bucket_size);
    if (pair_pos >= total_pairs) {
        throw std::out_of_range(
            "unrank_pair_position: pair_pos (" + std::to_string(pair_pos) +
            ") out of range for total_pairs (" + std::to_string(total_pairs) + ")."
        );
    }

    // Binary search for the largest row index i in [0, bucket_size - 2]
    // satisfying row_start_pair_pos(i, bucket_size) <= pair_pos.
    size_t low = 0;
    size_t high = bucket_size - 2;
    while (low < high) {
        size_t mid = low + (high - low + 1) / 2;
        if (row_start_pair_pos(mid, bucket_size) <= pair_pos) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    size_t i = low;
    uint64_t offset_in_row = pair_pos - row_start_pair_pos(i, bucket_size);
    size_t j = i + 1 + static_cast<size_t>(offset_in_row);
    return {i, j};
}

std::vector<HeavyBucketChunk> split_bucket_into_chunks(size_t band_index,
                                                       uint8_t band_value,
                                                       uint64_t estimated_pair_work,
                                                       uint64_t max_pair_task_size) {
    if (band_index >= kNumBands) {
        throw std::invalid_argument(
            "split_bucket_into_chunks: band_index must be in [0, " +
            std::to_string(kNumBands - 1) + "] (got " + std::to_string(band_index) + ")."
        );
    }
    if (max_pair_task_size == 0) {
        throw std::invalid_argument(
            "split_bucket_into_chunks: max_pair_task_size must be positive (> 0)."
        );
    }

    std::vector<HeavyBucketChunk> chunks;
    if (estimated_pair_work == 0) {
        return chunks;
    }

    uint64_t num_chunks = (estimated_pair_work - 1ULL) / max_pair_task_size + 1ULL;
    chunks.reserve(static_cast<size_t>(num_chunks));

    uint64_t start = 0;
    while (start < estimated_pair_work) {
        uint64_t remaining = estimated_pair_work - start;
        uint64_t chunk_len = std::min(remaining, max_pair_task_size);
        uint64_t end = start + chunk_len;

        chunks.push_back(HeavyBucketChunk{band_index, band_value, start, end});
        start = end;
    }

    return chunks;
}

AdaptivePipelineResult run_adaptive_pipeline(const std::vector<Document>& documents,
                                             const AdaptiveConfig& config) {
    config.validate();

    auto pipeline_start = SteadyClock::now();

    AdaptivePipelineResult adaptive_res;
    PipelineResult& result = adaptive_res.pipeline_result;
    AdaptiveDiagnostics& diag = adaptive_res.diagnostics;
    AdaptiveStageTimings& timings = adaptive_res.adaptive_timings;

    diag.cv_threshold = config.cv_threshold;
    diag.oversized_bucket_multiplier = config.oversized_bucket_multiplier;
    diag.configured_thread_count = config.thread_count;
    diag.actual_thread_count = 0;
    diag.max_pair_task_size = config.max_pair_task_size;

    size_t num_docs = documents.size();
    result.document_ids.resize(num_docs);
    result.fingerprints.resize(num_docs);

    if (documents.empty()) {
        auto decision_start = SteadyClock::now();
        diag.observed_cv = 0.0;
        diag.selected_ordinary_policy = select_ordinary_schedule_policy(
            0.0, config.cv_threshold, &diag.decision_reason
        );
        diag.decision_reason += " (empty dataset; 0 positive-work buckets)";
        result.actual_thread_count = 0;
        result.per_thread_pair_work.assign(static_cast<size_t>(config.thread_count), 0ULL);
        result.per_thread_units_processed.assign(static_cast<size_t>(config.thread_count), 0);
        diag.per_thread_pair_work = result.per_thread_pair_work;
        diag.per_thread_units_processed = result.per_thread_units_processed;
        auto decision_end = SteadyClock::now();

        timings.adaptive_decision_ms = elapsed_ms(decision_start, decision_end);
        timings.total_pipeline_ms = elapsed_ms(pipeline_start, decision_end);
        result.timings.candidate_generation_dedup_ms = timings.adaptive_decision_ms;
        result.timings.total_pipeline_ms = timings.total_pipeline_ms;
        return adaptive_res;
    }

    // -------------------------------------------------------------
    // Pre-Stage Validation & ID Mapping
    // -------------------------------------------------------------
    // Note: This sequential validation step runs before t1_start so that
    // fingerprinting_ms measures strictly the OpenMP parallel fingerprinting loop.
    std::unordered_map<uint64_t, size_t> id_to_index;
    id_to_index.reserve(num_docs);

    for (size_t i = 0; i < num_docs; ++i) {
        const auto& doc = documents[i];
        auto insert_res = id_to_index.emplace(doc.id, i);
        if (!insert_res.second) {
            throw std::invalid_argument(
                "run_adaptive_pipeline: Duplicate document ID detected (" +
                std::to_string(doc.id) + "). All document IDs must be unique."
            );
        }
        result.document_ids[i] = doc.id;
    }

    // -------------------------------------------------------------
    // Stage 1: Parallel Document Fingerprinting
    // -------------------------------------------------------------
    auto t1_start = SteadyClock::now();
    int observed_team_size = 0;
    {
        detail::OmpScheduleGuard fp_guard(SchedulePolicy::Static, 0);

        #pragma omp parallel num_threads(config.thread_count) default(none) \
            shared(documents, result, config, num_docs, observed_team_size)
        {
            #pragma omp single
            {
                observed_team_size = omp_get_num_threads();
            }

            #pragma omp for schedule(runtime)
            for (size_t i = 0; i < num_docs; ++i) {
                result.fingerprints[i] = compute_simhash(documents[i].text, config.base_config.shingle_size);
            }
        }
    }
    diag.actual_thread_count = observed_team_size;
    auto t1_end = SteadyClock::now();
    timings.fingerprinting_ms = elapsed_ms(t1_start, t1_end);

    // -------------------------------------------------------------
    // Stage 2: Band Bucket Construction & Workload Statistics
    // -------------------------------------------------------------
    auto t2_start = SteadyClock::now();

    std::array<std::array<std::vector<uint64_t>, 256>, kNumBands> buckets_table;

    for (size_t i = 0; i < num_docs; ++i) {
        uint64_t doc_id = result.document_ids[i];
        uint64_t fp = result.fingerprints[i];

        for (size_t band = 0; band < kNumBands; ++band) {
            uint8_t val = extract_band(fp, band);
            buckets_table[band][val].push_back(doc_id);
        }
    }

    size_t total_occupied_docs = 0;
    std::vector<uint64_t> positive_works;
    std::vector<BucketInfo> positive_bucket_infos;
    positive_bucket_infos.reserve(kNumBands * 256);

    for (size_t band = 0; band < kNumBands; ++band) {
        for (size_t val = 0; val < 256; ++val) {
            const auto& bucket_docs = buckets_table[band][val];
            if (!bucket_docs.empty()) {
                BucketInfo binfo;
                binfo.band_index = band;
                binfo.band_value = static_cast<uint8_t>(val);
                binfo.document_count = bucket_docs.size();
                binfo.estimated_pair_work = (binfo.document_count >= 2)
                    ? (static_cast<uint64_t>(binfo.document_count) * (binfo.document_count - 1)) / 2ULL
                    : 0ULL;

                result.workload_stats.occupied_buckets++;
                total_occupied_docs += binfo.document_count;

                if (binfo.document_count > result.workload_stats.max_bucket_size) {
                    result.workload_stats.max_bucket_size = binfo.document_count;
                }
                if (binfo.estimated_pair_work > result.workload_stats.max_estimated_pair_work) {
                    result.workload_stats.max_estimated_pair_work = binfo.estimated_pair_work;
                }

                if (binfo.document_count >= 2) {
                    result.workload_stats.positive_work_buckets++;
                    positive_works.push_back(binfo.estimated_pair_work);
                    positive_bucket_infos.push_back(binfo);
                }

                result.buckets.push_back(binfo);
            }
        }
    }

    if (result.workload_stats.occupied_buckets > 0) {
        result.workload_stats.mean_bucket_size = static_cast<double>(total_occupied_docs) /
            static_cast<double>(result.workload_stats.occupied_buckets);
    }

    if (result.workload_stats.positive_work_buckets > 0) {
        uint64_t sum_work = 0;
        for (uint64_t w : positive_works) {
            sum_work += w;
        }
        double mean_work = static_cast<double>(sum_work) /
            static_cast<double>(result.workload_stats.positive_work_buckets);
        result.workload_stats.mean_estimated_pair_work = mean_work;

        double variance_sum = 0.0;
        for (uint64_t w : positive_works) {
            double diff = static_cast<double>(w) - mean_work;
            variance_sum += (diff * diff);
        }
        double pop_sd = std::sqrt(variance_sum / static_cast<double>(result.workload_stats.positive_work_buckets));

        if (mean_work > 0.0) {
            result.workload_stats.cv_estimated_pair_work = pop_sd / mean_work;
        } else {
            result.workload_stats.cv_estimated_pair_work = 0.0;
        }
    } else {
        result.workload_stats.mean_estimated_pair_work = 0.0;
        result.workload_stats.cv_estimated_pair_work = 0.0;
    }

    auto t2_end = SteadyClock::now();
    timings.bucket_construction_stats_ms = elapsed_ms(t2_start, t2_end);

    // -------------------------------------------------------------
    // Stage 3a: Adaptive Scheduling Decision & Bucket Classification
    // -------------------------------------------------------------
    auto t3a_start = SteadyClock::now();

    diag.occupied_buckets = result.workload_stats.occupied_buckets;
    diag.positive_work_buckets = result.workload_stats.positive_work_buckets;
    diag.observed_cv = result.workload_stats.cv_estimated_pair_work;
    diag.mean_positive_pair_work = result.workload_stats.mean_estimated_pair_work;

    diag.selected_ordinary_policy = select_ordinary_schedule_policy(
        diag.observed_cv, config.cv_threshold, &diag.decision_reason
    );

    std::vector<const std::vector<uint64_t>*> ordinary_bucket_ptrs;
    std::vector<BucketInfo> oversized_bucket_infos;

    if (diag.positive_work_buckets > 0 && diag.mean_positive_pair_work > 0.0) {
        diag.oversized_pair_work_threshold =
            config.oversized_bucket_multiplier * diag.mean_positive_pair_work;

        ordinary_bucket_ptrs.reserve(positive_bucket_infos.size());
        for (const auto& binfo : positive_bucket_infos) {
            if (static_cast<double>(binfo.estimated_pair_work) > diag.oversized_pair_work_threshold) {
                oversized_bucket_infos.push_back(binfo);
                diag.oversized_buckets++;
                diag.oversized_pair_work += binfo.estimated_pair_work;
            } else {
                ordinary_bucket_ptrs.push_back(&buckets_table[binfo.band_index][binfo.band_value]);
                diag.ordinary_buckets++;
                diag.ordinary_pair_work += binfo.estimated_pair_work;
            }
        }
    } else {
        diag.oversized_pair_work_threshold = 0.0;
        diag.decision_reason += " (no positive-work buckets; no oversized buckets)";
    }

    auto t3a_end = SteadyClock::now();
    timings.adaptive_decision_ms = elapsed_ms(t3a_start, t3a_end);

    // Preallocate per-thread candidate buffers shared across Stage 3b and Stage 3d
    size_t team_slots = static_cast<size_t>(std::max(config.thread_count, diag.actual_thread_count));
    std::vector<std::vector<CandidatePair>> per_thread_candidates(team_slots);
    std::vector<size_t> per_thread_units(team_slots, 0);

    // -------------------------------------------------------------
    // Stage 3b: Ordinary-Bucket Candidate Generation
    // -------------------------------------------------------------
    auto t3b_start = SteadyClock::now();

    size_t num_ordinary = ordinary_bucket_ptrs.size();
    if (num_ordinary > 0) {
        detail::OmpScheduleGuard ordinary_guard(
            diag.selected_ordinary_policy, config.ordinary_chunk_size
        );

        #pragma omp parallel num_threads(config.thread_count) default(none) \
            shared(ordinary_bucket_ptrs, num_ordinary, per_thread_candidates, per_thread_units)
        {
            int tid = omp_get_thread_num();
            size_t local_units = 0;

            #pragma omp single
            {
                int actual_threads = omp_get_num_threads();
                if (actual_threads > static_cast<int>(per_thread_candidates.size())) {
                    per_thread_candidates.resize(actual_threads);
                    per_thread_units.resize(actual_threads, 0);
                }
            }

            #pragma omp for schedule(runtime)
            for (size_t b = 0; b < num_ordinary; ++b) {
                ++local_units;
                const auto& bucket_docs = *ordinary_bucket_ptrs[b];
                size_t bsize = bucket_docs.size();

                for (size_t i = 0; i < bsize; ++i) {
                    for (size_t j = i + 1; j < bsize; ++j) {
                        uint64_t u = bucket_docs[i];
                        uint64_t v = bucket_docs[j];
                        if (u != v) {
                            CandidatePair pair = (u < v) ? CandidatePair{u, v} : CandidatePair{v, u};
                            per_thread_candidates[tid].push_back(pair);
                        }
                    }
                }
            }

            per_thread_units[tid] += local_units;
        }
    }

    auto t3b_end = SteadyClock::now();
    timings.ordinary_candidate_gen_ms = elapsed_ms(t3b_start, t3b_end);

    // -------------------------------------------------------------
    // Stage 3c: Heavy-Bucket Chunk Generation
    // -------------------------------------------------------------
    auto t3c_start = SteadyClock::now();

    for (const auto& binfo : oversized_bucket_infos) {
        auto bucket_chunks = split_bucket_into_chunks(
            binfo.band_index,
            binfo.band_value,
            binfo.estimated_pair_work,
            config.max_pair_task_size
        );
        for (const auto& ch : bucket_chunks) {
            uint64_t csize = ch.pair_count();
            if (csize > diag.max_generated_chunk_size) {
                diag.max_generated_chunk_size = csize;
            }
            adaptive_res.heavy_chunks.push_back(ch);
        }
    }
    diag.heavy_bucket_chunks = adaptive_res.heavy_chunks.size();

    auto t3c_end = SteadyClock::now();
    timings.heavy_chunk_generation_ms = elapsed_ms(t3c_start, t3c_end);

    // -------------------------------------------------------------
    // Stage 3d: Heavy-Bucket Pair Processing (OpenMP Dynamic)
    // -------------------------------------------------------------
    auto t3d_start = SteadyClock::now();

    size_t num_heavy_chunks = adaptive_res.heavy_chunks.size();
    if (num_heavy_chunks > 0) {
        // Process bounded heavy-bucket chunks using OpenMP dynamic scheduling
        detail::OmpScheduleGuard heavy_guard(SchedulePolicy::Dynamic, 1);

        const auto& heavy_chunks_ref = adaptive_res.heavy_chunks;

        #pragma omp parallel num_threads(config.thread_count) default(none) \
            shared(heavy_chunks_ref, num_heavy_chunks, buckets_table, per_thread_candidates, per_thread_units)
        {
            int tid = omp_get_thread_num();
            size_t local_units = 0;

            #pragma omp single
            {
                int actual_threads = omp_get_num_threads();
                if (actual_threads > static_cast<int>(per_thread_candidates.size())) {
                    per_thread_candidates.resize(actual_threads);
                    per_thread_units.resize(actual_threads, 0);
                }
            }

            #pragma omp for schedule(runtime)
            for (size_t c = 0; c < num_heavy_chunks; ++c) {
                ++local_units;
                const HeavyBucketChunk& chunk = heavy_chunks_ref[c];
                const auto& bucket_docs = buckets_table[chunk.band_index][chunk.band_value];
                size_t bsize = bucket_docs.size();

                if (chunk.start_pair_pos < chunk.end_pair_pos) {
                    // Unrank the starting linear pair position into its initial (i, j) indices once,
                    // then advance (i, j) lexicographically in O(1) across the chunk's range.
                    auto [i, j] = unrank_pair_position(bsize, chunk.start_pair_pos);

                    for (uint64_t pos = chunk.start_pair_pos; pos < chunk.end_pair_pos; ++pos) {
                        uint64_t u = bucket_docs[i];
                        uint64_t v = bucket_docs[j];
                        if (u != v) {
                            CandidatePair pair = (u < v) ? CandidatePair{u, v} : CandidatePair{v, u};
                            per_thread_candidates[tid].push_back(pair);
                        }

                        ++j;
                        if (j >= bsize) {
                            ++i;
                            j = i + 1;
                        }
                    }
                }
            }

            per_thread_units[tid] += local_units;
        }
    }

    auto t3d_end = SteadyClock::now();
    timings.heavy_pair_processing_ms = elapsed_ms(t3d_start, t3d_end);

    // -------------------------------------------------------------
    // Stage 3e: Sequential Merge, Sort & Deduplication
    // -------------------------------------------------------------
    auto t3e_start = SteadyClock::now();

    result.actual_thread_count = diag.actual_thread_count;
    result.per_thread_pair_work.resize(per_thread_candidates.size(), 0ULL);
    for (size_t t = 0; t < per_thread_candidates.size(); ++t) {
        result.per_thread_pair_work[t] = static_cast<uint64_t>(per_thread_candidates[t].size());
    }
    result.per_thread_units_processed = std::move(per_thread_units);
    diag.per_thread_pair_work = result.per_thread_pair_work;
    diag.per_thread_units_processed = result.per_thread_units_processed;

    size_t total_raw_candidates = 0;
    for (const auto& pt_vec : per_thread_candidates) {
        total_raw_candidates += pt_vec.size();
    }

    std::vector<CandidatePair> raw_candidates;
    raw_candidates.reserve(total_raw_candidates);
    for (auto& pt_vec : per_thread_candidates) {
        raw_candidates.insert(
            raw_candidates.end(),
            std::make_move_iterator(pt_vec.begin()),
            std::make_move_iterator(pt_vec.end())
        );
    }

    std::sort(raw_candidates.begin(), raw_candidates.end());
    raw_candidates.erase(
        std::unique(raw_candidates.begin(), raw_candidates.end()),
        raw_candidates.end()
    );
    result.candidate_pairs = std::move(raw_candidates);

    auto t3e_end = SteadyClock::now();
    timings.merge_sort_dedup_ms = elapsed_ms(t3e_start, t3e_end);

    // -------------------------------------------------------------
    // Stage 4: Parallel Hamming Distance Verification
    // -------------------------------------------------------------
    auto t4_start = SteadyClock::now();

    size_t num_candidates = result.candidate_pairs.size();
    std::vector<std::vector<VerifiedPair>> per_thread_verified(config.thread_count);

    if (num_candidates > 0) {
        detail::OmpScheduleGuard verify_guard(SchedulePolicy::Static, 0);

        #pragma omp parallel num_threads(config.thread_count) default(none) \
            shared(result, num_candidates, id_to_index, config, per_thread_verified)
        {
            int tid = omp_get_thread_num();

            #pragma omp single
            {
                int actual_threads = omp_get_num_threads();
                if (actual_threads > static_cast<int>(per_thread_verified.size())) {
                    per_thread_verified.resize(actual_threads);
                }
            }

            #pragma omp for schedule(runtime)
            for (size_t c = 0; c < num_candidates; ++c) {
                const auto& candidate = result.candidate_pairs[c];
                size_t idx1 = id_to_index.at(candidate.first);
                size_t idx2 = id_to_index.at(candidate.second);

                uint64_t fp1 = result.fingerprints[idx1];
                uint64_t fp2 = result.fingerprints[idx2];

                int dist = hamming_distance(fp1, fp2);
                if (dist <= config.base_config.max_hamming_threshold) {
                    per_thread_verified[tid].push_back(
                        VerifiedPair{candidate.first, candidate.second, dist}
                    );
                }
            }
        }
    }

    size_t total_verified = 0;
    for (const auto& pt_vec : per_thread_verified) {
        total_verified += pt_vec.size();
    }

    result.verified_pairs.reserve(total_verified);
    for (auto& pt_vec : per_thread_verified) {
        result.verified_pairs.insert(
            result.verified_pairs.end(),
            std::make_move_iterator(pt_vec.begin()),
            std::make_move_iterator(pt_vec.end())
        );
    }

    std::sort(result.verified_pairs.begin(), result.verified_pairs.end());

    auto t4_end = SteadyClock::now();
    timings.verification_ms = elapsed_ms(t4_start, t4_end);

    auto pipeline_end = SteadyClock::now();
    timings.total_pipeline_ms = elapsed_ms(pipeline_start, pipeline_end);

    // Populate summary timings on PipelineResult for compatibility with StageTimings
    result.timings.fingerprinting_ms = timings.fingerprinting_ms;
    result.timings.bucket_construction_ms = timings.bucket_construction_stats_ms;
    result.timings.candidate_generation_dedup_ms =
        timings.adaptive_decision_ms +
        timings.ordinary_candidate_gen_ms +
        timings.heavy_chunk_generation_ms +
        timings.heavy_pair_processing_ms +
        timings.merge_sort_dedup_ms;
    result.timings.verification_ms = timings.verification_ms;
    result.timings.total_pipeline_ms = timings.total_pipeline_ms;

    return adaptive_res;
}
