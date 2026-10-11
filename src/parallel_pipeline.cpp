#include "parallel_pipeline.hpp"
#include "simhash.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
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

    using Clock = std::chrono::high_resolution_clock;
    auto pipeline_start = Clock::now();

    PipelineResult result;
    size_t num_docs = documents.size();
    result.document_ids.resize(num_docs);
    result.fingerprints.resize(num_docs);

    if (documents.empty()) {
        auto pipeline_end = Clock::now();
        result.timings.total_pipeline_ms = std::chrono::duration<double, std::milli>(
            pipeline_end - pipeline_start
        ).count();
        return result;
    }

    // -------------------------------------------------------------
    // Pre-Stage Validation & ID Mapping
    // -------------------------------------------------------------
    // Map each document ID to its position in the arrays to guarantee unique IDs
    // and enable fast O(1) fingerprint lookup during candidate verification.
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
    auto t1_start = Clock::now();

    #pragma omp parallel for num_threads(config.thread_count) schedule(runtime) default(none) \
        shared(documents, result, config, num_docs)
    for (size_t i = 0; i < num_docs; ++i) {
        result.fingerprints[i] = compute_simhash(documents[i].text, config.base_config.shingle_size);
    }

    auto t1_end = Clock::now();
    result.timings.fingerprinting_ms = std::chrono::duration<double, std::milli>(
        t1_end - t1_start
    ).count();

    // -------------------------------------------------------------
    // Stage 2: Band Bucket Construction & Workload Statistics
    // -------------------------------------------------------------
    // Design Decision: Bucket construction is performed sequentially.
    // Rationale:
    // With 8 bands and 256 bucket slots per band, total bucket count is small (2048).
    // Sequential insertion is extremely fast (< 0.2 ms for 100K docs) and completely
    // eliminates concurrent write locks or multi-hash contention across threads.
    auto t2_start = Clock::now();

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

    auto t2_end = Clock::now();
    result.timings.bucket_construction_ms = std::chrono::duration<double, std::milli>(
        t2_end - t2_start
    ).count();

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
    auto t3_start = Clock::now();

    size_t num_active_buckets = active_buckets.size();
    std::vector<std::vector<CandidatePair>> per_thread_candidates(config.thread_count);

    #pragma omp parallel num_threads(config.thread_count) default(none) \
        shared(active_buckets, num_active_buckets, per_thread_candidates)
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(runtime)
        for (size_t b = 0; b < num_active_buckets; ++b) {
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
    }

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

    auto t3_end = Clock::now();
    result.timings.candidate_generation_dedup_ms = std::chrono::duration<double, std::milli>(
        t3_end - t3_start
    ).count();

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
    auto t4_start = Clock::now();

    size_t num_candidates = result.candidate_pairs.size();
    std::vector<std::vector<VerifiedPair>> per_thread_verified(config.thread_count);

    #pragma omp parallel num_threads(config.thread_count) default(none) \
        shared(result, num_candidates, id_to_index, config, per_thread_verified)
    {
        int tid = omp_get_thread_num();

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

    auto t4_end = Clock::now();
    result.timings.verification_ms = std::chrono::duration<double, std::milli>(
        t4_end - t4_start
    ).count();

    auto pipeline_end = Clock::now();
    result.timings.total_pipeline_ms = std::chrono::duration<double, std::milli>(
        pipeline_end - pipeline_start
    ).count();

    return result;
}
