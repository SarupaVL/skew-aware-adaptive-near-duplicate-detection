#include "parallel_pipeline.hpp"
#include "simhash.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
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

namespace {

/**
 * @brief Slice descriptor for dividing uneven per-thread candidate buffers into bounded
 *        contiguous ranges so all OpenMP threads participate evenly during counting and scattering.
 */
struct CandidateBufferSlice {
    size_t buffer_idx{0};
    size_t start_offset{0};
    size_t end_offset{0};
};

inline constexpr size_t kScatterSlicePairs = 65536; ///< Max pairs per scatter slice

/**
 * @brief Parallel fallback for sorting and deduplicating a single heavily skewed partition
 *        in-place across multiple OpenMP threads using sub-chunk sort/unique + inplace_merge.
 */
size_t parallel_sort_dedup_oversized_partition(
    CandidatePair* begin_ptr,
    size_t count,
    int thread_count
) {
    if (count <= 1) {
        return count;
    }

    size_t num_chunks = std::min<size_t>(
        static_cast<size_t>(thread_count),
        std::max<size_t>(1, count / 1024)
    );
    if (num_chunks <= 1) {
        std::sort(begin_ptr, begin_ptr + count);
        CandidatePair* uend = std::unique(begin_ptr, begin_ptr + count);
        return static_cast<size_t>(uend - begin_ptr);
    }

    std::vector<size_t> sub_starts(num_chunks + 1, 0);
    std::vector<size_t> sub_unique_lens(num_chunks, 0);
    for (size_t c = 0; c < num_chunks; ++c) {
        sub_starts[c] = (c * count) / num_chunks;
    }
    sub_starts[num_chunks] = count;

    // 1. Sort and deduplicate each sub-chunk in parallel
    #pragma omp parallel for num_threads(thread_count) schedule(static) default(none) \
        shared(begin_ptr, num_chunks, sub_starts, sub_unique_lens)
    for (size_t c = 0; c < num_chunks; ++c) {
        CandidatePair* c_begin = begin_ptr + sub_starts[c];
        CandidatePair* c_end = begin_ptr + sub_starts[c + 1];
        std::sort(c_begin, c_end);
        CandidatePair* c_uend = std::unique(c_begin, c_end);
        sub_unique_lens[c] = static_cast<size_t>(c_uend - c_begin);
    }

    // 2. Compact deduplicated sub-chunks into contiguous adjacent segments
    std::vector<size_t> compact_offs(num_chunks + 1, 0);
    for (size_t c = 0; c < num_chunks; ++c) {
        compact_offs[c + 1] = compact_offs[c] + sub_unique_lens[c];
        if (c > 0 && sub_unique_lens[c] > 0 && compact_offs[c] < sub_starts[c]) {
            std::copy(
                begin_ptr + sub_starts[c],
                begin_ptr + sub_starts[c] + sub_unique_lens[c],
                begin_ptr + compact_offs[c]
            );
        }
    }

    size_t total_compact = compact_offs[num_chunks];
    if (total_compact <= 1) {
        return total_compact;
    }

    // 3. Bottom-up pairwise inplace_merge across compacted sorted segments
    for (size_t width = 1; width < num_chunks; width *= 2) {
        for (size_t i = 0; i + width < num_chunks; i += 2 * width) {
            size_t left = compact_offs[i];
            size_t mid = compact_offs[i + width];
            size_t right = compact_offs[std::min(i + 2 * width, num_chunks)];
            if (left < mid && mid < right) {
                std::inplace_merge(begin_ptr + left, begin_ptr + mid, begin_ptr + right);
            }
        }
    }

    CandidatePair* final_end = std::unique(begin_ptr, begin_ptr + total_compact);
    return static_cast<size_t>(final_end - begin_ptr);
}

} // namespace

std::vector<CandidatePair> aggregate_and_deduplicate_candidates(
    std::vector<std::vector<CandidatePair>>& worker_buffers,
    int thread_count,
    size_t requested_partitions
) {
    if (thread_count <= 0) {
        throw std::invalid_argument(
            "aggregate_and_deduplicate_candidates: thread_count must be positive (got " +
            std::to_string(thread_count) + ")."
        );
    }

    size_t total_raw = 0;
    size_t non_empty_buffers = 0;
    size_t single_non_empty_idx = 0;

    for (size_t b = 0; b < worker_buffers.size(); ++b) {
        size_t sz = worker_buffers[b].size();
        if (sz > 0) {
            total_raw += sz;
            ++non_empty_buffers;
            single_non_empty_idx = b;
        }
    }

    if (total_raw == 0) {
        for (auto& buf : worker_buffers) {
            std::vector<CandidatePair>().swap(buf);
        }
        return {};
    }

    // Determine effective number of ordered range partitions
    size_t num_partitions = requested_partitions;
    if (num_partitions == 0) {
        if (thread_count == 1 || total_raw < kMinParallelAggregationPairs) {
            num_partitions = 1;
        } else {
            size_t target = static_cast<size_t>(thread_count) * kPartitionsPerThread;
            num_partitions = std::min(kMaxAggregationPartitions, std::max<size_t>(2, target));
            num_partitions = std::min(num_partitions, total_raw);
        }
    }

    // Fast path for single partition (or T=1 / small input):
    // Moves a single non-empty worker buffer directly in-place without allocating a second raw vector.
    if (num_partitions <= 1) {
        std::vector<CandidatePair> result;
        if (non_empty_buffers == 1) {
            result = std::move(worker_buffers[single_non_empty_idx]);
        } else {
            result = std::move(worker_buffers[0]);
            result.reserve(total_raw);
            for (size_t b = 1; b < worker_buffers.size(); ++b) {
                if (!worker_buffers[b].empty()) {
                    result.insert(
                        result.end(),
                        std::make_move_iterator(worker_buffers[b].begin()),
                        std::make_move_iterator(worker_buffers[b].end())
                    );
                }
                std::vector<CandidatePair>().swap(worker_buffers[b]);
            }
        }
        for (auto& buf : worker_buffers) {
            std::vector<CandidatePair>().swap(buf);
        }

        if (thread_count > 1 && result.size() >= kMinParallelAggregationPairs) {
            size_t ucount = parallel_sort_dedup_oversized_partition(
                result.data(), result.size(), thread_count
            );
            result.resize(ucount);
        } else {
            std::sort(result.begin(), result.end());
            result.erase(std::unique(result.begin(), result.end()), result.end());
        }
        return result;
    }

    // Build bounded contiguous slices across non-empty worker buffers so counting and scattering
    // remain load-balanced across threads even if worker_buffers have uneven sizes.
    std::vector<CandidateBufferSlice> slices;
    slices.reserve(non_empty_buffers + (total_raw / kScatterSlicePairs) + 1);
    for (size_t b = 0; b < worker_buffers.size(); ++b) {
        size_t sz = worker_buffers[b].size();
        size_t off = 0;
        while (off < sz) {
            size_t len = std::min(kScatterSlicePairs, sz - off);
            slices.push_back(CandidateBufferSlice{b, off, off + len});
            off += len;
        }
    }
    size_t num_slices = slices.size();

    // Pass 1a: Parallel min/max reduction on first and second document IDs
    uint64_t min_first = std::numeric_limits<uint64_t>::max();
    uint64_t max_first = 0;
    uint64_t min_second = std::numeric_limits<uint64_t>::max();
    uint64_t max_second = 0;

    #pragma omp parallel for num_threads(thread_count) schedule(static) default(none) \
        shared(slices, num_slices, worker_buffers) \
        reduction(min: min_first, min_second) reduction(max: max_first, max_second)
    for (size_t s = 0; s < num_slices; ++s) {
        const auto& sl = slices[s];
        const CandidatePair* ptr = worker_buffers[sl.buffer_idx].data() + sl.start_offset;
        size_t len = sl.end_offset - sl.start_offset;
        for (size_t i = 0; i < len; ++i) {
            uint64_t f = ptr[i].first;
            uint64_t sec = ptr[i].second;
            if (f < min_first) min_first = f;
            if (f > max_first) max_first = f;
            if (sec < min_second) min_second = sec;
            if (sec > max_second) max_second = sec;
        }
    }

    // If every candidate pair in the entire collection is identical, return that 1 unique pair immediately
    if (min_first == max_first && min_second == max_second) {
        for (auto& buf : worker_buffers) {
            std::vector<CandidatePair>().swap(buf);
        }
        return {CandidatePair{min_first, min_second}};
    }

    // Partition by `first` ID when multiple distinct `first` values exist;
    // if all pairs share the exact same `first` ID, partition by `second` ID.
    // Both preserve the strict lexicographical total order of CandidatePair::operator<.
    const bool partition_by_first = (min_first < max_first);
    const uint64_t min_key = partition_by_first ? min_first : min_second;
    const uint64_t max_key = partition_by_first ? max_first : max_second;
    const uint64_t span = max_key - min_key;

    // Safe ceiling division: bucket_width = floor(span / num_partitions) + 1 >= (span + 1) / num_partitions.
    // Never overflows uint64_t since num_partitions >= 2.
    const uint64_t bucket_width = (span / static_cast<uint64_t>(num_partitions)) + 1ULL;

    // Pass 1b: Count per-slice partition histograms in parallel
    std::vector<size_t> slice_counts(num_slices * num_partitions, 0);

    #pragma omp parallel for num_threads(thread_count) schedule(static) default(none) \
        shared(slices, num_slices, worker_buffers, slice_counts, \
               num_partitions, min_key, bucket_width, partition_by_first)
    for (size_t s = 0; s < num_slices; ++s) {
        const auto& sl = slices[s];
        const CandidatePair* src = worker_buffers[sl.buffer_idx].data() + sl.start_offset;
        size_t len = sl.end_offset - sl.start_offset;
        size_t* counts_row = &slice_counts[s * num_partitions];

        for (size_t i = 0; i < len; ++i) {
            uint64_t key = partition_by_first ? src[i].first : src[i].second;
            size_t p = static_cast<size_t>((key - min_key) / bucket_width);
            if (p >= num_partitions) {
                p = num_partitions - 1;
            }
            ++counts_row[p];
        }
    }

    // Compute contiguous partition offsets and disjoint per-slice write offsets
    std::vector<size_t> part_offsets(num_partitions + 1, 0);
    std::vector<size_t> slice_write_offsets(num_slices * num_partitions, 0);
    size_t running_offset = 0;
    for (size_t p = 0; p < num_partitions; ++p) {
        part_offsets[p] = running_offset;
        for (size_t s = 0; s < num_slices; ++s) {
            slice_write_offsets[s * num_partitions + p] = running_offset;
            running_offset += slice_counts[s * num_partitions + p];
        }
    }
    part_offsets[num_partitions] = running_offset;

    // Pass 2: Parallel lock-free scatter into contiguous range-partitioned workspace
    std::vector<CandidatePair> partitioned_raw(total_raw);

    #pragma omp parallel for num_threads(thread_count) schedule(static) default(none) \
        shared(slices, num_slices, worker_buffers, partitioned_raw, slice_write_offsets, \
               num_partitions, min_key, bucket_width, partition_by_first)
    for (size_t s = 0; s < num_slices; ++s) {
        const auto& sl = slices[s];
        const CandidatePair* src = worker_buffers[sl.buffer_idx].data() + sl.start_offset;
        size_t len = sl.end_offset - sl.start_offset;
        size_t* out_pos = &slice_write_offsets[s * num_partitions];

        for (size_t i = 0; i < len; ++i) {
            const CandidatePair& cp = src[i];
            uint64_t key = partition_by_first ? cp.first : cp.second;
            size_t p = static_cast<size_t>((key - min_key) / bucket_width);
            if (p >= num_partitions) {
                p = num_partitions - 1;
            }
            partitioned_raw[out_pos[p]++] = cp;
        }
    }

    // Release per-thread raw buffers immediately before sorting and output allocation
    for (auto& buf : worker_buffers) {
        std::vector<CandidatePair>().swap(buf);
    }

    // Pass 3: Sort and deduplicate each partition in-place.
    // - If a single skewed partition holds > 50% of all pairs (and >= kMinParallelAggregationPairs),
    //   it is sorted and deduplicated across all threads via parallel_sort_dedup_oversized_partition.
    // - All remaining positive-size partitions are sorted in Longest-Processing-Time-First (LPT)
    //   order under OpenMP schedule(dynamic, 1) to mitigate inter-partition skew.
    std::vector<size_t> unique_counts(num_partitions, 0);
    std::vector<size_t> regular_partitions;
    regular_partitions.reserve(num_partitions);

    for (size_t p = 0; p < num_partitions; ++p) {
        size_t count = part_offsets[p + 1] - part_offsets[p];
        if (count == 0) {
            unique_counts[p] = 0;
        } else if (count == 1) {
            unique_counts[p] = 1;
        } else if (thread_count > 1 &&
                   count >= kMinParallelAggregationPairs &&
                   count > (total_raw / 2)) {
            unique_counts[p] = parallel_sort_dedup_oversized_partition(
                partitioned_raw.data() + part_offsets[p], count, thread_count
            );
        } else {
            regular_partitions.push_back(p);
        }
    }

    // Order regular partitions from largest to smallest (LPT) so larger partitions start first
    std::sort(regular_partitions.begin(), regular_partitions.end(),
              [&](size_t a, size_t b) {
                  size_t ca = part_offsets[a + 1] - part_offsets[a];
                  size_t cb = part_offsets[b + 1] - part_offsets[b];
                  if (ca != cb) {
                      return ca > cb;
                  }
                  return a < b;
              });

    size_t num_regular = regular_partitions.size();
    if (num_regular > 0) {
        #pragma omp parallel for num_threads(thread_count) schedule(dynamic, 1) default(none) \
            shared(regular_partitions, num_regular, part_offsets, partitioned_raw, unique_counts)
        for (size_t idx = 0; idx < num_regular; ++idx) {
            size_t p = regular_partitions[idx];
            CandidatePair* begin_ptr = partitioned_raw.data() + part_offsets[p];
            CandidatePair* end_ptr = partitioned_raw.data() + part_offsets[p + 1];
            std::sort(begin_ptr, end_ptr);
            CandidatePair* uend = std::unique(begin_ptr, end_ptr);
            unique_counts[p] = static_cast<size_t>(uend - begin_ptr);
        }
    }

    // Pass 4: Compute output prefix sums in canonical partition order 0 .. P-1 and compact in parallel
    std::vector<size_t> out_offsets(num_partitions + 1, 0);
    for (size_t p = 0; p < num_partitions; ++p) {
        out_offsets[p + 1] = out_offsets[p] + unique_counts[p];
    }
    size_t total_unique = out_offsets[num_partitions];

    std::vector<CandidatePair> result(total_unique);

    #pragma omp parallel for num_threads(thread_count) schedule(dynamic, 1) default(none) \
        shared(num_partitions, part_offsets, out_offsets, unique_counts, partitioned_raw, result)
    for (size_t p = 0; p < num_partitions; ++p) {
        size_t ucount = unique_counts[p];
        if (ucount > 0) {
            const CandidatePair* src = partitioned_raw.data() + part_offsets[p];
            CandidatePair* dst = result.data() + out_offsets[p];
            std::copy(src, src + ucount, dst);
        }
    }

    return result;
}

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

    // Aggregate, sort, and deduplicate per-thread candidate lists in parallel using ordered range partitioning
    result.candidate_pairs = aggregate_and_deduplicate_candidates(
        per_thread_candidates, config.thread_count
    );

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
    // Stage 3e: Parallel Candidate Aggregation, Sort & Deduplication
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

    result.candidate_pairs = aggregate_and_deduplicate_candidates(
        per_thread_candidates, config.thread_count
    );

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
