#ifndef CANDIDATE_GEN_HPP
#define CANDIDATE_GEN_HPP

#include "document.hpp"
#include "simhash.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Monotonic clock alias used for all elapsed-time measurements across pipelines and benchmarks.
 *
 * std::chrono::steady_clock is guaranteed to be monotonic and unaffected by system wall-clock adjustments.
 */
using SteadyClock = std::chrono::steady_clock;

/**
 * @brief Converts the duration between two SteadyClock time points into milliseconds.
 */
inline double elapsed_ms(SteadyClock::time_point start, SteadyClock::time_point end) noexcept {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * @brief Default parameters for the candidate-generation pipeline.
 */
inline constexpr size_t kNumBands = 8;          ///< 8 non-overlapping bands for 64-bit fingerprint
inline constexpr size_t kBandBits = 8;          ///< 8 bits per band (8 * 8 = 64 bits)
inline constexpr int kDefaultHammingThreshold = 3; ///< Default maximum Hamming distance for duplicate match

/**
 * @brief Represents an unordered pair of documents by their unique IDs.
 *
 * Invariant: `first < second` (normalized).
 */
struct CandidatePair {
    uint64_t first{0};
    uint64_t second{0};

    bool operator==(const CandidatePair& other) const noexcept {
        return first == other.first && second == other.second;
    }

    bool operator<(const CandidatePair& other) const noexcept {
        if (first != other.first) {
            return first < other.first;
        }
        return second < other.second;
    }
};

/**
 * @brief Represents a verified near-duplicate pair with its measured Hamming distance.
 */
struct VerifiedPair {
    uint64_t first{0};
    uint64_t second{0};
    int distance{0};

    bool operator==(const VerifiedPair& other) const noexcept {
        return first == other.first && second == other.second && distance == other.distance;
    }

    bool operator<(const VerifiedPair& other) const noexcept {
        if (first != other.first) {
            return first < other.first;
        }
        if (second != other.second) {
            return second < other.second;
        }
        return distance < other.distance;
    }
};

/**
 * @brief Represents a single candidate bucket identified by band index and band value.
 */
struct BucketInfo {
    size_t band_index{0};           ///< Band index in [0, kNumBands - 1].
    uint8_t band_value{0};          ///< 8-bit band value in [0, 255].
    size_t document_count{0};       ///< Number of documents in this bucket (k_i).
    uint64_t estimated_pair_work{0};///< Pairwise comparison cost: k_i * (k_i - 1) / 2.
};

/**
 * @brief Workload statistics summarizing the distribution of candidate buckets.
 */
struct WorkloadStatistics {
    size_t occupied_buckets{0};           ///< Buckets containing at least 1 document across all 8 bands.
    size_t positive_work_buckets{0};       ///< Buckets containing at least 2 documents (work > 0).
    size_t max_bucket_size{0};            ///< Maximum documents in any single bucket.
    double mean_bucket_size{0.0};         ///< Mean document count across all occupied buckets.
    uint64_t max_estimated_pair_work{0};  ///< Maximum estimated pair work across all buckets.
    double mean_estimated_pair_work{0.0}; ///< Mean estimated pair work across positive-work buckets.
    double cv_estimated_pair_work{0.0};   ///< Coefficient of variation (population SD / mean) of positive-work buckets.

    void print(std::ostream& os) const;
};

/**
 * @brief Elapsed monotonic timings (std::chrono::steady_clock) for each stage of the pipeline in milliseconds.
 *
 * Timing Scope Notes:
 * - Stages 1 through 4 are strictly non-overlapping intervals.
 * - Across `run_sequential_pipeline`, `run_parallel_pipeline`, and `run_adaptive_pipeline`,
 *   sequential `id_to_index` construction and duplicate-ID validation occur in a pre-stage step
 *   before Stage 1 (`t1_start`) so that `fingerprinting_ms` consistently measures strictly
 *   SimHash fingerprinting across all implementations.
 * - `total_pipeline_ms` measures end-to-end execution from pipeline entry to return (including
 *   initial vector allocation and pre-stage ID validation/indexing), and therefore slightly exceeds
 *   the sum of the four individual stage timings.
 */
struct StageTimings {
    double fingerprinting_ms{0.0};
    double bucket_construction_ms{0.0};
    double candidate_generation_dedup_ms{0.0}; ///< Candidate pair expansion and deduplication
    double verification_ms{0.0};
    double total_pipeline_ms{0.0};

    void print(std::ostream& os) const;
};

/**
 * @brief Configuration parameters for the sequential detection pipeline.
 */
struct PipelineConfig {
    size_t shingle_size{kDefaultShingleSize};       ///< Token shingle size for SimHash (default: 3).
    int max_hamming_threshold{kDefaultHammingThreshold}; ///< Distance threshold for match (default: 3).
    bool enable_band_suppression{false};            ///< Opt-in band-aware duplicate suppression (Phase 10, default: false).
};

/**
 * @brief Comprehensive results produced by running the sequential or parallel pipeline.
 */
struct PipelineResult {
    std::vector<uint64_t> document_ids;           ///< Associated document IDs.
    std::vector<uint64_t> fingerprints;           ///< 64-bit SimHash fingerprints.
    std::vector<CandidatePair> candidate_pairs;   ///< Deduplicated, sorted candidate pairs.
    std::vector<VerifiedPair> verified_pairs;     ///< Verified near-duplicate pairs (distance <= threshold).
    std::vector<BucketInfo> buckets;              ///< Non-empty bucket records.
    WorkloadStatistics workload_stats;            ///< Workload distribution statistics.
    StageTimings timings;                         ///< Stage timings in milliseconds.
    int actual_thread_count{1};                   ///< Worker threads in team (1 for sequential).
    std::vector<uint64_t> per_thread_pair_work;   ///< Estimated pair positions (W_t) evaluated per worker thread.
    std::vector<size_t> per_thread_units_processed; ///< Positive-work buckets/chunks processed per worker thread.
    bool band_suppression_enabled{false};         ///< Whether band-aware duplicate suppression was active.
    uint64_t raw_candidates_evaluated{0};         ///< Total bucket pair positions evaluated across all bands.
    uint64_t raw_candidates_emitted{0};           ///< Total raw candidate pairs emitted before Stage 3e aggregation.
    uint64_t suppressed_pair_count{0};            ///< Duplicate pair emissions suppressed across bands 1..7.
    std::vector<uint64_t> per_thread_emitted_pairs;    ///< Raw candidate pairs emitted per worker thread.
    std::vector<uint64_t> per_thread_suppressed_pairs; ///< Duplicate pairs suppressed per worker thread.
};

/**
 * @brief Extracts an 8-bit band value from a 64-bit fingerprint.
 *
 * Band 0: bits 0..7
 * Band 1: bits 8..15
 * ...
 * Band 7: bits 56..63
 *
 * @param fingerprint The 64-bit SimHash fingerprint.
 * @param band_index The band index in [0, 7].
 * @return uint8_t The 8-bit value of the specified band.
 */
uint8_t extract_band(uint64_t fingerprint, size_t band_index) noexcept;

/**
 * @brief Returns the SWAR high-bit mask for all bytes strictly below `band_index` (0 <= b' < band_index).
 *
 * For band_index == 0, returns 0ULL.
 * For band_index in [1, 7], sets bit (8*b' + 7) for each earlier band b' in [0, band_index - 1].
 */
inline constexpr uint64_t earlier_band_msb_mask(size_t band_index) noexcept {
    if (band_index == 0) {
        return 0ULL;
    }
    if (band_index >= kNumBands) {
        return 0x8080808080808080ULL;
    }
    return 0x8080808080808080ULL >> (64 - band_index * 8);
}

/**
 * @brief Branch-free O(1) SWAR zero-byte test determining whether two 64-bit fingerprints
 *        share an identical 8-bit band value in any earlier band covered by `earlier_msb_mask`.
 *
 * Exactness invariant:
 * - If all bytes below `band_index` of `xor_fp = fp_u ^ fp_v` are non-zero (>= 1), subtracting
 *   0x01 from each byte never borrows across byte boundaries below `band_index`, and
 *   ((byte - 1) & ~byte & 0x80) == 0 for every non-zero byte in [1, 255].
 * - If at least one earlier byte is 0x00, the lowest-index zero byte b'_0 < band_index receives
 *   no borrow from below, so ((0x00 - 0x01) & ~0x00) == 0xFF sets bit (8*b'_0 + 7), which is
 *   captured by `earlier_msb_mask`.
 */
inline constexpr bool matched_in_earlier_band_swar_mask(
    uint64_t fp_u,
    uint64_t fp_v,
    uint64_t earlier_msb_mask
) noexcept {
    const uint64_t xor_fp = fp_u ^ fp_v;
    return (((xor_fp - 0x0101010101010101ULL) & ~xor_fp) & earlier_msb_mask) != 0ULL;
}

/**
 * @brief Branch-free O(1) SWAR test determining whether two fingerprints matched in any band b' < band_index.
 */
inline constexpr bool matched_in_earlier_band_swar(
    uint64_t fp_u,
    uint64_t fp_v,
    size_t band_index
) noexcept {
    if (band_index == 0) {
        return false;
    }
    return matched_in_earlier_band_swar_mask(fp_u, fp_v, earlier_band_msb_mask(band_index));
}

/**
 * @brief Reference byte-by-byte check determining whether two fingerprints matched in any band b' < band_index.
 */
inline bool matched_in_earlier_band_ref(
    uint64_t fp_u,
    uint64_t fp_v,
    size_t band_index
) noexcept {
    const size_t limit = (band_index < kNumBands) ? band_index : kNumBands;
    for (size_t b = 0; b < limit; ++b) {
        if (extract_band(fp_u, b) == extract_band(fp_v, b)) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Executes the end-to-end sequential near-duplicate detection pipeline.
 *
 * 1. Fingerprinting: Computes 64-bit SimHash for each document.
 * 2. Bucket Construction: Hashes fingerprints into 8 bands of 8 bits each (256 buckets per band).
 * 3. Candidate Generation & Deduplication: Generates normalized (first < second) pairs
 *    within each bucket and deduplicates them into a sorted candidate list.
 * 4. Verification: Evaluates exact bitwise Hamming distance for each candidate pair;
 *    retains pairs where distance <= max_hamming_threshold.
 * 5. Statistics & Timings: Calculates bucket workload metrics and stage execution runtimes.
 *
 * @param documents Input document collection.
 * @param config Pipeline configuration.
 * @return PipelineResult Complete pipeline output.
 */
PipelineResult run_sequential_pipeline(const std::vector<Document>& documents,
                                       const PipelineConfig& config = PipelineConfig{});

#endif // CANDIDATE_GEN_HPP
