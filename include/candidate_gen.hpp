#ifndef CANDIDATE_GEN_HPP
#define CANDIDATE_GEN_HPP

#include "document.hpp"
#include "simhash.hpp"
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

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
 * @brief Elapsed wall-clock timings for each stage of the sequential pipeline in milliseconds.
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
};

/**
 * @brief Comprehensive results produced by running the sequential pipeline.
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
    std::vector<uint64_t> per_thread_pair_work;   ///< Estimated pair positions (W_t) assigned per worker thread.
    std::vector<size_t> per_thread_units_processed; ///< Positive-work buckets/chunks processed per worker thread.
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
