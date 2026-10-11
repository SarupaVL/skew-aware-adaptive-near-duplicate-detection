#include "candidate_gen.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <unordered_map>

uint8_t extract_band(uint64_t fingerprint, size_t band_index) noexcept {
    // Each band spans 8 bits; shift by (band_index * 8) and mask with 0xFF
    return static_cast<uint8_t>((fingerprint >> (band_index * 8)) & 0xFFULL);
}

void WorkloadStatistics::print(std::ostream& os) const {
    os << "---------------------------------------------------------\n";
    os << " SimHash Bucket Workload Statistics (8 bands x 8 bits)\n";
    os << "---------------------------------------------------------\n";
    os << "  Occupied Buckets (k >= 1)         : " << occupied_buckets << "\n";
    os << "  Buckets with Pair Work (k >= 2)   : " << positive_work_buckets << "\n";
    os << "  Max Bucket Size                   : " << max_bucket_size << "\n";
    os << "  Mean Bucket Size (Occupied)       : " << std::fixed << std::setprecision(2)
       << mean_bucket_size << "\n";
    os << "  Max Estimated Bucket Pair Work    : " << max_estimated_pair_work << "\n";
    os << "  Mean Estimated Pair Work (k >= 2) : " << mean_estimated_pair_work << "\n";
    os << "  Coefficient of Variation (CV)     : " << cv_estimated_pair_work << "\n";
    os << "---------------------------------------------------------\n";
}

void StageTimings::print(std::ostream& os) const {
    os << "---------------------------------------------------------\n";
    os << " Sequential Pipeline Stage Timings\n";
    os << "---------------------------------------------------------\n";
    os << "  1. Fingerprinting                 : " << std::fixed << std::setprecision(3)
       << fingerprinting_ms << " ms\n";
    os << "  2. Bucket Construction            : " << bucket_construction_ms << " ms\n";
    os << "  3. Candidate Gen & Deduplication  : " << candidate_generation_dedup_ms << " ms\n";
    os << "  4. Hamming Distance Verification  : " << verification_ms << " ms\n";
    os << "  -------------------------------------------------------\n";
    os << "  Total Pipeline Runtime            : " << total_pipeline_ms << " ms\n";
    os << "---------------------------------------------------------\n";
}

PipelineResult run_sequential_pipeline(const std::vector<Document>& documents,
                                       const PipelineConfig& config) {
    using Clock = std::chrono::high_resolution_clock;
    auto pipeline_start = Clock::now();

    PipelineResult result;
    size_t num_docs = documents.size();
    result.document_ids.reserve(num_docs);
    result.fingerprints.reserve(num_docs);

    if (documents.empty()) {
        result.actual_thread_count = 1;
        result.per_thread_pair_work = {0ULL};
        result.per_thread_units_processed = {0};
        auto pipeline_end = Clock::now();
        result.timings.total_pipeline_ms = std::chrono::duration<double, std::milli>(
            pipeline_end - pipeline_start
        ).count();
        return result;
    }

    // -------------------------------------------------------------
    // Stage 1: Document Fingerprinting
    // -------------------------------------------------------------
    auto t1_start = Clock::now();

    // Map each document ID to its position in result arrays for fast fingerprint lookup
    std::unordered_map<uint64_t, size_t> id_to_index;
    id_to_index.reserve(num_docs);

    for (size_t i = 0; i < num_docs; ++i) {
        const auto& doc = documents[i];
        auto insert_res = id_to_index.emplace(doc.id, i);
        if (!insert_res.second) {
            throw std::invalid_argument(
                "run_sequential_pipeline: Duplicate document ID detected (" +
                std::to_string(doc.id) + "). All document IDs must be unique."
            );
        }
        result.document_ids.push_back(doc.id);
        result.fingerprints.push_back(compute_simhash(doc.text, config.shingle_size));
    }

    auto t1_end = Clock::now();
    result.timings.fingerprinting_ms = std::chrono::duration<double, std::milli>(
        t1_end - t1_start
    ).count();

    // -------------------------------------------------------------
    // Stage 2: Band Bucket Construction
    // -------------------------------------------------------------
    auto t2_start = Clock::now();

    // 8 bands, each with 256 buckets (possible values 0..255).
    // buckets_table[band][value] stores the list of document IDs mapped to that bucket.
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

        // Coefficient of variation (CV = population SD / mean)
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
    // Stage 3: Candidate Generation & Deduplication
    // -------------------------------------------------------------
    auto t3_start = Clock::now();

    std::vector<CandidatePair> raw_candidates;
    size_t units_processed = 0;

    for (size_t band = 0; band < kNumBands; ++band) {
        for (size_t val = 0; val < 256; ++val) {
            const auto& bucket_docs = buckets_table[band][val];
            if (bucket_docs.size() < 2) {
                continue;
            }
            ++units_processed;

            // Generate all unordered pairs within this bucket
            for (size_t i = 0; i < bucket_docs.size(); ++i) {
                for (size_t j = i + 1; j < bucket_docs.size(); ++j) {
                    uint64_t u = bucket_docs[i];
                    uint64_t v = bucket_docs[j];
                    if (u != v) {
                        // Normalize pair so smaller ID comes first
                        CandidatePair pair = (u < v) ? CandidatePair{u, v} : CandidatePair{v, u};
                        raw_candidates.push_back(pair);
                    }
                }
            }
        }
    }

    result.actual_thread_count = 1;
    result.per_thread_pair_work = {static_cast<uint64_t>(raw_candidates.size())};
    result.per_thread_units_processed = {units_processed};

    // Sort and deduplicate candidate pairs deterministically
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
    // Stage 4: Hamming Distance Verification
    // -------------------------------------------------------------
    auto t4_start = Clock::now();

    for (const auto& candidate : result.candidate_pairs) {
        size_t idx1 = id_to_index[candidate.first];
        size_t idx2 = id_to_index[candidate.second];

        uint64_t fp1 = result.fingerprints[idx1];
        uint64_t fp2 = result.fingerprints[idx2];

        int dist = hamming_distance(fp1, fp2);
        if (dist <= config.max_hamming_threshold) {
            result.verified_pairs.push_back(VerifiedPair{candidate.first, candidate.second, dist});
        }
    }

    // Verified pairs inherit deterministic order from candidate_pairs
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
