#include "candidate_gen.hpp"
#include "dataset_generator.hpp"
#include <cassert>
#include <iostream>
#include <set>
#include <string>
#include <vector>

void check(bool condition, const std::string& test_name, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << test_name << ": " << message << "\n";
        std::exit(1);
    }
    std::cout << "[PASS] " << test_name << "\n";
}

void test_empty_and_single_document() {
    // 1. Empty input
    PipelineResult res_empty = run_sequential_pipeline({});
    check(res_empty.document_ids.empty(), "test_empty_doc_ids", "Doc IDs must be empty.");
    check(res_empty.fingerprints.empty(), "test_empty_fingerprints", "Fingerprints must be empty.");
    check(res_empty.candidate_pairs.empty(), "test_empty_candidates", "Candidates must be empty.");
    check(res_empty.verified_pairs.empty(), "test_empty_verified", "Verified pairs must be empty.");
    check(res_empty.workload_stats.occupied_buckets == 0, "test_empty_occupied_buckets", "Occupied buckets must be 0.");
    check(res_empty.workload_stats.cv_estimated_pair_work == 0.0, "test_empty_cv", "CV must be 0.");

    // 2. Single document
    Document doc;
    doc.id = 42;
    doc.group_id = 1;
    doc.kind = DocumentKind::Original;
    doc.text = "multicore parallel scheduling";

    PipelineResult res_single = run_sequential_pipeline({doc});
    check(res_single.document_ids.size() == 1, "test_single_doc_ids", "Doc IDs size must be 1.");
    check(res_single.fingerprints.size() == 1, "test_single_fingerprints", "Fingerprints size must be 1.");
    check(res_single.candidate_pairs.empty(), "test_single_candidates", "No candidate pairs for single doc.");
    check(res_single.verified_pairs.empty(), "test_single_verified", "No verified pairs for single doc.");
    check(res_single.workload_stats.occupied_buckets == 8, "test_single_occupied_buckets", "1 doc maps to 8 buckets (1 per band).");
    check(res_single.workload_stats.positive_work_buckets == 0, "test_single_pos_work", "Positive work buckets must be 0.");
    check(res_single.workload_stats.cv_estimated_pair_work == 0.0, "test_single_cv", "CV must be 0.");
}

void test_band_extraction_boundaries() {
    // 64-bit fingerprint: 0x0123456789ABCDEF
    // Band 0 (bits 0..7)   : 0xEF
    // Band 1 (bits 8..15)  : 0xCD
    // Band 2 (bits 16..23) : 0xAB
    // Band 3 (bits 24..31) : 0x89
    // Band 4 (bits 32..39) : 0x67
    // Band 5 (bits 40..47) : 0x45
    // Band 6 (bits 48..55) : 0x23
    // Band 7 (bits 56..63) : 0x01
    uint64_t fp = 0x0123456789ABCDEFULL;

    check(extract_band(fp, 0) == 0xEF, "test_band_0", "Band 0 mismatch.");
    check(extract_band(fp, 1) == 0xCD, "test_band_1", "Band 1 mismatch.");
    check(extract_band(fp, 2) == 0xAB, "test_band_2", "Band 2 mismatch.");
    check(extract_band(fp, 3) == 0x89, "test_band_3", "Band 3 mismatch.");
    check(extract_band(fp, 4) == 0x67, "test_band_4", "Band 4 mismatch.");
    check(extract_band(fp, 5) == 0x45, "test_band_5", "Band 5 mismatch.");
    check(extract_band(fp, 6) == 0x23, "test_band_6", "Band 6 mismatch.");
    check(extract_band(fp, 7) == 0x01, "test_band_7", "Band 7 mismatch.");
}

void test_pair_normalization_and_deduplication() {
    // Two documents with exact same text share all 8 bands.
    // They must produce exactly ONE candidate pair, with smaller ID first.
    Document docA;
    docA.id = 100;
    docA.text = "identical document text shingle algorithm voting";

    Document docB;
    docB.id = 50; // Note: smaller ID than docA
    docB.text = docA.text;

    PipelineResult res = run_sequential_pipeline({docA, docB});
    check(res.candidate_pairs.size() == 1, "test_pair_dedup_count", "Must deduplicate to 1 candidate pair.");
    check(res.candidate_pairs[0].first == 50 && res.candidate_pairs[0].second == 100,
          "test_pair_normalization", "Normalized pair must have first < second (50, 100).");
    check(res.verified_pairs.size() == 1, "test_verified_dedup_count", "Must have 1 verified pair.");
    check(res.verified_pairs[0].distance == 0, "test_verified_exact_dist", "Exact copy distance must be 0.");
}

void test_hamming_threshold_boundary() {
    // Test that candidate verification strictly respects threshold:
    // Distance <= threshold is accepted, distance > threshold is rejected.
    Document docA;
    docA.id = 1;
    docA.text = "benchmark latency throughput multicore parallel memory";

    Document docB;
    docB.id = 2;
    docB.text = docA.text; // distance 0

    // Compute actual distance
    uint64_t fpA = compute_simhash(docA.text);
    uint64_t fpB = compute_simhash(docB.text);
    int dist = hamming_distance(fpA, fpB);
    assert(dist == 0);

    // With threshold = 0, should be accepted
    PipelineConfig cfg0;
    cfg0.max_hamming_threshold = 0;
    PipelineResult res0 = run_sequential_pipeline({docA, docB}, cfg0);
    check(res0.verified_pairs.size() == 1, "test_threshold_boundary_accept", "Distance 0 <= threshold 0 must be accepted.");

    // Now construct two docs where fingerprints differ by known amount
    // Let's test with a synthetic case by running with dataset generator
    DatasetConfig dcfg;
    dcfg.total_documents = 20;
    dcfg.exact_duplicate_ratio = 0.20;
    dcfg.modified_copy_ratio = 0.20;
    dcfg.random_seed = 42;
    auto docs = generate_dataset(dcfg);

    PipelineConfig cfg3;
    cfg3.max_hamming_threshold = 3;
    PipelineResult res3 = run_sequential_pipeline(docs, cfg3);

    for (const auto& vp : res3.verified_pairs) {
        check(vp.distance <= 3, "test_threshold_enforced", "All verified pairs must have distance <= 3.");
    }
}

void test_deterministic_ordering() {
    DatasetConfig dcfg;
    dcfg.total_documents = 30;
    dcfg.exact_duplicate_ratio = 0.30;
    dcfg.modified_copy_ratio = 0.20;
    dcfg.random_seed = 999;
    auto docs = generate_dataset(dcfg);

    PipelineResult res1 = run_sequential_pipeline(docs);
    PipelineResult res2 = run_sequential_pipeline(docs);

    check(res1.candidate_pairs.size() == res2.candidate_pairs.size(), "test_order_candidate_size", "Sizes must match.");
    check(res1.verified_pairs.size() == res2.verified_pairs.size(), "test_order_verified_size", "Verified sizes must match.");

    for (size_t i = 0; i < res1.candidate_pairs.size(); ++i) {
        check(res1.candidate_pairs[i] == res2.candidate_pairs[i], "test_order_candidate_item", "Candidate ordering must match.");
    }
    for (size_t i = 0; i < res1.verified_pairs.size(); ++i) {
        check(res1.verified_pairs[i] == res2.verified_pairs[i], "test_order_verified_item", "Verified ordering must match.");
    }
}

void test_workload_statistics_and_cv() {
    // 3 documents: doc1 and doc2 have identical text (bucket size 2 across all 8 bands)
    // doc3 is unrelated and maps to different buckets
    Document d1{1, 1, DocumentKind::Original, -1, "apple banana cherry date elderberry"};
    Document d2{2, 1, DocumentKind::ExactCopy, 1, "apple banana cherry date elderberry"};
    Document d3{3, 2, DocumentKind::Unrelated, -1, "network socket packet protocol interconnect"};

    PipelineResult res = run_sequential_pipeline({d1, d2, d3});
    const auto& ws = res.workload_stats;

    check(ws.occupied_buckets > 0, "test_ws_occupied", "Occupied buckets > 0.");
    check(ws.max_bucket_size >= 2, "test_ws_max_size", "Max bucket size >= 2.");
    check(ws.max_estimated_pair_work >= 1, "test_ws_max_work", "Max estimated pair work >= 1.");
    check(ws.mean_bucket_size >= 1.0, "test_ws_mean_size", "Mean bucket size >= 1.0.");
    check(ws.positive_work_buckets > 0, "test_ws_pos_work_count", "Positive work bucket count > 0.");
    check(ws.cv_estimated_pair_work >= 0.0, "test_ws_cv_non_negative", "CV must be non-negative.");

    // Zero-workload test: 2 documents that map to completely disjoint buckets across all bands
    // (can be simulated with an empty or 1-document dataset)
    PipelineResult res_singleton = run_sequential_pipeline({d1});
    check(res_singleton.workload_stats.positive_work_buckets == 0, "test_zero_work_buckets", "0 positive work buckets.");
    check(res_singleton.workload_stats.cv_estimated_pair_work == 0.0, "test_zero_work_cv", "CV must be 0.0 when no work.");
}

void test_empty_and_whitespace_documents() {
    Document d_empty{1, 1, DocumentKind::Original, -1, ""};
    Document d_spaces{2, 2, DocumentKind::Original, -1, "   \t\n  "};

    PipelineResult res = run_sequential_pipeline({d_empty, d_spaces});
    // Both produce fingerprint 0ULL, so they share all bands (bucket 0)
    check(res.fingerprints[0] == 0ULL && res.fingerprints[1] == 0ULL,
          "test_empty_fingerprints_zero", "Empty documents produce 0ULL.");
    check(res.candidate_pairs.size() == 1, "test_empty_candidate_pair", "Must match in bucket 0.");
    check(res.verified_pairs.size() == 1 && res.verified_pairs[0].distance == 0,
          "test_empty_verified_pair", "Hamming distance between empty docs is 0.");
}

/**
 * @brief ESSENTIAL CORRECTNESS GUARANTEE TEST
 *
 * Mathematical Guarantee:
 * With 8 bands of 8 bits each across a 64-bit fingerprint:
 * If Hamming distance <= 7, by the Pigeonhole Principle, at least one of the
 * 8 bands must contain 0 differing bits (i.e. identical 8-bit band value).
 * Therefore, ANY pair of documents with Hamming distance <= 7 MUST be generated
 * as a candidate pair.
 *
 * Furthermore, for any threshold <= 7 (including the default threshold 3),
 * candidate-based verification must produce the EXACT same near-duplicate pairs
 * as exhaustive all-pairs O(N^2) verification (100% recall).
 */
void test_essential_correctness_guarantee() {
    DatasetConfig dcfg;
    dcfg.total_documents = 60;
    dcfg.min_words = 30;
    dcfg.max_words = 60;
    dcfg.exact_duplicate_ratio = 0.25;
    dcfg.modified_copy_ratio = 0.25;
    dcfg.unrelated_ratio = 0.10;
    dcfg.modification_rate = 0.10; // 10% modification ensures many near-duplicates
    dcfg.random_seed = 12345;
    dcfg.profile = WorkloadProfile::Balanced;

    auto docs = generate_dataset(dcfg);

    PipelineConfig pcfg;
    pcfg.max_hamming_threshold = 3;
    PipelineResult pipeline_res = run_sequential_pipeline(docs, pcfg);

    // 1. Exhaustive all-pairs baseline (ground truth for fingerprints)
    std::set<CandidatePair> expected_candidates_le7;
    std::set<VerifiedPair> expected_verified_thresh3;

    size_t n = docs.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            uint64_t id1 = pipeline_res.document_ids[i];
            uint64_t id2 = pipeline_res.document_ids[j];
            CandidatePair pair = (id1 < id2) ? CandidatePair{id1, id2} : CandidatePair{id2, id1};

            int dist = hamming_distance(pipeline_res.fingerprints[i], pipeline_res.fingerprints[j]);

            // Pigeonhole guarantee: If distance <= 7, pair MUST be in candidate set
            if (dist <= 7) {
                expected_candidates_le7.insert(pair);
            }

            // Target threshold = 3: pair MUST be in verified set
            if (dist <= pcfg.max_hamming_threshold) {
                expected_verified_thresh3.insert(VerifiedPair{pair.first, pair.second, dist});
            }
        }
    }

    // Convert pipeline candidates to set for fast membership checking
    std::set<CandidatePair> actual_candidates(
        pipeline_res.candidate_pairs.begin(),
        pipeline_res.candidate_pairs.end()
    );

    // Verification 1: Every pair with Hamming distance <= 7 was caught by candidate generation
    size_t guaranteed_le7_count = 0;
    for (const auto& pair : expected_candidates_le7) {
        check(actual_candidates.find(pair) != actual_candidates.end(),
              "test_pigeonhole_candidate_recall",
              "Pigeonhole guarantee violated: Pair with dist <= 7 was missing from candidate set!");
        guaranteed_le7_count++;
    }
    check(guaranteed_le7_count > 0, "test_guaranteed_le7_exists", "Must have verified le7 candidate recall.");

    // Verification 2: Final verified pairs exactly match exhaustive verification at threshold 3
    std::set<VerifiedPair> actual_verified(
        pipeline_res.verified_pairs.begin(),
        pipeline_res.verified_pairs.end()
    );

    check(actual_verified == expected_verified_thresh3,
          "test_exhaustive_parity_thresh3",
          "Pipeline verified pairs must exactly match exhaustive all-pairs verification at threshold 3!");
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << " Running Phase 3 Test Suite: Sequential Pipeline\n";
    std::cout << "=========================================================\n";

    test_empty_and_single_document();
    test_band_extraction_boundaries();
    test_pair_normalization_and_deduplication();
    test_hamming_threshold_boundary();
    test_deterministic_ordering();
    test_workload_statistics_and_cv();
    test_empty_and_whitespace_documents();
    test_essential_correctness_guarantee();

    std::cout << "\n=========================================================\n";
    std::cout << " All Phase 3 Pipeline Tests Passed Successfully!\n";
    std::cout << "=========================================================\n";
    return 0;
}
