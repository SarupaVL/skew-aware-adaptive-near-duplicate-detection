#include "candidate_gen.hpp"
#include "dataset_generator.hpp"
#include "parallel_pipeline.hpp"
#include "simhash.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

void check(bool condition, const std::string& test_name, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << test_name << ": " << message << "\n";
        std::exit(1);
    }
    std::cout << "[PASS] " << test_name << "\n";
}

// Helper to assert bitwise equivalence between sequential reference and adaptive pipeline results
void assert_adaptive_matches_sequential(const PipelineResult& seq,
                                        const AdaptivePipelineResult& adap_res,
                                        const std::string& context) {
    const PipelineResult& par = adap_res.pipeline_result;

    // 1. Document IDs
    check(seq.document_ids == par.document_ids, context + "_doc_ids",
          "Document IDs mismatch between sequential and adaptive.");

    // 2. Fingerprints
    check(seq.fingerprints == par.fingerprints, context + "_fingerprints",
          "Fingerprints mismatch between sequential and adaptive.");

    // 3. Candidate Pairs (Normalized, Deduplicated, Sorted)
    check(seq.candidate_pairs.size() == par.candidate_pairs.size(), context + "_candidate_size",
          "Candidate pair count mismatch: seq=" + std::to_string(seq.candidate_pairs.size()) +
          " adap=" + std::to_string(par.candidate_pairs.size()));
    check(seq.candidate_pairs == par.candidate_pairs, context + "_candidate_pairs",
          "Candidate pairs mismatch between sequential and adaptive.");

    // Verify normalization (first < second) and no self-pairs (first != second)
    for (const auto& cp : par.candidate_pairs) {
        if (!(cp.first < cp.second)) {
            check(false, context + "_candidate_norm", "Unnormalized or self candidate pair found.");
        }
    }

    // 4. Verified Pairs (Distance <= Threshold, Sorted)
    check(seq.verified_pairs.size() == par.verified_pairs.size(), context + "_verified_size",
          "Verified pair count mismatch: seq=" + std::to_string(seq.verified_pairs.size()) +
          " adap=" + std::to_string(par.verified_pairs.size()));
    check(seq.verified_pairs == par.verified_pairs, context + "_verified_pairs",
          "Verified pairs mismatch between sequential and adaptive.");

    // 5. Workload Statistics
    const auto& s_ws = seq.workload_stats;
    const auto& p_ws = par.workload_stats;
    check(s_ws.occupied_buckets == p_ws.occupied_buckets, context + "_ws_occ",
          "Occupied buckets count mismatch.");
    check(s_ws.positive_work_buckets == p_ws.positive_work_buckets, context + "_ws_pos_work",
          "Positive work buckets count mismatch.");
    check(s_ws.max_bucket_size == p_ws.max_bucket_size, context + "_ws_max_size",
          "Max bucket size mismatch.");
    check(s_ws.max_estimated_pair_work == p_ws.max_estimated_pair_work, context + "_ws_max_work",
          "Max estimated pair work mismatch.");
    check(std::abs(s_ws.mean_bucket_size - p_ws.mean_bucket_size) < 1e-6, context + "_ws_mean_size",
          "Mean bucket size mismatch.");
    check(std::abs(s_ws.mean_estimated_pair_work - p_ws.mean_estimated_pair_work) < 1e-6, context + "_ws_mean_work",
          "Mean estimated pair work mismatch.");
    check(std::abs(s_ws.cv_estimated_pair_work - p_ws.cv_estimated_pair_work) < 1e-6, context + "_ws_cv",
          "CV of estimated pair work mismatch.");

    // 6. Diagnostics consistency with workload statistics
    const auto& diag = adap_res.diagnostics;
    check(diag.occupied_buckets == p_ws.occupied_buckets, context + "_diag_occ",
          "Diagnostics occupied_buckets mismatch.");
    check(diag.positive_work_buckets == p_ws.positive_work_buckets, context + "_diag_pos",
          "Diagnostics positive_work_buckets mismatch.");
    check(diag.ordinary_buckets + diag.oversized_buckets == diag.positive_work_buckets,
          context + "_diag_bucket_partition",
          "Ordinary + oversized buckets must equal positive_work_buckets.");
    check(std::abs(diag.observed_cv - p_ws.cv_estimated_pair_work) < 1e-9, context + "_diag_cv",
          "Diagnostics observed_cv must match workload_stats.cv_estimated_pair_work.");
}

// 1. Empty input and single-document input
void test_empty_and_single_document() {
    AdaptiveConfig acfg;
    acfg.thread_count = 4;

    // Empty input
    PipelineResult seq_empty = run_sequential_pipeline({});
    AdaptivePipelineResult adap_empty = run_adaptive_pipeline({}, acfg);
    assert_adaptive_matches_sequential(seq_empty, adap_empty, "test_adap_empty");
    check(adap_empty.diagnostics.configured_thread_count == 4, "test_adap_empty_cfg_threads",
          "Configured thread count must be 4.");
    check(adap_empty.diagnostics.actual_thread_count == 0, "test_adap_empty_act_threads",
          "Actual thread count on empty input must be 0.");
    check(adap_empty.diagnostics.oversized_buckets == 0, "test_adap_empty_oversized",
          "Empty input must have 0 oversized buckets.");
    check(adap_empty.diagnostics.heavy_bucket_chunks == 0, "test_adap_empty_chunks",
          "Empty input must have 0 heavy-bucket chunks.");
    check(adap_empty.diagnostics.selected_ordinary_policy == SchedulePolicy::Static,
          "test_adap_empty_policy", "Empty input with CV=0 <= 1.0 must select Static.");

    // Single document
    Document doc{42, 1, DocumentKind::Original, -1, "adaptive parallel multicore cpu simhash"};
    PipelineResult seq_single = run_sequential_pipeline({doc});
    AdaptivePipelineResult adap_single = run_adaptive_pipeline({doc}, acfg);
    assert_adaptive_matches_sequential(seq_single, adap_single, "test_adap_single");
    check(adap_single.diagnostics.occupied_buckets == 8, "test_adap_single_occ",
          "Single document must occupy 8 buckets (1 per band).");
    check(adap_single.diagnostics.positive_work_buckets == 0, "test_adap_single_pos",
          "Single document must have 0 positive-work buckets.");
    check(adap_single.diagnostics.oversized_buckets == 0, "test_adap_single_oversized",
          "Single document must have 0 oversized buckets.");
    check(adap_single.diagnostics.actual_thread_count >= 1, "test_adap_single_act_threads",
          "Actual thread count on non-empty input must be >= 1.");
}

// 2. A workload with no positive-work buckets (multiple documents sharing no bands)
void test_no_positive_work_buckets() {
    // Deterministically find 3 documents whose fingerprints share no band values in any of the 8 bands
    std::vector<Document> docs;
    std::vector<uint64_t> fps;
    uint64_t seed_idx = 1;
    while (docs.size() < 3) {
        std::string text = "unique_token_a_" + std::to_string(seed_idx) +
                           " unique_token_b_" + std::to_string(seed_idx * 17) +
                           " unique_token_c_" + std::to_string(seed_idx * 31);
        uint64_t fp = compute_simhash(text);
        bool collision = false;
        for (uint64_t existing_fp : fps) {
            for (size_t b = 0; b < kNumBands; ++b) {
                if (extract_band(fp, b) == extract_band(existing_fp, b)) {
                    collision = true;
                    break;
                }
            }
            if (collision) break;
        }
        if (!collision) {
            docs.push_back(Document{seed_idx, seed_idx, DocumentKind::Original, -1, text});
            fps.push_back(fp);
        }
        ++seed_idx;
    }

    PipelineResult seq = run_sequential_pipeline(docs);
    AdaptiveConfig acfg;
    acfg.thread_count = 4;
    AdaptivePipelineResult adap = run_adaptive_pipeline(docs, acfg);

    assert_adaptive_matches_sequential(seq, adap, "test_no_pos_work");
    check(adap.diagnostics.occupied_buckets == 24, "test_no_pos_work_occ",
          "3 non-colliding documents must occupy 24 buckets.");
    check(adap.diagnostics.positive_work_buckets == 0, "test_no_pos_work_pos",
          "Must have 0 positive-work buckets.");
    check(adap.diagnostics.oversized_buckets == 0, "test_no_pos_work_oversized",
          "Must have 0 oversized buckets.");
    check(adap.diagnostics.heavy_bucket_chunks == 0, "test_no_pos_work_chunks",
          "Must have 0 heavy-bucket chunks.");
    check(adap.diagnostics.selected_ordinary_policy == SchedulePolicy::Static,
          "test_no_pos_work_policy", "CV=0.0 <= 1.0 must select Static.");
}

// 3 & 5. Workload whose CV is at or below threshold, and workload with no oversized buckets
void test_cv_at_or_below_threshold_and_no_oversized() {
    // Direct helper tests for exact boundary CV <= threshold vs CV > threshold
    std::string reason;
    SchedulePolicy pol_below = select_ordinary_schedule_policy(0.45, 1.0, &reason);
    check(pol_below == SchedulePolicy::Static && !reason.empty(),
          "test_policy_helper_below", "CV = 0.45 <= 1.0 must select Static.");

    SchedulePolicy pol_equal = select_ordinary_schedule_policy(1.0, 1.0, &reason);
    check(pol_equal == SchedulePolicy::Static && !reason.empty(),
          "test_policy_helper_equal", "CV = 1.0 <= 1.0 (exact boundary) must select Static.");

    SchedulePolicy pol_above = select_ordinary_schedule_policy(1.0001, 1.0, &reason);
    check(pol_above == SchedulePolicy::Dynamic && !reason.empty(),
          "test_policy_helper_above", "CV = 1.0001 > 1.0 must select Dynamic.");

    // Controlled dataset where all positive-work buckets have identical size (k = 2, W_i = 1),
    // yielding CV = 0.0 <= 1.0 and 0 oversized buckets (since W_i = 1 <= 4.0 * 1.0).
    Document d1{10, 1, DocumentKind::Original, -1, "uniform pair workload test document one"};
    Document d2{20, 1, DocumentKind::ExactCopy, 10, "uniform pair workload test document one"};
    std::vector<Document> docs = {d1, d2};

    PipelineResult seq = run_sequential_pipeline(docs);
    AdaptiveConfig acfg;
    acfg.thread_count = 4;
    acfg.cv_threshold = 1.0;
    acfg.oversized_bucket_multiplier = 4.0;

    AdaptivePipelineResult adap = run_adaptive_pipeline(docs, acfg);
    assert_adaptive_matches_sequential(seq, adap, "test_cv_zero_uniform");
    check(std::abs(adap.diagnostics.observed_cv - 0.0) < 1e-9,
          "test_cv_zero_val", "Uniform bucket work must have CV == 0.0.");
    check(adap.diagnostics.selected_ordinary_policy == SchedulePolicy::Static,
          "test_cv_zero_static", "CV = 0.0 <= 1.0 must select Static policy.");
    check(adap.diagnostics.positive_work_buckets == 8,
          "test_cv_zero_pos_buckets", "Must have 8 positive-work buckets.");
    check(adap.diagnostics.ordinary_buckets == 8 && adap.diagnostics.oversized_buckets == 0,
          "test_cv_zero_no_oversized", "All 8 buckets must be ordinary (0 oversized).");

    // Also test exact boundary where cv_threshold == observed_cv (0.0 == 0.0)
    acfg.cv_threshold = 0.0;
    AdaptivePipelineResult adap_boundary = run_adaptive_pipeline(docs, acfg);
    check(adap_boundary.diagnostics.selected_ordinary_policy == SchedulePolicy::Static,
          "test_cv_exact_boundary_pipeline", "observed_cv == cv_threshold must select Static.");
}

// 4 & 6. Workload whose CV is above threshold AND contains exactly ONE oversized bucket
void test_cv_above_threshold_and_single_oversized_bucket() {
    // Deterministically construct a dataset where 30 documents share the exact same Band 0 value
    // (producing 1 heavy bucket in Band 0 with k = 30, W = 435), while Bands 1..7 have only small
    // incidental collisions (k = 2 or 3, W = 1 or 3).
    const uint8_t target_band0 = 0x5A;
    std::vector<Document> docs;
    docs.reserve(30);

    uint64_t id_counter = 1;
    uint64_t probe = 1;
    while (docs.size() < 30) {
        std::string text = "band0_target_token_" + std::to_string(probe) +
                           " secondary_word_" + std::to_string(probe * 131 + 7) +
                           " tertiary_word_" + std::to_string(probe * 997 + 3);
        uint64_t fp = compute_simhash(text);
        if (extract_band(fp, 0) == target_band0) {
            docs.push_back(Document{id_counter++, probe, DocumentKind::Original, -1, text});
        }
        ++probe;
    }

    PipelineResult seq = run_sequential_pipeline(docs);

    AdaptiveConfig acfg;
    acfg.thread_count = 4;
    acfg.cv_threshold = 1.0;
    acfg.oversized_bucket_multiplier = 4.0;
    acfg.max_pair_task_size = 150; // 435 pairs -> chunks of 150, 150, 135 (3 chunks)

    AdaptivePipelineResult adap = run_adaptive_pipeline(docs, acfg);
    assert_adaptive_matches_sequential(seq, adap, "test_single_oversized");

    check(adap.diagnostics.observed_cv > 1.0, "test_single_oversized_cv_above_1",
          "Observed CV must exceed 1.0 (got " + std::to_string(adap.diagnostics.observed_cv) + ").");
    check(adap.diagnostics.selected_ordinary_policy == SchedulePolicy::Dynamic,
          "test_single_oversized_dynamic_policy",
          "CV > 1.0 must select Dynamic scheduling for ordinary buckets.");
    check(adap.diagnostics.oversized_buckets == 1, "test_single_oversized_count_is_1",
          "Must identify exactly 1 oversized bucket (got " +
          std::to_string(adap.diagnostics.oversized_buckets) + ").");
    check(adap.diagnostics.oversized_pair_work == 435, "test_single_oversized_work_435",
          "Oversized bucket pair work must be 30*29/2 = 435.");
    check(adap.diagnostics.heavy_bucket_chunks == 3, "test_single_oversized_3_chunks",
          "435 pairs with max_pair_task_size=150 must generate 3 chunks.");
    check(adap.diagnostics.max_generated_chunk_size == 150, "test_single_oversized_max_chunk",
          "Max generated chunk size must be 150.");

    // Also verify case where CV > 1.0 but no buckets exceed a high oversized multiplier (Test 4 + Test 5)
    AdaptiveConfig acfg_no_split = acfg;
    acfg_no_split.oversized_bucket_multiplier = 1000.0;
    AdaptivePipelineResult adap_no_split = run_adaptive_pipeline(docs, acfg_no_split);
    assert_adaptive_matches_sequential(seq, adap_no_split, "test_cv_above_no_oversized");
    check(adap_no_split.diagnostics.observed_cv > 1.0 &&
          adap_no_split.diagnostics.selected_ordinary_policy == SchedulePolicy::Dynamic &&
          adap_no_split.diagnostics.oversized_buckets == 0 &&
          adap_no_split.diagnostics.heavy_bucket_chunks == 0,
          "test_cv_above_no_oversized_diag",
          "CV > 1.0 with high multiplier must select Dynamic and 0 oversized buckets.");
}

// 7. Multiple oversized buckets
void test_multiple_oversized_buckets() {
    // Construct 1 heavy duplicate group of 25 identical documents (produces k >= 25, W >= 300 in all 8 bands)
    // alongside 20 small duplicate groups of 2 identical documents each (producing ~150 buckets with k = 2, W = 1).
    std::vector<Document> docs;
    uint64_t next_id = 1;

    std::string heavy_text = "heavy cluster alpha beta gamma delta epsilon zeta eta theta";
    for (int c = 0; c < 25; ++c) {
        DocumentKind kind = (c == 0) ? DocumentKind::Original : DocumentKind::ExactCopy;
        int64_t src = (c == 0) ? -1 : 1;
        docs.push_back(Document{next_id++, 1, kind, src, heavy_text});
    }

    for (uint64_t g = 2; g <= 21; ++g) {
        std::string small_text = "small_cluster_group_" + std::to_string(g) +
                                 " word_x_" + std::to_string(g * 73) +
                                 " word_y_" + std::to_string(g * 199);
        uint64_t orig_id = next_id++;
        docs.push_back(Document{orig_id, g, DocumentKind::Original, -1, small_text});
        docs.push_back(Document{next_id++, g, DocumentKind::ExactCopy, static_cast<int64_t>(orig_id), small_text});
    }

    PipelineResult seq = run_sequential_pipeline(docs);

    AdaptiveConfig acfg;
    acfg.thread_count = 4;
    acfg.cv_threshold = 1.0;
    acfg.oversized_bucket_multiplier = 4.0;
    acfg.max_pair_task_size = 100; // Split each W >= 300 bucket into >= 3 chunks

    AdaptivePipelineResult adap = run_adaptive_pipeline(docs, acfg);
    assert_adaptive_matches_sequential(seq, adap, "test_multiple_oversized");

    check(adap.diagnostics.observed_cv > 1.0, "test_multi_oversized_cv",
          "Multiple heavy buckets + many small buckets must yield CV > 1.0.");
    check(adap.diagnostics.selected_ordinary_policy == SchedulePolicy::Dynamic,
          "test_multi_oversized_policy", "Must select Dynamic for ordinary buckets.");
    check(adap.diagnostics.oversized_buckets >= 8, "test_multi_oversized_count",
          "Must detect multiple (>= 8) oversized buckets (got " +
          std::to_string(adap.diagnostics.oversized_buckets) + ").");
    check(adap.diagnostics.heavy_bucket_chunks >= 24, "test_multi_oversized_chunks",
          "Each of the >= 8 oversized buckets (W >= 300) with max task size 100 must split into >= 3 chunks.");
    check(adap.diagnostics.max_generated_chunk_size == 100, "test_multi_oversized_max_chunk",
          "Max generated chunk size must equal configured limit 100.");
}

// 8 & 9. Pair-work chunk boundaries (< 4096, == 4096, > 4096, 10000) and no missing/repeated pairs
void test_chunk_boundaries_and_pair_completeness() {
    // A. Direct boundary tests on split_bucket_into_chunks with default 4,096 limit
    {
        // Fewer than 4,096 pair positions (e.g., 4,095 = 91 * 90 / 2)
        auto chunks_less = split_bucket_into_chunks(0, 10, 4095, 4096);
        check(chunks_less.size() == 1 &&
              chunks_less[0].start_pair_pos == 0 &&
              chunks_less[0].end_pair_pos == 4095 &&
              chunks_less[0].pair_count() == 4095,
              "test_chunk_boundary_less_4096",
              "4,095 pair positions must produce 1 chunk of 4,095.");

        // Exactly 4,096 pair positions
        auto chunks_exact = split_bucket_into_chunks(2, 25, 4096, 4096);
        check(chunks_exact.size() == 1 &&
              chunks_exact[0].start_pair_pos == 0 &&
              chunks_exact[0].end_pair_pos == 4096 &&
              chunks_exact[0].pair_count() == 4096,
              "test_chunk_boundary_exact_4096",
              "4,096 pair positions must produce 1 chunk of 4,096.");

        // 4,097 pair positions (1 more than 4,096)
        auto chunks_more = split_bucket_into_chunks(3, 99, 4097, 4096);
        check(chunks_more.size() == 2 &&
              chunks_more[0].pair_count() == 4096 &&
              chunks_more[1].start_pair_pos == 4096 &&
              chunks_more[1].end_pair_pos == 4097 &&
              chunks_more[1].pair_count() == 1,
              "test_chunk_boundary_more_4097",
              "4,097 pair positions must produce chunks of 4,096 and 1.");

        // Specification example: 10,000 pair positions -> 4,096, 4,096, and 1,808
        auto chunks_10k = split_bucket_into_chunks(7, 255, 10000, 4096);
        check(chunks_10k.size() == 3 &&
              chunks_10k[0].pair_count() == 4096 &&
              chunks_10k[1].pair_count() == 4096 &&
              chunks_10k[2].pair_count() == 1808 &&
              chunks_10k[0].start_pair_pos == 0 &&
              chunks_10k[1].start_pair_pos == 4096 &&
              chunks_10k[2].start_pair_pos == 8192 &&
              chunks_10k[2].end_pair_pos == 10000,
              "test_chunk_boundary_10000_spec_example",
              "10,000 pair positions with limit 4,096 must produce chunks of 4096, 4096, and 1808.");
    }

    // B. Verify complete, non-overlapping pair enumeration across chunks for k = 95 (W = 4,465 > 4,096)
    {
        const size_t k = 95;
        const uint64_t total_pairs = (k * (k - 1)) / 2; // 4,465 pairs
        auto chunks = split_bucket_into_chunks(0, 0, total_pairs, 4096);
        check(chunks.size() == 2 && chunks[0].pair_count() == 4096 && chunks[1].pair_count() == 369,
              "test_k95_chunk_sizes", "k=95 (4,465 pairs) must split into 4,096 and 369.");

        std::vector<std::pair<size_t, size_t>> enumerated_pairs;
        enumerated_pairs.reserve(total_pairs);

        for (const auto& chunk : chunks) {
            auto [i, j] = unrank_pair_position(k, chunk.start_pair_pos);
            for (uint64_t pos = chunk.start_pair_pos; pos < chunk.end_pair_pos; ++pos) {
                // Verify incremental (i, j) matches direct unrank_pair_position(k, pos)
                auto direct_ij = unrank_pair_position(k, pos);
                if (direct_ij.first != i || direct_ij.second != j) {
                    check(false, "test_unrank_incremental_match",
                          "Incremental (i, j) diverged from unrank_pair_position at pos=" + std::to_string(pos));
                }
                if (!(i < j && j < k)) {
                    check(false, "test_unrank_bounds", "Invalid pair indices produced.");
                }
                enumerated_pairs.push_back({i, j});

                ++j;
                if (j >= k) {
                    ++i;
                    j = i + 1;
                }
            }
        }

        // Generate expected pairs via reference nested loops
        std::vector<std::pair<size_t, size_t>> expected_pairs;
        expected_pairs.reserve(total_pairs);
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = i + 1; j < k; ++j) {
                expected_pairs.push_back({i, j});
            }
        }

        check(enumerated_pairs == expected_pairs, "test_k95_no_missing_or_repeated_pairs",
              "Chunked pair enumeration must match exhaustive nested loops with 0 missing or repeated pairs.");
    }

    // C. End-to-end adaptive pipeline test with a heavy bucket exceeding 4,096 pairs (k = 95 -> W = 4,465)
    {
        std::vector<Document> docs;
        uint64_t next_id = 1;
        std::string heavy_text = "large_oversized_bucket_4096_boundary_test_document_content";
        for (size_t c = 0; c < 95; ++c) {
            docs.push_back(Document{
                next_id++, 1,
                (c == 0) ? DocumentKind::Original : DocumentKind::ExactCopy,
                (c == 0) ? -1 : 1,
                heavy_text
            });
        }
        // Add 15 small 2-doc groups so mean positive work is well below 4,465 / 4
        for (uint64_t g = 2; g <= 16; ++g) {
            std::string small_text = "small_aux_group_" + std::to_string(g) +
                                     " token_" + std::to_string(g * 401);
            uint64_t oid = next_id++;
            docs.push_back(Document{oid, g, DocumentKind::Original, -1, small_text});
            docs.push_back(Document{next_id++, g, DocumentKind::ExactCopy, static_cast<int64_t>(oid), small_text});
        }

        PipelineResult seq = run_sequential_pipeline(docs);
        AdaptiveConfig acfg;
        acfg.thread_count = 4;
        acfg.max_pair_task_size = 4096; // Default 4,096
        AdaptivePipelineResult adap = run_adaptive_pipeline(docs, acfg);

        assert_adaptive_matches_sequential(seq, adap, "test_pipeline_gt_4096_pairs");
        check(adap.diagnostics.max_generated_chunk_size == 4096,
              "test_pipeline_gt_4096_max_chunk",
              "Heavy bucket with >= 4,465 pairs must cap max chunk size at 4,096.");
        check(adap.diagnostics.heavy_bucket_chunks >= 16,
              "test_pipeline_gt_4096_chunk_count",
              "8 bands x >= 2 chunks per band must produce >= 16 heavy chunks.");
    }
}

// 10. Correct exclusion of self-pairs and pair normalization
void test_self_pair_exclusion_and_normalization() {
    DatasetConfig dcfg;
    dcfg.total_documents = 90;
    dcfg.exact_duplicate_ratio = 0.35;
    dcfg.modified_copy_ratio = 0.35;
    dcfg.profile = WorkloadProfile::Skewed;
    dcfg.random_seed = 808;
    auto docs = generate_dataset(dcfg);

    AdaptiveConfig acfg;
    acfg.thread_count = 4;
    acfg.max_pair_task_size = 32; // Force fine-grained heavy-bucket splitting
    AdaptivePipelineResult adap = run_adaptive_pipeline(docs, acfg);

    for (const auto& cp : adap.pipeline_result.candidate_pairs) {
        if (cp.first >= cp.second) {
            check(false, "test_self_pair_candidate", "Self-pair or unnormalized candidate pair detected.");
        }
    }
    for (const auto& vp : adap.pipeline_result.verified_pairs) {
        if (vp.first >= vp.second) {
            check(false, "test_self_pair_verified", "Self-pair or unnormalized verified pair detected.");
        }
    }
    check(true, "test_self_pair_exclusion_and_normalization",
          "All candidate and verified pairs exclude self-pairs and satisfy first < second.");
}

// 11, 12, 15. Exact result parity across multiple thread counts, repeated runs, and fixed baselines
void test_thread_counts_repeated_runs_and_baselines() {
    DatasetConfig dcfg;
    dcfg.total_documents = 120;
    dcfg.exact_duplicate_ratio = 0.30;
    dcfg.modified_copy_ratio = 0.30;
    dcfg.unrelated_ratio = 0.10;
    dcfg.profile = WorkloadProfile::Skewed;
    dcfg.random_seed = 909;
    auto docs = generate_dataset(dcfg);

    PipelineResult seq_ref = run_sequential_pipeline(docs);

    // Verify fixed baselines (Static, Dynamic, Guided) still match sequential reference
    for (auto policy : {SchedulePolicy::Static, SchedulePolicy::Dynamic, SchedulePolicy::Guided}) {
        ParallelConfig pcfg;
        pcfg.thread_count = 4;
        pcfg.schedule_policy = policy;
        pcfg.chunk_size = (policy == SchedulePolicy::Static) ? 0 : 2;
        PipelineResult fixed_res = run_parallel_pipeline(docs, pcfg);
        check(seq_ref.candidate_pairs == fixed_res.candidate_pairs &&
              seq_ref.verified_pairs == fixed_res.verified_pairs,
              "test_fixed_baseline_no_regression_" + std::string(to_string(policy)),
              "Fixed schedule baseline must match sequential reference.");
    }

    // Verify adaptive pipeline across thread counts {1, 2, 4, 8}
    for (int threads : {1, 2, 4, 8}) {
        AdaptiveConfig acfg;
        acfg.thread_count = threads;
        acfg.max_pair_task_size = 25; // Ensure heavy buckets split into multiple chunks
        AdaptivePipelineResult adap = run_adaptive_pipeline(docs, acfg);
        assert_adaptive_matches_sequential(
            seq_ref, adap, "test_adap_threads_" + std::to_string(threads)
        );
    }

    // Verify repeated adaptive executions produce bitwise-identical results
    AdaptiveConfig acfg_rep;
    acfg_rep.thread_count = 4;
    acfg_rep.max_pair_task_size = 20;
    for (int trial = 1; trial <= 5; ++trial) {
        AdaptivePipelineResult run_res = run_adaptive_pipeline(docs, acfg_rep);
        assert_adaptive_matches_sequential(
            seq_ref, run_res, "test_adap_repeat_trial_" + std::to_string(trial)
        );
    }
}

// 13. Non-contiguous document IDs and reordered inputs
void test_non_contiguous_and_reordered_ids() {
    std::vector<Document> docs_orig;
    const std::string shared_text = "non_contiguous_id_heavy_bucket_verification_text_tokens";
    const std::vector<uint64_t> custom_ids = {
        900001, 12, 777777, 500, 42, 88888, 10000007, 3, 65537, 250000,
        19, 8191, 131071, 524287, 7, 99999999, 1048573, 3333, 44444, 555555
    };
    for (size_t idx = 0; idx < custom_ids.size(); ++idx) {
        docs_orig.push_back(Document{
            custom_ids[idx], 1,
            (idx == 0) ? DocumentKind::Original : DocumentKind::ExactCopy,
            (idx == 0) ? -1 : static_cast<int64_t>(custom_ids[0]),
            shared_text
        });
    }
    // Add unrelated documents with large non-contiguous IDs
    docs_orig.push_back(Document{400000001ULL, 2, DocumentKind::Unrelated, -1, "unrelated alpha bravo charlie"});
    docs_orig.push_back(Document{200000002ULL, 3, DocumentKind::Unrelated, -1, "unrelated delta echo foxtrot"});

    // Create a reversed/permuted ordering of the same documents
    std::vector<Document> docs_reordered = docs_orig;
    std::reverse(docs_reordered.begin(), docs_reordered.end());

    PipelineResult seq_orig = run_sequential_pipeline(docs_orig);
    PipelineResult seq_reord = run_sequential_pipeline(docs_reordered);

    AdaptiveConfig acfg;
    acfg.thread_count = 4;
    acfg.oversized_bucket_multiplier = 0.5; // Force heavy bucket splitting on the 20-doc cluster
    acfg.max_pair_task_size = 40;

    AdaptivePipelineResult adap_orig = run_adaptive_pipeline(docs_orig, acfg);
    AdaptivePipelineResult adap_reord = run_adaptive_pipeline(docs_reordered, acfg);

    assert_adaptive_matches_sequential(seq_orig, adap_orig, "test_non_contig_orig");
    assert_adaptive_matches_sequential(seq_reord, adap_reord, "test_non_contig_reordered");

    check(adap_orig.pipeline_result.candidate_pairs == adap_reord.pipeline_result.candidate_pairs,
          "test_reordered_candidates_identical",
          "Reordered input must yield identical normalized candidate pairs.");
    check(adap_orig.pipeline_result.verified_pairs == adap_reord.pipeline_result.verified_pairs,
          "test_reordered_verified_identical",
          "Reordered input must yield identical verified pairs.");
}

// 14. Invalid adaptive configuration values & schedule restoration
void test_invalid_configs_and_schedule_restoration() {
    // 1. thread_count <= 0
    {
        AdaptiveConfig cfg;
        cfg.thread_count = 0;
        bool caught = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "test_invalid_adap_threads_0", "thread_count = 0 must throw.");
    }
    // 2. Negative, NaN, and Infinite cv_threshold
    {
        AdaptiveConfig cfg;
        cfg.cv_threshold = -0.1;
        bool caught = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "test_invalid_adap_cv_neg", "Negative cv_threshold must throw.");

        cfg.cv_threshold = std::numeric_limits<double>::quiet_NaN();
        caught = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "test_invalid_adap_cv_nan", "NaN cv_threshold must throw.");

        cfg.cv_threshold = std::numeric_limits<double>::infinity();
        caught = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "test_invalid_adap_cv_inf", "Infinite cv_threshold must throw.");
    }
    // 3. Zero, negative, NaN, and Infinite oversized_bucket_multiplier
    {
        AdaptiveConfig cfg;
        cfg.oversized_bucket_multiplier = 0.0;
        bool caught = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "test_invalid_adap_mult_zero", "Zero oversized_bucket_multiplier must throw.");

        cfg.oversized_bucket_multiplier = -2.0;
        caught = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "test_invalid_adap_mult_neg", "Negative oversized_bucket_multiplier must throw.");

        cfg.oversized_bucket_multiplier = std::numeric_limits<double>::quiet_NaN();
        caught = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "test_invalid_adap_mult_nan", "NaN oversized_bucket_multiplier must throw.");

        cfg.oversized_bucket_multiplier = std::numeric_limits<double>::infinity();
        caught = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "test_invalid_adap_mult_inf", "Infinite oversized_bucket_multiplier must throw.");
    }
    // 4. max_pair_task_size == 0
    {
        AdaptiveConfig cfg;
        cfg.max_pair_task_size = 0;
        bool caught = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "test_invalid_adap_task_size_0", "max_pair_task_size = 0 must throw.");
    }
    // 5. Invalid unrank_pair_position calls
    {
        bool caught_small = false;
        try { unrank_pair_position(1, 0); } catch (const std::invalid_argument&) { caught_small = true; }
        check(caught_small, "test_invalid_unrank_k1", "unrank_pair_position with k < 2 must throw.");

        bool caught_oob = false;
        try { unrank_pair_position(4, 6); } catch (const std::out_of_range&) { caught_oob = true; }
        check(caught_oob, "test_invalid_unrank_oob", "unrank_pair_position with pos >= W must throw.");
    }
    // 6. Duplicate document IDs and OpenMP schedule restoration
    {
        omp_set_schedule(omp_sched_guided, 9);
        Document d1{77, 1, DocumentKind::Original, -1, "dup id test"};
        Document d2{77, 1, DocumentKind::ExactCopy, 77, "dup id test"};
        bool caught = false;
        try {
            run_adaptive_pipeline({d1, d2});
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        check(caught, "test_adap_duplicate_id_rejected", "Duplicate document IDs must throw.");

        omp_sched_t sched;
        int chunk = 0;
        omp_get_schedule(&sched, &chunk);
        check(sched == omp_sched_guided && chunk == 9,
              "test_adap_schedule_restored_after_exception",
              "Caller OpenMP schedule must remain intact after exception.");
    }
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << " Running Phase 5 Test Suite: Adaptive Scheduling & Heavy Buckets\n";
    std::cout << "=================================================================\n";

    test_empty_and_single_document();
    test_no_positive_work_buckets();
    test_cv_at_or_below_threshold_and_no_oversized();
    test_cv_above_threshold_and_single_oversized_bucket();
    test_multiple_oversized_buckets();
    test_chunk_boundaries_and_pair_completeness();
    test_self_pair_exclusion_and_normalization();
    test_thread_counts_repeated_runs_and_baselines();
    test_non_contiguous_and_reordered_ids();
    test_invalid_configs_and_schedule_restoration();

    std::cout << "\n=================================================================\n";
    std::cout << " All Phase 5 Adaptive Pipeline Tests Passed Successfully!\n";
    std::cout << "=================================================================\n";
    return 0;
}
