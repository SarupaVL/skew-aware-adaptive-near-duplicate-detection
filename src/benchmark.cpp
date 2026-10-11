#include "benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

double compute_median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("compute_median: values cannot be empty.");
    }
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double compute_mean(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("compute_mean: values cannot be empty.");
    }
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return sum / static_cast<double>(values.size());
}

double compute_sample_sd(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("compute_sample_sd: values cannot be empty.");
    }
    if (values.size() < 2) {
        return 0.0;
    }
    double mean = compute_mean(values);
    double sq_diff_sum = 0.0;
    for (double v : values) {
        double diff = v - mean;
        sq_diff_sum += diff * diff;
    }
    return std::sqrt(sq_diff_sum / static_cast<double>(values.size() - 1));
}

double compute_iqr(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("compute_iqr: values cannot be empty.");
    }
    if (values.size() < 2) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    size_t half = n / 2;

    std::vector<double> lower_half(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(half));
    std::vector<double> upper_half(
        values.begin() + static_cast<std::ptrdiff_t>((n % 2 == 0) ? half : (half + 1)),
        values.end()
    );

    double q1 = compute_median(std::move(lower_half));
    double q3 = compute_median(std::move(upper_half));
    return q3 - q1;
}

double compute_speedup(double sequential_ms, double parallel_ms) {
    if (!std::isfinite(sequential_ms) || sequential_ms < 0.0) {
        throw std::invalid_argument("compute_speedup: sequential_ms must be finite and non-negative.");
    }
    if (!std::isfinite(parallel_ms) || parallel_ms <= 0.0) {
        throw std::invalid_argument("compute_speedup: parallel_ms must be finite and strictly positive.");
    }
    return sequential_ms / parallel_ms;
}

double compute_efficiency(double speedup, int thread_count) {
    if (!std::isfinite(speedup) || speedup < 0.0) {
        throw std::invalid_argument("compute_efficiency: speedup must be finite and non-negative.");
    }
    if (thread_count <= 0) {
        throw std::invalid_argument("compute_efficiency: thread_count must be positive.");
    }
    return speedup / static_cast<double>(thread_count);
}

WorkBalanceMetrics compute_work_balance_metrics(
    const std::vector<uint64_t>& per_thread_pair_work,
    const std::vector<size_t>& per_thread_units_processed,
    int configured_threads,
    int actual_threads
) {
    if (configured_threads <= 0) {
        throw std::invalid_argument(
            "compute_work_balance_metrics: configured_threads must be positive."
        );
    }

    WorkBalanceMetrics metrics;
    metrics.configured_threads = configured_threads;
    metrics.actual_threads = (actual_threads > 0) ? actual_threads : configured_threads;

    size_t num_slots = static_cast<size_t>(std::max(configured_threads, metrics.actual_threads));
    if (per_thread_pair_work.size() > num_slots) {
        num_slots = per_thread_pair_work.size();
    }

    std::vector<uint64_t> padded_work(num_slots, 0ULL);
    for (size_t i = 0; i < per_thread_pair_work.size(); ++i) {
        padded_work[i] = per_thread_pair_work[i];
    }

    uint64_t total_work = 0;
    uint64_t max_work = 0;
    uint64_t min_work = padded_work.empty() ? 0ULL : padded_work[0];
    int participating = 0;
    std::ostringstream dist_oss;

    for (size_t t = 0; t < num_slots; ++t) {
        uint64_t w = padded_work[t];
        total_work += w;
        if (w > max_work) {
            max_work = w;
        }
        if (w < min_work) {
            min_work = w;
        }
        if (w > 0) {
            ++participating;
        }
        if (t > 0) {
            dist_oss << ';';
        }
        dist_oss << w;
    }

    size_t total_units = 0;
    for (size_t u : per_thread_units_processed) {
        total_units += u;
    }

    metrics.participating_threads = participating;
    metrics.total_pair_work = total_work;
    metrics.max_thread_pair_work = max_work;
    metrics.min_thread_pair_work = min_work;
    metrics.mean_thread_pair_work = (num_slots > 0)
        ? (static_cast<double>(total_work) / static_cast<double>(num_slots))
        : 0.0;

    if (num_slots <= 1 || total_work == 0ULL || metrics.mean_thread_pair_work <= 0.0) {
        metrics.work_imbalance_ratio = 1.0;
    } else {
        metrics.work_imbalance_ratio =
            static_cast<double>(max_work) / metrics.mean_thread_pair_work;
    }

    metrics.total_units_processed = total_units;
    metrics.per_thread_work_distribution = dist_oss.str();
    return metrics;
}

ParityCheckResult verify_pipeline_parity(
    const PipelineResult& reference_seq,
    const PipelineResult& candidate_res
) {
    ParityCheckResult res;
    res.passed = true;

    // 1. Document IDs
    if (reference_seq.document_ids != candidate_res.document_ids) {
        res.passed = false;
        res.failure_reason = "Document IDs vector mismatch against sequential reference.";
        return res;
    }

    // 2. Fingerprints
    if (reference_seq.fingerprints != candidate_res.fingerprints) {
        res.passed = false;
        res.failure_reason = "SimHash fingerprints mismatch against sequential reference.";
        return res;
    }

    // 3. Candidate pairs (count, exact values, normalization, and sorting)
    if (reference_seq.candidate_pairs.size() != candidate_res.candidate_pairs.size()) {
        res.passed = false;
        res.failure_reason = "Candidate pair count mismatch (seq=" +
            std::to_string(reference_seq.candidate_pairs.size()) + ", candidate=" +
            std::to_string(candidate_res.candidate_pairs.size()) + ").";
        return res;
    }
    if (reference_seq.candidate_pairs != candidate_res.candidate_pairs) {
        res.passed = false;
        res.failure_reason = "Candidate pairs content/order mismatch against sequential reference.";
        return res;
    }
    for (size_t i = 0; i < candidate_res.candidate_pairs.size(); ++i) {
        const auto& cp = candidate_res.candidate_pairs[i];
        if (!(cp.first < cp.second)) {
            res.passed = false;
            res.failure_reason = "Unnormalized or self candidate pair detected at index " +
                std::to_string(i) + ".";
            return res;
        }
        if (i > 0 && !(candidate_res.candidate_pairs[i - 1] < cp)) {
            res.passed = false;
            res.failure_reason = "Unsorted or duplicate candidate pair detected at index " +
                std::to_string(i) + ".";
            return res;
        }
    }

    // 4. Verified pairs
    if (reference_seq.verified_pairs.size() != candidate_res.verified_pairs.size()) {
        res.passed = false;
        res.failure_reason = "Verified pair count mismatch (seq=" +
            std::to_string(reference_seq.verified_pairs.size()) + ", candidate=" +
            std::to_string(candidate_res.verified_pairs.size()) + ").";
        return res;
    }
    if (reference_seq.verified_pairs != candidate_res.verified_pairs) {
        res.passed = false;
        res.failure_reason = "Verified pairs content/order mismatch against sequential reference.";
        return res;
    }

    // 5. Workload statistics
    const auto& s_ws = reference_seq.workload_stats;
    const auto& c_ws = candidate_res.workload_stats;
    if (s_ws.occupied_buckets != c_ws.occupied_buckets ||
        s_ws.positive_work_buckets != c_ws.positive_work_buckets ||
        s_ws.max_bucket_size != c_ws.max_bucket_size ||
        s_ws.max_estimated_pair_work != c_ws.max_estimated_pair_work ||
        std::abs(s_ws.mean_bucket_size - c_ws.mean_bucket_size) > 1e-6 ||
        std::abs(s_ws.mean_estimated_pair_work - c_ws.mean_estimated_pair_work) > 1e-6 ||
        std::abs(s_ws.cv_estimated_pair_work - c_ws.cv_estimated_pair_work) > 1e-6) {
        res.passed = false;
        res.failure_reason = "WorkloadStatistics mismatch against sequential reference.";
        return res;
    }

    // 6. Total per-thread estimated pair work conservation
    uint64_t expected_total_pair_work = 0;
    for (const auto& b : reference_seq.buckets) {
        expected_total_pair_work += b.estimated_pair_work;
    }
    uint64_t observed_thread_pair_work = 0;
    for (uint64_t w : candidate_res.per_thread_pair_work) {
        observed_thread_pair_work += w;
    }
    if (expected_total_pair_work != observed_thread_pair_work) {
        res.passed = false;
        res.failure_reason = "Per-thread pair work sum (" +
            std::to_string(observed_thread_pair_work) +
            ") does not equal total bucket pair work (" +
            std::to_string(expected_total_pair_work) + ").";
        return res;
    }

    // 7. Evaluated, emitted, and suppressed pair counter consistency (Phase 10)
    if (candidate_res.raw_candidates_evaluated != expected_total_pair_work) {
        res.passed = false;
        res.failure_reason = "raw_candidates_evaluated (" +
            std::to_string(candidate_res.raw_candidates_evaluated) +
            ") does not equal total bucket pair work (" +
            std::to_string(expected_total_pair_work) + ").";
        return res;
    }
    if (candidate_res.raw_candidates_emitted + candidate_res.suppressed_pair_count !=
        candidate_res.raw_candidates_evaluated) {
        res.passed = false;
        res.failure_reason = "raw_candidates_emitted + suppressed_pair_count does not equal raw_candidates_evaluated.";
        return res;
    }
    if (!candidate_res.band_suppression_enabled && candidate_res.suppressed_pair_count != 0) {
        res.passed = false;
        res.failure_reason = "suppressed_pair_count must be 0 when band_suppression_enabled is false.";
        return res;
    }
    if (candidate_res.band_suppression_enabled &&
        candidate_res.raw_candidates_emitted != candidate_res.candidate_pairs.size()) {
        res.passed = false;
        res.failure_reason = "When band_suppression_enabled is true, raw_candidates_emitted (" +
            std::to_string(candidate_res.raw_candidates_emitted) +
            ") must equal unique candidate_pairs.size() (" +
            std::to_string(candidate_res.candidate_pairs.size()) + ").";
        return res;
    }

    return res;
}

BenchmarkSummaryRow summarize_benchmark_samples(
    const std::vector<BenchmarkSampleRow>& samples,
    size_t warmup_runs,
    double seq_median_total_ms,
    double seq_median_cand_ms
) {
    if (samples.empty()) {
        throw std::invalid_argument("summarize_benchmark_samples: samples cannot be empty.");
    }

    const BenchmarkSampleRow& first = samples.front();
    BenchmarkSummaryRow summary;

    summary.dataset_size = first.dataset_size;
    summary.profile = first.profile;
    summary.random_seed = first.random_seed;
    summary.min_words = first.min_words;
    summary.max_words = first.max_words;
    summary.exact_dup_ratio = first.exact_dup_ratio;
    summary.modified_copy_ratio = first.modified_copy_ratio;
    summary.unrelated_ratio = first.unrelated_ratio;
    summary.modification_rate = first.modification_rate;
    summary.shingle_size = first.shingle_size;
    summary.hamming_threshold = first.hamming_threshold;

    summary.policy = first.policy;
    summary.configured_threads = first.configured_threads;
    summary.actual_threads = first.actual_threads;
    summary.chunk_size = first.chunk_size;
    summary.warmup_runs = warmup_runs;
    summary.measured_runs = samples.size();
    summary.order_seed = first.order_seed;

    summary.parity_all_passed = true;
    for (const auto& s : samples) {
        if (!s.parity_passed) {
            summary.parity_all_passed = false;
        }
    }
    summary.candidate_pairs_count = first.candidate_pairs_count;
    summary.verified_pairs_count = first.verified_pairs_count;

    summary.occupied_buckets = first.occupied_buckets;
    summary.positive_work_buckets = first.positive_work_buckets;
    summary.max_bucket_size = first.max_bucket_size;
    summary.mean_bucket_size = first.mean_bucket_size;
    summary.total_estimated_pair_work = first.total_estimated_pair_work;
    summary.max_estimated_pair_work = first.max_estimated_pair_work;
    summary.mean_estimated_pair_work = first.mean_estimated_pair_work;
    summary.bucket_work_cv = first.bucket_work_cv;

    summary.selected_ordinary_policy = first.selected_ordinary_policy;
    summary.ordinary_buckets = first.ordinary_buckets;
    summary.ordinary_pair_work = first.ordinary_pair_work;
    summary.oversized_buckets = first.oversized_buckets;
    summary.oversized_pair_work = first.oversized_pair_work;
    summary.heavy_bucket_chunks = first.heavy_bucket_chunks;
    summary.max_generated_chunk_size = first.max_generated_chunk_size;

    // Collect vectors of timings and work-imbalance ratios across repetitions
    std::vector<double> fp_times;
    std::vector<double> bucket_times;
    std::vector<double> cand_total_times;
    std::vector<double> adap_dec_times;
    std::vector<double> ord_cand_times;
    std::vector<double> heavy_chunk_times;
    std::vector<double> heavy_proc_times;
    std::vector<double> merge_times;
    std::vector<double> ver_times;
    std::vector<double> total_times;
    std::vector<double> imbalance_ratios;

    fp_times.reserve(samples.size());
    bucket_times.reserve(samples.size());
    cand_total_times.reserve(samples.size());
    adap_dec_times.reserve(samples.size());
    ord_cand_times.reserve(samples.size());
    heavy_chunk_times.reserve(samples.size());
    heavy_proc_times.reserve(samples.size());
    merge_times.reserve(samples.size());
    ver_times.reserve(samples.size());
    total_times.reserve(samples.size());
    imbalance_ratios.reserve(samples.size());

    for (const auto& s : samples) {
        fp_times.push_back(s.fingerprinting_ms);
        bucket_times.push_back(s.bucket_construction_ms);
        cand_total_times.push_back(s.candidate_gen_total_ms);
        adap_dec_times.push_back(s.adaptive_decision_ms);
        ord_cand_times.push_back(s.ordinary_candidate_gen_ms);
        heavy_chunk_times.push_back(s.heavy_chunk_gen_ms);
        heavy_proc_times.push_back(s.heavy_pair_proc_ms);
        merge_times.push_back(s.merge_sort_dedup_ms);
        ver_times.push_back(s.verification_ms);
        total_times.push_back(s.total_pipeline_ms);
        imbalance_ratios.push_back(s.work_imbalance_ratio);
    }

    // Select the repetition with the median work_imbalance_ratio for representative per-thread work
    std::vector<size_t> order(samples.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return samples[a].work_imbalance_ratio < samples[b].work_imbalance_ratio;
    });
    const BenchmarkSampleRow& rep_sample = samples[order[order.size() / 2]];

    summary.participating_threads = rep_sample.participating_threads;
    summary.max_thread_pair_work = rep_sample.max_thread_pair_work;
    summary.min_thread_pair_work = rep_sample.min_thread_pair_work;
    summary.mean_thread_pair_work = rep_sample.mean_thread_pair_work;
    summary.work_imbalance_ratio = compute_median(imbalance_ratios);
    summary.total_units_processed = rep_sample.total_units_processed;
    summary.per_thread_work_distribution = rep_sample.per_thread_work_distribution;

    summary.median_fingerprinting_ms = compute_median(fp_times);
    summary.median_bucket_construction_ms = compute_median(bucket_times);
    summary.median_candidate_gen_total_ms = compute_median(cand_total_times);
    summary.iqr_candidate_gen_total_ms = compute_iqr(cand_total_times);
    summary.sd_candidate_gen_total_ms = compute_sample_sd(cand_total_times);
    summary.median_adaptive_decision_ms = compute_median(adap_dec_times);
    summary.median_ordinary_candidate_gen_ms = compute_median(ord_cand_times);
    summary.median_heavy_chunk_gen_ms = compute_median(heavy_chunk_times);
    summary.median_heavy_pair_proc_ms = compute_median(heavy_proc_times);
    summary.median_merge_sort_dedup_ms = compute_median(merge_times);
    summary.median_verification_ms = compute_median(ver_times);

    summary.median_total_ms = compute_median(total_times);
    summary.mean_total_ms = compute_mean(total_times);
    summary.iqr_total_ms = compute_iqr(total_times);
    summary.sd_total_ms = compute_sample_sd(total_times);
    summary.min_total_ms = *std::min_element(total_times.begin(), total_times.end());
    summary.max_total_ms = *std::max_element(total_times.begin(), total_times.end());

    double effective_seq_total = (seq_median_total_ms > 0.0)
        ? seq_median_total_ms
        : summary.median_total_ms;
    double effective_seq_cand = (seq_median_cand_ms > 0.0)
        ? seq_median_cand_ms
        : summary.median_candidate_gen_total_ms;

    if (summary.median_total_ms > 0.0 && effective_seq_total > 0.0) {
        summary.speedup_total = compute_speedup(effective_seq_total, summary.median_total_ms);
        summary.efficiency_total = compute_efficiency(summary.speedup_total, summary.configured_threads);
    }
    if (summary.median_candidate_gen_total_ms > 0.0 && effective_seq_cand > 0.0) {
        summary.speedup_candidate_gen = compute_speedup(
            effective_seq_cand, summary.median_candidate_gen_total_ms
        );
        summary.efficiency_candidate_gen = compute_efficiency(
            summary.speedup_candidate_gen, summary.configured_threads
        );
    }

    summary.band_suppression_enabled = first.band_suppression_enabled;
    summary.raw_candidates_evaluated = first.raw_candidates_evaluated;
    summary.raw_candidates_emitted = first.raw_candidates_emitted;
    summary.suppressed_pair_count = first.suppressed_pair_count;

    return summary;
}

namespace {

/**
 * @brief Platform-independent 64-bit SplitMix64 hash mixer used to derive a deterministic
 *        per-round shuffle seed from (order_seed, dataset_size, profile, round_index).
 */
uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

uint64_t total_bucket_pair_work(const std::vector<BucketInfo>& buckets) {
    uint64_t sum = 0;
    for (const auto& b : buckets) {
        sum += b.estimated_pair_work;
    }
    return sum;
}

BenchmarkSampleRow build_sample_row(
    const DatasetConfig& dcfg,
    const PipelineConfig& pcfg,
    BenchmarkPolicy policy,
    int configured_threads,
    int chunk_size,
    size_t rep_idx,
    uint64_t order_seed,
    size_t execution_order_index,
    size_t round_execution_rank,
    const std::string& round_execution_order,
    const PipelineResult& res,
    const AdaptivePipelineResult* adap_ptr,
    bool parity_ok
) {
    BenchmarkSampleRow row;
    row.dataset_size = dcfg.total_documents;
    row.profile = dcfg.profile;
    row.random_seed = dcfg.random_seed;
    row.min_words = dcfg.min_words;
    row.max_words = dcfg.max_words;
    row.exact_dup_ratio = dcfg.exact_duplicate_ratio;
    row.modified_copy_ratio = dcfg.modified_copy_ratio;
    row.unrelated_ratio = dcfg.unrelated_ratio;
    row.modification_rate = dcfg.modification_rate;
    row.shingle_size = pcfg.shingle_size;
    row.hamming_threshold = pcfg.max_hamming_threshold;

    row.policy = policy;
    row.configured_threads = configured_threads;
    row.actual_threads = res.actual_thread_count;
    row.chunk_size = chunk_size;
    row.repetition_index = rep_idx;
    row.order_seed = order_seed;
    row.execution_order_index = execution_order_index;
    row.round_execution_rank = round_execution_rank;
    row.round_execution_order = round_execution_order;

    row.parity_passed = parity_ok;
    row.candidate_pairs_count = res.candidate_pairs.size();
    row.verified_pairs_count = res.verified_pairs.size();

    row.occupied_buckets = res.workload_stats.occupied_buckets;
    row.positive_work_buckets = res.workload_stats.positive_work_buckets;
    row.max_bucket_size = res.workload_stats.max_bucket_size;
    row.mean_bucket_size = res.workload_stats.mean_bucket_size;
    row.total_estimated_pair_work = total_bucket_pair_work(res.buckets);
    row.max_estimated_pair_work = res.workload_stats.max_estimated_pair_work;
    row.mean_estimated_pair_work = res.workload_stats.mean_estimated_pair_work;
    row.bucket_work_cv = res.workload_stats.cv_estimated_pair_work;

    if (adap_ptr != nullptr) {
        const auto& diag = adap_ptr->diagnostics;
        const auto& at = adap_ptr->adaptive_timings;
        row.selected_ordinary_policy = to_string(diag.selected_ordinary_policy);
        row.ordinary_buckets = diag.ordinary_buckets;
        row.ordinary_pair_work = diag.ordinary_pair_work;
        row.oversized_buckets = diag.oversized_buckets;
        row.oversized_pair_work = diag.oversized_pair_work;
        row.heavy_bucket_chunks = diag.heavy_bucket_chunks;
        row.max_generated_chunk_size = diag.max_generated_chunk_size;

        row.fingerprinting_ms = at.fingerprinting_ms;
        row.bucket_construction_ms = at.bucket_construction_stats_ms;
        row.candidate_gen_total_ms = res.timings.candidate_generation_dedup_ms;
        row.adaptive_decision_ms = at.adaptive_decision_ms;
        row.ordinary_candidate_gen_ms = at.ordinary_candidate_gen_ms;
        row.heavy_chunk_gen_ms = at.heavy_chunk_generation_ms;
        row.heavy_pair_proc_ms = at.heavy_pair_processing_ms;
        row.merge_sort_dedup_ms = at.merge_sort_dedup_ms;
        row.verification_ms = at.verification_ms;
        row.total_pipeline_ms = at.total_pipeline_ms;
    } else {
        row.selected_ordinary_policy = to_string(policy);
        row.ordinary_buckets = res.workload_stats.positive_work_buckets;
        row.ordinary_pair_work = row.total_estimated_pair_work;
        row.oversized_buckets = 0;
        row.oversized_pair_work = 0;
        row.heavy_bucket_chunks = 0;
        row.max_generated_chunk_size = 0;

        row.fingerprinting_ms = res.timings.fingerprinting_ms;
        row.bucket_construction_ms = res.timings.bucket_construction_ms;
        row.candidate_gen_total_ms = res.timings.candidate_generation_dedup_ms;
        row.adaptive_decision_ms = 0.0;
        row.ordinary_candidate_gen_ms = res.timings.candidate_generation_dedup_ms;
        row.heavy_chunk_gen_ms = 0.0;
        row.heavy_pair_proc_ms = 0.0;
        row.merge_sort_dedup_ms = 0.0;
        row.verification_ms = res.timings.verification_ms;
        row.total_pipeline_ms = res.timings.total_pipeline_ms;
    }

    WorkBalanceMetrics wb = compute_work_balance_metrics(
        res.per_thread_pair_work,
        res.per_thread_units_processed,
        configured_threads,
        res.actual_thread_count
    );
    row.participating_threads = wb.participating_threads;
    row.max_thread_pair_work = wb.max_thread_pair_work;
    row.min_thread_pair_work = wb.min_thread_pair_work;
    row.mean_thread_pair_work = wb.mean_thread_pair_work;
    row.work_imbalance_ratio = wb.work_imbalance_ratio;
    row.total_units_processed = wb.total_units_processed;
    row.per_thread_work_distribution = wb.per_thread_work_distribution;

    row.band_suppression_enabled = res.band_suppression_enabled;
    row.raw_candidates_evaluated = res.raw_candidates_evaluated;
    row.raw_candidates_emitted = res.raw_candidates_emitted;
    row.suppressed_pair_count = res.suppressed_pair_count;

    return row;
}

} // namespace

std::vector<BenchmarkConfigSpec> build_dataset_config_specs(
    const std::vector<int>& thread_counts
) {
    if (thread_counts.empty()) {
        throw std::invalid_argument("build_dataset_config_specs: thread_counts cannot be empty.");
    }
    for (int t : thread_counts) {
        if (t <= 0) {
            throw std::invalid_argument(
                "build_dataset_config_specs: thread_count must be positive (got " +
                std::to_string(t) + ")."
            );
        }
    }

    std::vector<BenchmarkConfigSpec> specs;
    specs.reserve(1 + 4 * thread_counts.size());

    // 1. Sequential reference baseline (1 thread)
    specs.push_back(BenchmarkConfigSpec{BenchmarkPolicy::Sequential, 1});

    // 2. Parallel policies across all requested thread counts in canonical order
    const std::vector<BenchmarkPolicy> parallel_policies = {
        BenchmarkPolicy::Static,
        BenchmarkPolicy::Dynamic,
        BenchmarkPolicy::Guided,
        BenchmarkPolicy::Adaptive
    };
    for (BenchmarkPolicy policy : parallel_policies) {
        for (int t : thread_counts) {
            specs.push_back(BenchmarkConfigSpec{policy, t});
        }
    }
    return specs;
}

std::vector<BenchmarkConfigSpec> generate_shuffled_config_order(
    const std::vector<BenchmarkConfigSpec>& base_specs,
    uint64_t order_seed,
    size_t dataset_size,
    WorkloadProfile profile,
    size_t round_index
) {
    std::vector<BenchmarkConfigSpec> shuffled = base_specs;
    if (shuffled.size() <= 1) {
        return shuffled;
    }

    // Deterministically mix (order_seed, dataset_size, profile, round_index) into a 64-bit seed.
    // Using std::mt19937_64 with explicit Fisher-Yates index selection guarantees identical
    // permutations across compilers and standard library implementations.
    uint64_t mixed = splitmix64(order_seed);
    mixed = splitmix64(mixed ^ static_cast<uint64_t>(dataset_size));
    mixed = splitmix64(mixed ^ (static_cast<uint64_t>(profile) + 0x517cc1b727220a95ULL));
    mixed = splitmix64(mixed ^ static_cast<uint64_t>(round_index));

    std::mt19937_64 rng(mixed);
    for (size_t i = shuffled.size() - 1; i > 0; --i) {
        size_t j = static_cast<size_t>(rng() % static_cast<uint64_t>(i + 1));
        std::swap(shuffled[i], shuffled[j]);
    }
    return shuffled;
}

std::vector<BenchmarkConfigSpec> generate_paired_config_order(
    const std::vector<BenchmarkConfigSpec>& base_specs,
    uint64_t order_seed,
    size_t dataset_size,
    WorkloadProfile profile,
    size_t round_index
) {
    const std::vector<BenchmarkConfigSpec> shuffled_configs =
        generate_shuffled_config_order(base_specs, order_seed, dataset_size, profile, round_index);

    std::vector<BenchmarkConfigSpec> paired;
    paired.reserve(shuffled_configs.size() * 2);

    for (size_t k = 0; k < shuffled_configs.size(); ++k) {
        BenchmarkConfigSpec base_variant = shuffled_configs[k];
        base_variant.enable_band_suppression = false;

        BenchmarkConfigSpec supp_variant = shuffled_configs[k];
        supp_variant.enable_band_suppression = true;

        // Alternate which variant executes first across rounds and positions
        if (((round_index + k) & 1ULL) == 0ULL) {
            paired.push_back(base_variant);
            paired.push_back(supp_variant);
        } else {
            paired.push_back(supp_variant);
            paired.push_back(base_variant);
        }
    }

    return paired;
}

std::string format_config_order(const std::vector<BenchmarkConfigSpec>& specs) {
    bool has_suppression = false;
    for (const auto& s : specs) {
        if (s.enable_band_suppression) {
            has_suppression = true;
            break;
        }
    }

    std::ostringstream oss;
    for (size_t i = 0; i < specs.size(); ++i) {
        if (i > 0) {
            oss << ';';
        }
        oss << to_string(specs[i].policy) << ':' << specs[i].thread_count;
        if (has_suppression) {
            oss << ':' << (specs[i].enable_band_suppression ? "Supp" : "Base");
        }
    }
    return oss.str();
}

BenchmarkSuiteResult run_benchmark_suite(
    const BenchmarkSuiteConfig& suite_cfg,
    std::ostream* progress_os
) {
    if (suite_cfg.dataset_sizes.empty()) {
        throw std::invalid_argument("run_benchmark_suite: dataset_sizes cannot be empty.");
    }
    if (suite_cfg.profiles.empty()) {
        throw std::invalid_argument("run_benchmark_suite: profiles cannot be empty.");
    }
    if (suite_cfg.thread_counts.empty()) {
        throw std::invalid_argument("run_benchmark_suite: thread_counts cannot be empty.");
    }
    if (suite_cfg.measured_runs == 0) {
        throw std::invalid_argument("run_benchmark_suite: measured_runs must be at least 1.");
    }

    const std::vector<BenchmarkConfigSpec> base_specs =
        build_dataset_config_specs(suite_cfg.thread_counts);

    const bool paired_mode = suite_cfg.compare_band_suppression;
    const bool single_suppression_enabled =
        suite_cfg.enable_band_suppression || suite_cfg.pipeline_config.enable_band_suppression;

    std::vector<BenchmarkConfigSpec> canonical_specs;
    if (paired_mode) {
        canonical_specs.reserve(base_specs.size() * 2);
        for (const auto& bs : base_specs) {
            canonical_specs.push_back(BenchmarkConfigSpec{bs.policy, bs.thread_count, false});
            canonical_specs.push_back(BenchmarkConfigSpec{bs.policy, bs.thread_count, true});
        }
    } else {
        canonical_specs = base_specs;
        for (auto& cs : canonical_specs) {
            cs.enable_band_suppression = single_suppression_enabled;
        }
    }

    PipelineConfig baseline_ref_cfg = suite_cfg.pipeline_config;
    baseline_ref_cfg.enable_band_suppression = false;

    BenchmarkSuiteResult suite_res;
    size_t global_execution_order_index = 0;

    for (size_t dsize : suite_cfg.dataset_sizes) {
        for (WorkloadProfile profile : suite_cfg.profiles) {
            DatasetConfig dcfg;
            dcfg.total_documents = dsize;
            dcfg.min_words = suite_cfg.min_words;
            dcfg.max_words = suite_cfg.max_words;
            dcfg.exact_duplicate_ratio = suite_cfg.exact_duplicate_ratio;
            dcfg.modified_copy_ratio = suite_cfg.modified_copy_ratio;
            dcfg.unrelated_ratio = suite_cfg.unrelated_ratio;
            dcfg.modification_rate = suite_cfg.modification_rate;
            dcfg.random_seed = suite_cfg.base_seed;
            dcfg.profile = profile;

            if (progress_os != nullptr) {
                *progress_os << "\n===============================================================\n"
                             << " Dataset: N=" << dsize
                             << " | Profile=" << to_string(profile)
                             << " | Seed=" << dcfg.random_seed
                             << " | OrderSeed=" << suite_cfg.order_seed
                             << " | BandSupp=" << (paired_mode ? "PAIRED (Base+Supp)" : (single_suppression_enabled ? "ON" : "OFF")) << "\n"
                             << "===============================================================\n";
            }

            // Generate dataset ONCE per (dataset_size, profile, seed) and reuse across all runs
            const std::vector<Document> dataset = generate_dataset(dcfg);

            // Compute the baseline sequential reference result independently BEFORE randomized rounds
            // so that every warm-up and measured run (including band-suppression runs) is verified
            // against the unsuppressed Phase 9 baseline sequential reference.
            const PipelineResult reference_seq_result =
                run_sequential_pipeline(dataset, baseline_ref_cfg);

            // Helper lambda to run a single BenchmarkConfigSpec on `dataset` and verify parity
            auto run_and_verify_spec = [&](const BenchmarkConfigSpec& spec,
                                           const std::string& phase_label,
                                           PipelineResult& out_res,
                                           AdaptivePipelineResult& out_adap,
                                           bool& out_is_adaptive,
                                           int& out_chunk_size) -> ParityCheckResult {
                out_is_adaptive = false;
                out_chunk_size = 0;

                PipelineConfig spec_pipeline_cfg = suite_cfg.pipeline_config;
                spec_pipeline_cfg.enable_band_suppression = spec.enable_band_suppression;

                if (spec.policy == BenchmarkPolicy::Sequential) {
                    out_res = run_sequential_pipeline(dataset, spec_pipeline_cfg);
                } else if (spec.policy == BenchmarkPolicy::Adaptive) {
                    out_is_adaptive = true;
                    AdaptiveConfig acfg;
                    acfg.base_config = spec_pipeline_cfg;
                    acfg.enable_band_suppression = spec.enable_band_suppression;
                    acfg.thread_count = spec.thread_count;
                    acfg.cv_threshold = suite_cfg.cv_threshold;
                    acfg.oversized_bucket_multiplier = suite_cfg.oversized_bucket_multiplier;
                    acfg.max_pair_task_size = suite_cfg.max_pair_task_size;
                    acfg.ordinary_chunk_size = suite_cfg.ordinary_chunk_size;
                    out_chunk_size = static_cast<int>(acfg.max_pair_task_size);

                    out_adap = run_adaptive_pipeline(dataset, acfg);
                    out_res = out_adap.pipeline_result;
                } else {
                    ParallelConfig pcfg;
                    pcfg.base_config = spec_pipeline_cfg;
                    pcfg.enable_band_suppression = spec.enable_band_suppression;
                    pcfg.thread_count = spec.thread_count;
                    pcfg.chunk_size = suite_cfg.fixed_chunk_size;
                    out_chunk_size = pcfg.chunk_size;
                    if (spec.policy == BenchmarkPolicy::Static) {
                        pcfg.schedule_policy = SchedulePolicy::Static;
                    } else if (spec.policy == BenchmarkPolicy::Dynamic) {
                        pcfg.schedule_policy = SchedulePolicy::Dynamic;
                    } else {
                        pcfg.schedule_policy = SchedulePolicy::Guided;
                    }

                    out_res = run_parallel_pipeline(dataset, pcfg);
                }

                ParityCheckResult parity = verify_pipeline_parity(reference_seq_result, out_res);
                if (!parity.passed) {
                    throw std::runtime_error(
                        "Parity failure during " + std::string(to_string(spec.policy)) + " " +
                        phase_label + " (N=" + std::to_string(dsize) + ", T=" +
                        std::to_string(spec.thread_count) + ", Supp=" +
                        std::to_string(spec.enable_band_suppression ? 1 : 0) + "): " + parity.failure_reason
                    );
                }
                return parity;
            };

            // 1. Interleaved / Randomized Warm-up Rounds (also verified for parity)
            for (size_t w = 0; w < suite_cfg.warmup_runs; ++w) {
                const std::vector<BenchmarkConfigSpec> warm_order = paired_mode
                    ? generate_paired_config_order(base_specs, suite_cfg.order_seed, dsize, profile, w)
                    : generate_shuffled_config_order(canonical_specs, suite_cfg.order_seed, dsize, profile, w);
                for (const auto& spec : warm_order) {
                    PipelineResult warm_res;
                    AdaptivePipelineResult warm_adap;
                    bool is_adap = false;
                    int chunk_sz = 0;
                    run_and_verify_spec(
                        spec,
                        "warm-up[" + std::to_string(w) + "]",
                        warm_res,
                        warm_adap,
                        is_adap,
                        chunk_sz
                    );
                }
            }

            // 2. Interleaved / Randomized Measured Repetition Rounds
            std::vector<std::vector<BenchmarkSampleRow>> samples_by_canonical_idx(canonical_specs.size());
            for (auto& vec : samples_by_canonical_idx) {
                vec.reserve(suite_cfg.measured_runs);
            }

            for (size_t r = 0; r < suite_cfg.measured_runs; ++r) {
                const size_t round_idx = suite_cfg.warmup_runs + r;
                const std::vector<BenchmarkConfigSpec> round_order = paired_mode
                    ? generate_paired_config_order(
                        base_specs, suite_cfg.order_seed, dsize, profile, round_idx
                    )
                    : generate_shuffled_config_order(
                        canonical_specs, suite_cfg.order_seed, dsize, profile, round_idx
                    );
                const std::string round_order_str = format_config_order(round_order);

                for (size_t rank = 0; rank < round_order.size(); ++rank) {
                    const BenchmarkConfigSpec& spec = round_order[rank];
                    PipelineResult run_res;
                    AdaptivePipelineResult run_adap;
                    bool is_adap = false;
                    int chunk_sz = 0;

                    ParityCheckResult parity = run_and_verify_spec(
                        spec,
                        "rep[" + std::to_string(r) + "]",
                        run_res,
                        run_adap,
                        is_adap,
                        chunk_sz
                    );

                    PipelineConfig spec_pipeline_cfg = suite_cfg.pipeline_config;
                    spec_pipeline_cfg.enable_band_suppression = spec.enable_band_suppression;

                    BenchmarkSampleRow row = build_sample_row(
                        dcfg,
                        spec_pipeline_cfg,
                        spec.policy,
                        spec.thread_count,
                        chunk_sz,
                        r,
                        suite_cfg.order_seed,
                        global_execution_order_index++,
                        rank,
                        round_order_str,
                        run_res,
                        is_adap ? &run_adap : nullptr,
                        parity.passed
                    );

                    suite_res.raw_samples.push_back(row);

                    auto it = std::find(canonical_specs.begin(), canonical_specs.end(), spec);
                    if (it != canonical_specs.end()) {
                        size_t canonical_idx = static_cast<size_t>(std::distance(canonical_specs.begin(), it));
                        samples_by_canonical_idx[canonical_idx].push_back(row);
                    }
                }
            }

            // 3. Summarize in canonical order: Sequential first (to establish seq_median_total_ms
            //    and seq_median_cand_ms), followed by Static, Dynamic, Guided, and Adaptive.
            BenchmarkSummaryRow seq_summary = summarize_benchmark_samples(
                samples_by_canonical_idx[0],
                suite_cfg.warmup_runs,
                0.0,
                0.0
            );
            double seq_median_total_ms = seq_summary.median_total_ms;
            double seq_median_cand_ms = seq_summary.median_candidate_gen_total_ms;
            suite_res.summary_rows.push_back(seq_summary);

            if (progress_os != nullptr) {
                *progress_os << "  [Sequential | T=1 | Supp=" << (seq_summary.band_suppression_enabled ? "ON" : "OFF")
                             << "] Median Total=" << std::fixed
                             << std::setprecision(3) << seq_summary.median_total_ms
                             << " ms | CandGen=" << seq_summary.median_candidate_gen_total_ms
                             << " ms | MaxBucket=" << seq_summary.max_bucket_size
                             << " | EvalPairs=" << seq_summary.raw_candidates_evaluated
                             << " | EmitPairs=" << seq_summary.raw_candidates_emitted
                             << " | SuppPairs=" << seq_summary.suppressed_pair_count
                             << " | CV=" << std::setprecision(4) << seq_summary.bucket_work_cv << "\n";
            }

            for (size_t idx = 1; idx < canonical_specs.size(); ++idx) {
                BenchmarkSummaryRow summary = summarize_benchmark_samples(
                    samples_by_canonical_idx[idx],
                    suite_cfg.warmup_runs,
                    seq_median_total_ms,
                    seq_median_cand_ms
                );
                suite_res.summary_rows.push_back(summary);

                if (progress_os != nullptr) {
                    *progress_os << "  [" << std::setw(10) << std::left << to_string(summary.policy)
                                 << std::right << " | T=" << summary.configured_threads
                                 << " | Supp=" << std::setw(3) << (summary.band_suppression_enabled ? "ON" : "OFF") << "] "
                                 << "Median=" << std::fixed << std::setprecision(3)
                                 << std::setw(7) << summary.median_total_ms << " ms "
                                 << "(IQR=" << std::setw(5) << summary.iqr_total_ms << ") | "
                                 << "Speedup=" << std::setprecision(2) << std::setw(4)
                                 << summary.speedup_total << "x | "
                                 << "Eff=" << std::setw(4) << summary.efficiency_total << " | "
                                 << "Imbalance I=" << std::setprecision(2) << std::setw(4)
                                 << summary.work_imbalance_ratio
                                 << " (Part=" << summary.participating_threads << "/"
                                 << summary.configured_threads << ")\n";
                }
            }
        }
    }

    return suite_res;
}

void write_raw_csv(std::ostream& os, const std::vector<BenchmarkSampleRow>& rows) {
    os << "dataset_size,profile,random_seed,min_words,max_words,"
       << "exact_dup_ratio,modified_copy_ratio,unrelated_ratio,modification_rate,"
       << "shingle_size,hamming_threshold,policy,configured_threads,actual_threads,"
       << "chunk_size,repetition_index,order_seed,execution_order_index,"
       << "round_execution_rank,round_execution_order,"
       << "parity_passed,candidate_pairs_count,verified_pairs_count,"
       << "occupied_buckets,positive_work_buckets,max_bucket_size,mean_bucket_size,"
       << "total_estimated_pair_work,max_estimated_pair_work,mean_estimated_pair_work,bucket_work_cv,"
       << "selected_ordinary_policy,ordinary_buckets,ordinary_pair_work,"
       << "oversized_buckets,oversized_pair_work,heavy_bucket_chunks,max_generated_chunk_size,"
       << "participating_threads,max_thread_pair_work,min_thread_pair_work,mean_thread_pair_work,"
       << "work_imbalance_ratio,total_units_processed,per_thread_work_distribution,"
       << "fingerprinting_ms,bucket_construction_ms,candidate_gen_total_ms,"
       << "adaptive_decision_ms,ordinary_candidate_gen_ms,heavy_chunk_gen_ms,"
       << "heavy_pair_proc_ms,merge_sort_dedup_ms,verification_ms,total_pipeline_ms,"
       << "band_suppression_enabled,raw_candidates_evaluated,raw_candidates_emitted,suppressed_pair_count\n";

    os << std::fixed;
    for (const auto& r : rows) {
        os << r.dataset_size << ','
           << to_string(r.profile) << ','
           << r.random_seed << ','
           << r.min_words << ','
           << r.max_words << ','
           << std::setprecision(4)
           << r.exact_dup_ratio << ','
           << r.modified_copy_ratio << ','
           << r.unrelated_ratio << ','
           << r.modification_rate << ','
           << r.shingle_size << ','
           << r.hamming_threshold << ','
           << to_string(r.policy) << ','
           << r.configured_threads << ','
           << r.actual_threads << ','
           << r.chunk_size << ','
           << r.repetition_index << ','
           << r.order_seed << ','
           << r.execution_order_index << ','
           << r.round_execution_rank << ','
           << '"' << r.round_execution_order << '"' << ','
           << (r.parity_passed ? 1 : 0) << ','
           << r.candidate_pairs_count << ','
           << r.verified_pairs_count << ','
           << r.occupied_buckets << ','
           << r.positive_work_buckets << ','
           << r.max_bucket_size << ','
           << std::setprecision(4) << r.mean_bucket_size << ','
           << r.total_estimated_pair_work << ','
           << r.max_estimated_pair_work << ','
           << std::setprecision(4) << r.mean_estimated_pair_work << ','
           << std::setprecision(6) << r.bucket_work_cv << ','
           << r.selected_ordinary_policy << ','
           << r.ordinary_buckets << ','
           << r.ordinary_pair_work << ','
           << r.oversized_buckets << ','
           << r.oversized_pair_work << ','
           << r.heavy_bucket_chunks << ','
           << r.max_generated_chunk_size << ','
           << r.participating_threads << ','
           << r.max_thread_pair_work << ','
           << r.min_thread_pair_work << ','
           << std::setprecision(4) << r.mean_thread_pair_work << ','
           << std::setprecision(6) << r.work_imbalance_ratio << ','
           << r.total_units_processed << ','
           << '"' << r.per_thread_work_distribution << '"' << ','
           << std::setprecision(4)
           << r.fingerprinting_ms << ','
           << r.bucket_construction_ms << ','
           << r.candidate_gen_total_ms << ','
           << r.adaptive_decision_ms << ','
           << r.ordinary_candidate_gen_ms << ','
           << r.heavy_chunk_gen_ms << ','
           << r.heavy_pair_proc_ms << ','
           << r.merge_sort_dedup_ms << ','
           << r.verification_ms << ','
           << r.total_pipeline_ms << ','
           << (r.band_suppression_enabled ? 1 : 0) << ','
           << r.raw_candidates_evaluated << ','
           << r.raw_candidates_emitted << ','
           << r.suppressed_pair_count << '\n';
    }
}

void write_summary_csv(std::ostream& os, const std::vector<BenchmarkSummaryRow>& rows) {
    os << "dataset_size,profile,random_seed,min_words,max_words,"
       << "exact_dup_ratio,modified_copy_ratio,unrelated_ratio,modification_rate,"
       << "shingle_size,hamming_threshold,policy,configured_threads,actual_threads,"
       << "chunk_size,warmup_runs,measured_runs,order_seed,parity_all_passed,"
       << "candidate_pairs_count,verified_pairs_count,"
       << "occupied_buckets,positive_work_buckets,max_bucket_size,mean_bucket_size,"
       << "total_estimated_pair_work,max_estimated_pair_work,mean_estimated_pair_work,bucket_work_cv,"
       << "selected_ordinary_policy,ordinary_buckets,ordinary_pair_work,"
       << "oversized_buckets,oversized_pair_work,heavy_bucket_chunks,max_generated_chunk_size,"
       << "participating_threads,max_thread_pair_work,min_thread_pair_work,mean_thread_pair_work,"
       << "work_imbalance_ratio,total_units_processed,per_thread_work_distribution,"
       << "median_fingerprinting_ms,median_bucket_construction_ms,"
       << "median_candidate_gen_total_ms,iqr_candidate_gen_total_ms,sd_candidate_gen_total_ms,"
       << "median_adaptive_decision_ms,median_ordinary_candidate_gen_ms,"
       << "median_heavy_chunk_gen_ms,median_heavy_pair_proc_ms,median_merge_sort_dedup_ms,"
       << "median_verification_ms,"
       << "median_total_ms,mean_total_ms,iqr_total_ms,sd_total_ms,min_total_ms,max_total_ms,"
       << "speedup_total,efficiency_total,speedup_candidate_gen,efficiency_candidate_gen,"
       << "band_suppression_enabled,raw_candidates_evaluated,raw_candidates_emitted,suppressed_pair_count\n";

    os << std::fixed;
    for (const auto& r : rows) {
        os << r.dataset_size << ','
           << to_string(r.profile) << ','
           << r.random_seed << ','
           << r.min_words << ','
           << r.max_words << ','
           << std::setprecision(4)
           << r.exact_dup_ratio << ','
           << r.modified_copy_ratio << ','
           << r.unrelated_ratio << ','
           << r.modification_rate << ','
           << r.shingle_size << ','
           << r.hamming_threshold << ','
           << to_string(r.policy) << ','
           << r.configured_threads << ','
           << r.actual_threads << ','
           << r.chunk_size << ','
           << r.warmup_runs << ','
           << r.measured_runs << ','
           << r.order_seed << ','
           << (r.parity_all_passed ? 1 : 0) << ','
           << r.candidate_pairs_count << ','
           << r.verified_pairs_count << ','
           << r.occupied_buckets << ','
           << r.positive_work_buckets << ','
           << r.max_bucket_size << ','
           << std::setprecision(4) << r.mean_bucket_size << ','
           << r.total_estimated_pair_work << ','
           << r.max_estimated_pair_work << ','
           << std::setprecision(4) << r.mean_estimated_pair_work << ','
           << std::setprecision(6) << r.bucket_work_cv << ','
           << r.selected_ordinary_policy << ','
           << r.ordinary_buckets << ','
           << r.ordinary_pair_work << ','
           << r.oversized_buckets << ','
           << r.oversized_pair_work << ','
           << r.heavy_bucket_chunks << ','
           << r.max_generated_chunk_size << ','
           << r.participating_threads << ','
           << r.max_thread_pair_work << ','
           << r.min_thread_pair_work << ','
           << std::setprecision(4) << r.mean_thread_pair_work << ','
           << std::setprecision(6) << r.work_imbalance_ratio << ','
           << r.total_units_processed << ','
           << '"' << r.per_thread_work_distribution << '"' << ','
           << std::setprecision(4)
           << r.median_fingerprinting_ms << ','
           << r.median_bucket_construction_ms << ','
           << r.median_candidate_gen_total_ms << ','
           << r.iqr_candidate_gen_total_ms << ','
           << r.sd_candidate_gen_total_ms << ','
           << r.median_adaptive_decision_ms << ','
           << r.median_ordinary_candidate_gen_ms << ','
           << r.median_heavy_chunk_gen_ms << ','
           << r.median_heavy_pair_proc_ms << ','
           << r.median_merge_sort_dedup_ms << ','
           << r.median_verification_ms << ','
           << r.median_total_ms << ','
           << r.mean_total_ms << ','
           << r.iqr_total_ms << ','
           << r.sd_total_ms << ','
           << r.min_total_ms << ','
           << r.max_total_ms << ','
           << std::setprecision(4)
           << r.speedup_total << ','
           << r.efficiency_total << ','
           << r.speedup_candidate_gen << ','
           << r.efficiency_candidate_gen << ','
           << (r.band_suppression_enabled ? 1 : 0) << ','
           << r.raw_candidates_evaluated << ','
           << r.raw_candidates_emitted << ','
           << r.suppressed_pair_count << '\n';
    }
}

void print_summary_table(std::ostream& os, const std::vector<BenchmarkSummaryRow>& rows) {
    os << "\n========================================================================================================================\n";
    os << " Phase 7 Benchmark Summary Table (Medians Across Measured Repetitions)\n";
    os << "========================================================================================================================\n";
    os << "   N  | Profile  | Policy     | Th | Total(ms) | IQR(ms) | CandGen(ms) | Speedup | Effic | Imbal(I) | OvBkt | Chunks | Parity\n";
    os << "------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : rows) {
        os << " " << std::setw(4) << r.dataset_size << " | "
           << std::setw(8) << std::left << to_string(r.profile) << " | "
           << std::setw(10) << std::left << to_string(r.policy) << std::right << " | "
           << std::setw(2) << r.configured_threads << " | "
           << std::fixed << std::setprecision(3)
           << std::setw(9) << r.median_total_ms << " | "
           << std::setw(7) << r.iqr_total_ms << " | "
           << std::setw(11) << r.median_candidate_gen_total_ms << " | "
           << std::setprecision(2)
           << std::setw(6) << r.speedup_total << "x | "
           << std::setw(5) << r.efficiency_total << " | "
           << std::setw(8) << r.work_imbalance_ratio << " | "
           << std::setw(5) << r.oversized_buckets << " | "
           << std::setw(6) << r.heavy_bucket_chunks << " | "
           << (r.parity_all_passed ? "PASS" : "FAIL") << "\n";
    }
    os << "========================================================================================================================\n";
}
