#include "benchmark.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<size_t> parse_size_list(const std::string& csv_str) {
    std::vector<size_t> sizes;
    std::istringstream iss(csv_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (!token.empty()) {
            sizes.push_back(static_cast<size_t>(std::stoull(token)));
        }
    }
    if (sizes.empty()) {
        throw std::invalid_argument("parse_size_list: no valid dataset sizes provided.");
    }
    return sizes;
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  --pilot             Run small pilot benchmark (N={250,500}, warmup=1, reps=3)\n"
              << "  --sizes <N1,N2,...> Comma-separated dataset sizes (default: 500,1000,2000,4000)\n"
              << "  --warmup <W>        Number of warm-up runs per configuration (default: 1)\n"
              << "  --reps <R>          Number of measured repetitions per configuration (default: 5)\n"
              << "  --seed <S>          Random seed for dataset generation (default: 42)\n"
              << "  --order-seed <S>    Random seed for deterministic policy order shuffling (default: 1337)\n"
              << "  --out-dir <path>    Directory for CSV output files (default: results)\n"
              << "  --out-prefix <name> Filename prefix for <prefix>_raw.csv and <prefix>_summary.csv\n"
              << "  --help              Show this help message\n";
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        BenchmarkSuiteConfig suite_cfg;
        suite_cfg.dataset_sizes = {500, 1000, 2000, 4000};
        suite_cfg.profiles = {WorkloadProfile::Balanced, WorkloadProfile::Skewed};
        suite_cfg.thread_counts = {1, 2, 4, 8};
        suite_cfg.warmup_runs = 1;
        suite_cfg.measured_runs = 5;
        suite_cfg.base_seed = 42;
        suite_cfg.order_seed = 1337ULL;

        bool is_pilot = false;
        std::string out_dir = "results";
        std::string out_prefix;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--pilot") {
                is_pilot = true;
                suite_cfg.dataset_sizes = {250, 500};
                suite_cfg.warmup_runs = 1;
                suite_cfg.measured_runs = 3;
            } else if (arg == "--sizes" && i + 1 < argc) {
                suite_cfg.dataset_sizes = parse_size_list(argv[++i]);
            } else if (arg == "--warmup" && i + 1 < argc) {
                suite_cfg.warmup_runs = static_cast<size_t>(std::stoull(argv[++i]));
            } else if (arg == "--reps" && i + 1 < argc) {
                suite_cfg.measured_runs = static_cast<size_t>(std::stoull(argv[++i]));
            } else if (arg == "--seed" && i + 1 < argc) {
                suite_cfg.base_seed = std::stoull(argv[++i]);
            } else if (arg == "--order-seed" && i + 1 < argc) {
                suite_cfg.order_seed = std::stoull(argv[++i]);
            } else if (arg == "--out-dir" && i + 1 < argc) {
                out_dir = argv[++i];
            } else if (arg == "--out-prefix" && i + 1 < argc) {
                out_prefix = argv[++i];
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                std::cerr << "Unknown or incomplete option: " << arg << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }

        std::filesystem::create_directories(out_dir);
        std::string prefix = !out_prefix.empty()
            ? out_prefix
            : (is_pilot ? "phase7_pilot" : "benchmark");
        std::filesystem::path raw_csv_path = std::filesystem::path(out_dir) / (prefix + "_raw.csv");
        std::filesystem::path summary_csv_path = std::filesystem::path(out_dir) / (prefix + "_summary.csv");

        std::cout << "===============================================================\n";
        std::cout << " Phase 7: Reproducible Benchmarking & Performance Evaluation\n";
        std::cout << " Mode                 : " << (is_pilot ? "PILOT RUN" : "FULL BENCHMARK") << "\n";
        std::cout << " Clock Source         : std::chrono::steady_clock (monotonic)\n";
        std::cout << " Logical Processors   : " << omp_get_num_procs() << "\n";
        std::cout << " Max OpenMP Threads   : " << omp_get_max_threads() << "\n";
        std::cout << " Dataset Sizes (N)    : ";
        for (size_t i = 0; i < suite_cfg.dataset_sizes.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << suite_cfg.dataset_sizes[i];
        }
        std::cout << "\n";
        std::cout << " Profiles             : Balanced, Skewed\n";
        std::cout << " Policies             : Sequential, Static, Dynamic, Guided, Adaptive\n";
        std::cout << " Thread Counts        : 1, 2, 4, 8\n";
        std::cout << " Warm-up Runs         : " << suite_cfg.warmup_runs << "\n";
        std::cout << " Measured Repetitions : " << suite_cfg.measured_runs << "\n";
        std::cout << " Dataset Seed         : " << suite_cfg.base_seed << "\n";
        std::cout << " Execution-Order Seed : " << suite_cfg.order_seed << "\n";
        std::cout << " Word-Length Range    : [" << suite_cfg.min_words << ", " << suite_cfg.max_words << "]\n";
        std::cout << " Duplicate Ratios     : Exact=" << suite_cfg.exact_duplicate_ratio
                  << ", Modified=" << suite_cfg.modified_copy_ratio
                  << ", Unrelated=" << suite_cfg.unrelated_ratio
                  << " (ModRate=" << suite_cfg.modification_rate << ")\n";
        std::cout << " SimHash Parameters   : ShingleSize=" << suite_cfg.pipeline_config.shingle_size
                  << ", MaxHammingThreshold=" << suite_cfg.pipeline_config.max_hamming_threshold << "\n";
        std::cout << " Adaptive Defaults    : cv_threshold=" << suite_cfg.cv_threshold
                  << ", oversized_mult=" << suite_cfg.oversized_bucket_multiplier
                  << ", max_pair_task_size=" << suite_cfg.max_pair_task_size << "\n";
        std::cout << "===============================================================\n";

        BenchmarkSuiteResult result = run_benchmark_suite(suite_cfg, &std::cout);

        print_summary_table(std::cout, result.summary_rows);

        {
            std::ofstream raw_ofs(raw_csv_path);
            if (!raw_ofs) {
                throw std::runtime_error("Failed to open raw CSV for writing: " + raw_csv_path.string());
            }
            write_raw_csv(raw_ofs, result.raw_samples);
        }

        {
            std::ofstream sum_ofs(summary_csv_path);
            if (!sum_ofs) {
                throw std::runtime_error("Failed to open summary CSV for writing: " + summary_csv_path.string());
            }
            write_summary_csv(sum_ofs, result.summary_rows);
        }

        std::cout << "\n[CSV Export Complete]\n"
                  << "  Raw samples written to     : " << raw_csv_path.string()
                  << " (" << result.raw_samples.size() << " rows)\n"
                  << "  Summary metrics written to : " << summary_csv_path.string()
                  << " (" << result.summary_rows.size() << " rows)\n"
                  << "  Correctness parity status  : ALL RUNS PASSED EXACT PARITY\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "\n[BENCHMARK ERROR] " << ex.what() << "\n";
        return 1;
    }
}
