#include "dataset_generator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

// Curated vocabulary of 120 readable domain words for synthetic text generation
const std::array<std::string, 120> kVocabulary = {
    "algorithm", "architecture", "analysis", "array", "bandwidth", "benchmark",
    "bottleneck", "buffer", "cache", "capacity", "cluster", "compiler",
    "computation", "concurrency", "contention", "core", "cost", "cycle",
    "data", "database", "deadlock", "decomposition", "deduplication", "dependency",
    "deterministic", "device", "distribution", "document", "dynamic", "efficiency",
    "execution", "experiment", "fault", "fingerprint", "flow", "framework",
    "granularity", "guided", "hardware", "hashing", "heuristic", "hierarchical",
    "index", "instruction", "integer", "interconnect", "iteration", "kernel",
    "latency", "library", "linkage", "load", "locality", "lock",
    "loop", "machine", "memory", "message", "methodology", "metric",
    "migration", "multicore", "multithreaded", "mutex", "network", "node",
    "numa", "openmp", "operating", "operation", "optimization", "overhead",
    "packet", "parallel", "partition", "performance", "pipeline", "platform",
    "pointer", "policy", "pool", "processor", "profiling", "protocol",
    "queue", "race", "random", "reduction", "register", "reproducibility",
    "resource", "response", "runtime", "scalability", "scheduling", "segment",
    "semaphore", "sequential", "shared", "signal", "simulation", "simhash",
    "size", "skew", "socket", "speedup", "static", "storage",
    "strategy", "stream", "structure", "subband", "synchronization", "system",
    "task", "throughput", "threshold", "timer", "token", "topology"
};

// Generates a space-separated sequence of random tokens from the vocabulary
std::string generate_random_text(std::mt19937_64& rng, size_t min_tokens, size_t max_tokens) {
    std::uniform_int_distribution<size_t> length_dist(min_tokens, max_tokens);
    std::uniform_int_distribution<size_t> vocab_dist(0, kVocabulary.size() - 1);

    size_t length = length_dist(rng);
    std::string result;
    result.reserve(length * 10); // Reserve approximate capacity

    for (size_t i = 0; i < length; ++i) {
        if (i > 0) {
            result += ' ';
        }
        result += kVocabulary[vocab_dist(rng)];
    }
    return result;
}

// Splits space-separated text into individual tokens
std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

// Joins tokens back into space-separated text
std::string join_tokens(const std::vector<std::string>& tokens) {
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            result += ' ';
        }
        result += tokens[i];
    }
    return result;
}

// Modifies a source text by replacing a controlled proportion of tokens
std::string modify_text(const std::string& source_text, double modification_rate, std::mt19937_64& rng) {
    std::vector<std::string> tokens = tokenize(source_text);
    if (tokens.empty()) {
        return "";
    }

    // Determine number of tokens to substitute (at least 1, at most tokens.size())
    size_t changes = static_cast<size_t>(std::ceil(static_cast<double>(tokens.size()) * modification_rate));
    changes = std::max<size_t>(1, std::min(changes, tokens.size()));

    // Create a vector of token indices and shuffle to pick substitution positions
    std::vector<size_t> indices(tokens.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::uniform_int_distribution<size_t> vocab_dist(0, kVocabulary.size() - 1);

    for (size_t i = 0; i < changes; ++i) {
        size_t target_idx = indices[i];
        const std::string& original_token = tokens[target_idx];

        // Pick a replacement token that is distinct from the original token
        std::string replacement;
        do {
            replacement = kVocabulary[vocab_dist(rng)];
        } while (replacement == original_token);

        tokens[target_idx] = replacement;
    }

    return join_tokens(tokens);
}

} // namespace

void DatasetConfig::validate() const {
    if (min_words == 0) {
        throw std::invalid_argument("DatasetConfig::validate: min_words must be greater than 0.");
    }
    if (min_words > max_words) {
        throw std::invalid_argument("DatasetConfig::validate: min_words cannot exceed max_words.");
    }
    if (exact_duplicate_ratio < 0.0 || exact_duplicate_ratio > 1.0) {
        throw std::invalid_argument("DatasetConfig::validate: exact_duplicate_ratio must be in [0.0, 1.0].");
    }
    if (modified_copy_ratio < 0.0 || modified_copy_ratio > 1.0) {
        throw std::invalid_argument("DatasetConfig::validate: modified_copy_ratio must be in [0.0, 1.0].");
    }
    if (unrelated_ratio < 0.0 || unrelated_ratio > 1.0) {
        throw std::invalid_argument("DatasetConfig::validate: unrelated_ratio must be in [0.0, 1.0].");
    }
    if (exact_duplicate_ratio + modified_copy_ratio + unrelated_ratio > 1.0) {
        throw std::invalid_argument(
            "DatasetConfig::validate: Sum of exact_duplicate_ratio, modified_copy_ratio, "
            "and unrelated_ratio cannot exceed 1.0."
        );
    }
    if (modification_rate < 0.0 || modification_rate > 1.0) {
        throw std::invalid_argument("DatasetConfig::validate: modification_rate must be in [0.0, 1.0].");
    }

    // If duplicate or modified copies are requested, we must allow at least one original document
    if (total_documents > 1 && (exact_duplicate_ratio + modified_copy_ratio) > 0.0) {
        double copies_ratio = exact_duplicate_ratio + modified_copy_ratio + unrelated_ratio;
        if (copies_ratio >= 1.0) {
            throw std::invalid_argument(
                "DatasetConfig::validate: At least one original document must be permitted "
                "when duplicate copies are requested (ratio sum cannot be 1.0)."
            );
        }
    }
}

std::vector<Document> generate_dataset(const DatasetConfig& config) {
    config.validate();

    if (config.total_documents == 0) {
        return {};
    }

    std::mt19937_64 rng(config.random_seed);
    std::vector<Document> dataset;
    dataset.reserve(config.total_documents);

    // Compute target counts for each category
    size_t exact_copy_count = static_cast<size_t>(
        std::round(static_cast<double>(config.total_documents) * config.exact_duplicate_ratio)
    );
    size_t modified_copy_count = static_cast<size_t>(
        std::round(static_cast<double>(config.total_documents) * config.modified_copy_ratio)
    );
    size_t unrelated_count = static_cast<size_t>(
        std::round(static_cast<double>(config.total_documents) * config.unrelated_ratio)
    );

    // Ensure we always have at least one original if total_documents >= 1
    size_t non_original_count = exact_copy_count + modified_copy_count + unrelated_count;
    if (non_original_count >= config.total_documents) {
        if (config.total_documents > 0) {
            exact_copy_count = (exact_copy_count > 0) ? exact_copy_count - 1 : 0;
            non_original_count = exact_copy_count + modified_copy_count + unrelated_count;
        }
    }
    size_t original_count = config.total_documents - non_original_count;
    if (original_count == 0 && config.total_documents > 0) {
        original_count = 1;
        if (exact_copy_count > 0) exact_copy_count--;
        else if (modified_copy_count > 0) modified_copy_count--;
        else if (unrelated_count > 0) unrelated_count--;
    }

    uint64_t next_doc_id = 0;
    uint64_t next_group_id = 1;

    // 1. Generate Original Documents (seed masters for duplicate groups)
    std::vector<size_t> original_indices;
    original_indices.reserve(original_count);

    for (size_t i = 0; i < original_count; ++i) {
        Document doc;
        doc.id = next_doc_id++;
        doc.group_id = next_group_id++;
        doc.kind = DocumentKind::Original;
        doc.source_id = -1;
        doc.text = generate_random_text(rng, config.min_words, config.max_words);

        original_indices.push_back(dataset.size());
        dataset.push_back(std::move(doc));
    }

    // Helper: Select which original document to derive a copy from
    // Sets up probabilities depending on whether Balanced or Skewed profile is selected
    std::vector<double> original_weights(original_count, 1.0);
    if (config.profile == WorkloadProfile::Skewed && original_count > 1) {
        // Zipfian / power-law skew: weight_i = 1 / (i + 1)^2.0
        // The first few originals receive the vast majority of duplicate derivatives
        for (size_t i = 0; i < original_count; ++i) {
            double rank = static_cast<double>(i + 1);
            original_weights[i] = 1.0 / (rank * rank);
        }
    }
    std::discrete_distribution<size_t> parent_selector(original_weights.begin(), original_weights.end());

    // 2. Generate Exact Copies
    for (size_t i = 0; i < exact_copy_count; ++i) {
        size_t parent_idx = 0;
        if (config.profile == WorkloadProfile::Balanced) {
            // Round-robin distribution ensures even cluster sizes
            parent_idx = i % original_count;
        } else {
            parent_idx = parent_selector(rng);
        }

        const Document& parent = dataset[original_indices[parent_idx]];

        Document doc;
        doc.id = next_doc_id++;
        doc.group_id = parent.group_id; // Shares ground-truth duplicate group
        doc.kind = DocumentKind::ExactCopy;
        doc.source_id = static_cast<int64_t>(parent.id);
        doc.text = parent.text; // Verbatim duplicate

        dataset.push_back(std::move(doc));
    }

    // 3. Generate Modified Copies
    for (size_t i = 0; i < modified_copy_count; ++i) {
        size_t parent_idx = 0;
        if (config.profile == WorkloadProfile::Balanced) {
            parent_idx = (exact_copy_count + i) % original_count;
        } else {
            parent_idx = parent_selector(rng);
        }

        const Document& parent = dataset[original_indices[parent_idx]];

        Document doc;
        doc.id = next_doc_id++;
        doc.group_id = parent.group_id; // Shares ground-truth duplicate group
        doc.kind = DocumentKind::ModifiedCopy;
        doc.source_id = static_cast<int64_t>(parent.id);
        doc.text = modify_text(parent.text, config.modification_rate, rng);

        dataset.push_back(std::move(doc));
    }

    // 4. Generate Unrelated Documents (singletons with distinct group IDs)
    for (size_t i = 0; i < unrelated_count; ++i) {
        Document doc;
        doc.id = next_doc_id++;
        doc.group_id = next_group_id++; // Fresh unique group
        doc.kind = DocumentKind::Unrelated;
        doc.source_id = -1;
        doc.text = generate_random_text(rng, config.min_words, config.max_words);

        dataset.push_back(std::move(doc));
    }

    return dataset;
}

DatasetSummary summarize_dataset(const std::vector<Document>& dataset, const DatasetConfig& config) {
    DatasetSummary summary;
    summary.total_documents = dataset.size();
    summary.profile = config.profile;
    summary.random_seed = config.random_seed;

    if (dataset.empty()) {
        return summary;
    }

    std::unordered_map<uint64_t, size_t> group_counts;

    for (const auto& doc : dataset) {
        switch (doc.kind) {
            case DocumentKind::Original:     summary.original_count++; break;
            case DocumentKind::ExactCopy:    summary.exact_copy_count++; break;
            case DocumentKind::ModifiedCopy: summary.modified_copy_count++; break;
            case DocumentKind::Unrelated:    summary.unrelated_count++; break;
        }
        group_counts[doc.group_id]++;
    }

    summary.total_ground_truth_groups = group_counts.size();

    size_t max_size = 0;
    size_t min_size = std::numeric_limits<size_t>::max();

    for (const auto& pair : group_counts) {
        if (pair.second > 1) {
            summary.duplicate_groups_count++;
        }
        if (pair.second > max_size) {
            max_size = pair.second;
        }
        if (pair.second < min_size) {
            min_size = pair.second;
        }
    }

    summary.max_group_size = max_size;
    summary.min_group_size = (group_counts.empty()) ? 0 : min_size;

    return summary;
}

void DatasetSummary::print(std::ostream& os) const {
    os << "---------------------------------------------------------\n";
    os << " Synthetic Dataset Summary\n";
    os << "---------------------------------------------------------\n";
    os << "  Total Documents          : " << total_documents << "\n";
    os << "  Original Documents       : " << original_count << "\n";
    os << "  Exact Copies             : " << exact_copy_count << "\n";
    os << "  Modified Copies          : " << modified_copy_count << "\n";
    os << "  Unrelated Singletons     : " << unrelated_count << "\n";
    os << "  Total Ground-Truth Groups: " << total_ground_truth_groups << "\n";
    os << "  Duplicate Groups (size>1): " << duplicate_groups_count << "\n";
    os << "  Max Group Size           : " << max_group_size << "\n";
    os << "  Min Group Size           : " << min_group_size << "\n";
    os << "  Workload Profile         : " << to_string(profile) << "\n";
    os << "  Random Seed              : " << random_seed << "\n";
    os << "---------------------------------------------------------\n";
}

bool save_dataset_tsv(const std::string& filepath, const std::vector<Document>& dataset) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        return false;
    }

    out << "id\tgroup_id\tkind\tsource_id\ttext\n";
    for (const auto& doc : dataset) {
        // Sanitize text by replacing tabs and newlines with spaces
        std::string safe_text = doc.text;
        std::replace(safe_text.begin(), safe_text.end(), '\t', ' ');
        std::replace(safe_text.begin(), safe_text.end(), '\n', ' ');

        out << doc.id << '\t'
            << doc.group_id << '\t'
            << to_string(doc.kind) << '\t'
            << doc.source_id << '\t'
            << safe_text << '\n';
    }

    return true;
}
