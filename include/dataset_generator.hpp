#ifndef DATASET_GENERATOR_HPP
#define DATASET_GENERATOR_HPP

#include "document.hpp"
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

/**
 * @brief Workload distribution profile for ground-truth duplicate groups.
 *
 * - Balanced: Distributes duplicate copies relatively evenly across original
 *             seed documents, producing similar group sizes.
 * - Skewed:   Distributes duplicate copies according to a heavy-tailed/power-law
 *             skew, where a small number of original documents accumulate
 *             a large portion of copies, leaving other groups small.
 */
enum class WorkloadProfile {
    Balanced,
    Skewed
};

inline const char* to_string(WorkloadProfile profile) {
    switch (profile) {
        case WorkloadProfile::Balanced: return "Balanced";
        case WorkloadProfile::Skewed:   return "Skewed";
        default:                        return "Unknown";
    }
}

/**
 * @brief Configuration parameters for synthetic dataset generation.
 */
struct DatasetConfig {
    size_t total_documents{100};       ///< Total number of documents to generate.
    size_t min_words{40};              ///< Minimum token count per document.
    size_t max_words{80};              ///< Maximum token count per document.
    double exact_duplicate_ratio{0.20};///< Fraction of total documents that are exact copies.
    double modified_copy_ratio{0.20};  ///< Fraction of total documents that are modified copies.
    double unrelated_ratio{0.10};      ///< Fraction of total documents that are unrelated singletons.
    double modification_rate{0.15};    ///< Fraction of tokens modified in a ModifiedCopy.
    uint64_t random_seed{42};          ///< Deterministic pseudorandom generator seed.
    WorkloadProfile profile{WorkloadProfile::Balanced}; ///< Cluster size workload distribution.

    /**
     * @brief Validates configuration invariants. Throws std::invalid_argument if invalid.
     */
    void validate() const;
};

/**
 * @brief Comprehensive summary of a generated dataset.
 */
struct DatasetSummary {
    size_t total_documents{0};
    size_t original_count{0};
    size_t exact_copy_count{0};
    size_t modified_copy_count{0};
    size_t unrelated_count{0};
    size_t total_ground_truth_groups{0};
    size_t duplicate_groups_count{0};  ///< Number of groups containing >= 2 documents.
    size_t max_group_size{0};
    size_t min_group_size{0};
    WorkloadProfile profile{WorkloadProfile::Balanced};
    uint64_t random_seed{0};

    void print(std::ostream& os) const;
};

/**
 * @brief Generates a deterministic synthetic dataset based on the provided configuration.
 *
 * @param config Validated dataset configuration.
 * @return std::vector<Document> Generated collection of documents.
 */
std::vector<Document> generate_dataset(const DatasetConfig& config);

/**
 * @brief Computes a statistical summary for a generated dataset.
 *
 * @param dataset The document collection to analyze.
 * @param config The configuration used to generate the dataset.
 * @return DatasetSummary Summary statistics.
 */
DatasetSummary summarize_dataset(const std::vector<Document>& dataset, const DatasetConfig& config);

/**
 * @brief Exports documents to a tab-separated values (TSV) file.
 *
 * Each line contains: ID, GroupID, Kind, SourceID, and Text (newlines sanitized).
 *
 * @param filepath Destination file path.
 * @param dataset Document collection.
 * @return true if write succeeded, false otherwise.
 */
bool save_dataset_tsv(const std::string& filepath, const std::vector<Document>& dataset);

#endif // DATASET_GENERATOR_HPP
