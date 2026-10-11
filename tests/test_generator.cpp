#include "dataset_generator.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

// Simple macro-free test assertion helper that prints informative failure context
void check(bool condition, const std::string& test_name, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << test_name << ": " << message << "\n";
        std::exit(1);
    }
    std::cout << "[PASS] " << test_name << "\n";
}

void test_empty_dataset() {
    DatasetConfig config;
    config.total_documents = 0;
    auto dataset = generate_dataset(config);
    check(dataset.empty(), "test_empty_dataset", "Dataset should contain 0 documents.");
    
    auto summary = summarize_dataset(dataset, config);
    check(summary.total_documents == 0, "test_empty_dataset_summary", "Summary count must be 0.");
}

void test_one_document_dataset() {
    DatasetConfig config;
    config.total_documents = 1;
    config.exact_duplicate_ratio = 0.0;
    config.modified_copy_ratio = 0.0;
    config.unrelated_ratio = 0.0;
    
    auto dataset = generate_dataset(config);
    check(dataset.size() == 1, "test_one_document_size", "Dataset must have size 1.");
    check(dataset[0].id == 0, "test_one_document_id", "ID must be 0.");
    check(dataset[0].kind == DocumentKind::Original, "test_one_document_kind", "Must be Original.");
    check(!dataset[0].text.empty(), "test_one_document_text", "Text must not be empty.");
}

void test_unique_document_ids() {
    DatasetConfig config;
    config.total_documents = 100;
    auto dataset = generate_dataset(config);

    std::unordered_set<uint64_t> ids;
    for (const auto& doc : dataset) {
        ids.insert(doc.id);
    }
    check(ids.size() == dataset.size(), "test_unique_document_ids", "All document IDs must be unique.");
}

void test_exact_copies_match_source() {
    DatasetConfig config;
    config.total_documents = 100;
    config.exact_duplicate_ratio = 0.30;
    config.modified_copy_ratio = 0.0;
    auto dataset = generate_dataset(config);

    std::unordered_map<uint64_t, const Document*> doc_map;
    for (const auto& doc : dataset) {
        doc_map[doc.id] = &doc;
    }

    size_t exact_copy_verified = 0;
    for (const auto& doc : dataset) {
        if (doc.kind == DocumentKind::ExactCopy) {
            check(doc.source_id >= 0, "test_exact_copies_source_id", "source_id must be non-negative.");
            auto it = doc_map.find(static_cast<uint64_t>(doc.source_id));
            check(it != doc_map.end(), "test_exact_copies_source_exists", "Source document must exist.");
            check(doc.text == it->second->text, "test_exact_copies_text_match", "Exact copy text must match source verbatim.");
            check(doc.group_id == it->second->group_id, "test_exact_copies_group_match", "Must share ground-truth group_id.");
            exact_copy_verified++;
        }
    }
    check(exact_copy_verified > 0, "test_exact_copies_count", "Must have verified exact copies.");
}

void test_modified_copies_derived_from_source() {
    DatasetConfig config;
    config.total_documents = 100;
    config.exact_duplicate_ratio = 0.0;
    config.modified_copy_ratio = 0.30;
    config.modification_rate = 0.20;
    auto dataset = generate_dataset(config);

    std::unordered_map<uint64_t, const Document*> doc_map;
    for (const auto& doc : dataset) {
        doc_map[doc.id] = &doc;
    }

    size_t modified_copy_verified = 0;
    for (const auto& doc : dataset) {
        if (doc.kind == DocumentKind::ModifiedCopy) {
            check(doc.source_id >= 0, "test_modified_copies_source_id", "source_id must be non-negative.");
            auto it = doc_map.find(static_cast<uint64_t>(doc.source_id));
            check(it != doc_map.end(), "test_modified_copies_source_exists", "Source document must exist.");
            check(doc.text != it->second->text, "test_modified_copies_text_diff", "Modified copy text must differ from source.");
            check(doc.group_id == it->second->group_id, "test_modified_copies_group_match", "Must share ground-truth group_id.");
            modified_copy_verified++;
        }
    }
    check(modified_copy_verified > 0, "test_modified_copies_count", "Must have verified modified copies.");
}

void test_unrelated_documents_are_singletons() {
    DatasetConfig config;
    config.total_documents = 100;
    config.exact_duplicate_ratio = 0.20;
    config.modified_copy_ratio = 0.20;
    config.unrelated_ratio = 0.20;
    auto dataset = generate_dataset(config);

    std::unordered_map<uint64_t, size_t> group_sizes;
    for (const auto& doc : dataset) {
        group_sizes[doc.group_id]++;
    }

    size_t unrelated_count = 0;
    for (const auto& doc : dataset) {
        if (doc.kind == DocumentKind::Unrelated) {
            check(doc.source_id == -1, "test_unrelated_source_id", "Unrelated source_id must be -1.");
            check(group_sizes[doc.group_id] == 1, "test_unrelated_singleton", "Unrelated documents must belong to singleton group.");
            unrelated_count++;
        }
    }
    check(unrelated_count > 0, "test_unrelated_count", "Must have generated unrelated documents.");
}

void test_deterministic_reproducibility() {
    DatasetConfig config;
    config.total_documents = 50;
    config.random_seed = 12345;

    auto dataset1 = generate_dataset(config);
    auto dataset2 = generate_dataset(config);

    check(dataset1.size() == dataset2.size(), "test_reproducibility_size", "Datasets must have identical size.");
    for (size_t i = 0; i < dataset1.size(); ++i) {
        check(dataset1[i].id == dataset2[i].id, "test_reproducibility_id", "IDs must match.");
        check(dataset1[i].group_id == dataset2[i].group_id, "test_reproducibility_group_id", "group_ids must match.");
        check(dataset1[i].kind == dataset2[i].kind, "test_reproducibility_kind", "kinds must match.");
        check(dataset1[i].source_id == dataset2[i].source_id, "test_reproducibility_source_id", "source_ids must match.");
        check(dataset1[i].text == dataset2[i].text, "test_reproducibility_text", "texts must match identically.");
    }
}

void test_different_seeds_accepted() {
    DatasetConfig config1;
    config1.total_documents = 50;
    config1.random_seed = 11111;

    DatasetConfig config2 = config1;
    config2.random_seed = 99999;

    auto d1 = generate_dataset(config1);
    auto d2 = generate_dataset(config2);

    check(d1.size() == d2.size(), "test_diff_seeds_size", "Both seeds produce datasets of equal count.");
    // At least one original text should differ between seeds
    bool found_difference = false;
    for (size_t i = 0; i < d1.size(); ++i) {
        if (d1[i].text != d2[i].text) {
            found_difference = true;
            break;
        }
    }
    check(found_difference, "test_diff_seeds_text", "Different seeds should produce different document text.");
}

void test_invalid_configurations_rejected() {
    // 1. min_words == 0
    {
        DatasetConfig config;
        config.min_words = 0;
        bool threw = false;
        try { config.validate(); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "test_invalid_min_words_zero", "min_words = 0 must be rejected.");
    }
    // 2. min_words > max_words
    {
        DatasetConfig config;
        config.min_words = 100;
        config.max_words = 50;
        bool threw = false;
        try { config.validate(); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "test_invalid_min_greater_max", "min_words > max_words must be rejected.");
    }
    // 3. Ratio sum > 1.0
    {
        DatasetConfig config;
        config.exact_duplicate_ratio = 0.60;
        config.modified_copy_ratio = 0.50;
        bool threw = false;
        try { config.validate(); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "test_invalid_ratio_sum_overflow", "ratio sum > 1.0 must be rejected.");
    }
    // 4. Negative ratio
    {
        DatasetConfig config;
        config.exact_duplicate_ratio = -0.10;
        bool threw = false;
        try { config.validate(); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "test_invalid_negative_ratio", "Negative ratio must be rejected.");
    }
    // 5. Modification rate > 1.0
    {
        DatasetConfig config;
        config.modification_rate = 1.5;
        bool threw = false;
        try { config.validate(); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "test_invalid_mod_rate_high", "modification_rate > 1.0 must be rejected.");
    }
}

void test_balanced_vs_skewed_profiles() {
    DatasetConfig balanced_config;
    balanced_config.total_documents = 200;
    balanced_config.exact_duplicate_ratio = 0.35;
    balanced_config.modified_copy_ratio = 0.35;
    balanced_config.unrelated_ratio = 0.05;
    balanced_config.random_seed = 42;
    balanced_config.profile = WorkloadProfile::Balanced;

    DatasetConfig skewed_config = balanced_config;
    skewed_config.profile = WorkloadProfile::Skewed;

    auto balanced_data = generate_dataset(balanced_config);
    auto skewed_data = generate_dataset(skewed_config);

    auto balanced_sum = summarize_dataset(balanced_data, balanced_config);
    auto skewed_sum = summarize_dataset(skewed_data, skewed_config);

    check(balanced_sum.total_documents == 200, "test_profiles_balanced_count", "Balanced count must match.");
    check(skewed_sum.total_documents == 200, "test_profiles_skewed_count", "Skewed count must match.");

    // The skewed profile concentrates duplicates into heavy-hitter clusters,
    // so its maximum group size should be significantly larger than balanced
    check(skewed_sum.max_group_size > balanced_sum.max_group_size,
          "test_skewed_profile_higher_max_group",
          "Skewed profile must generate a larger maximum group size than Balanced profile.");
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << " Running Phase 1 Test Suite: Document & Dataset Generator\n";
    std::cout << "=========================================================\n";

    test_empty_dataset();
    test_one_document_dataset();
    test_unique_document_ids();
    test_exact_copies_match_source();
    test_modified_copies_derived_from_source();
    test_unrelated_documents_are_singletons();
    test_deterministic_reproducibility();
    test_different_seeds_accepted();
    test_invalid_configurations_rejected();
    test_balanced_vs_skewed_profiles();

    std::cout << "\n=========================================================\n";
    std::cout << " All 10 Phase 1 Test Suites Passed Successfully!\n";
    std::cout << "=========================================================\n";
    return 0;
}
