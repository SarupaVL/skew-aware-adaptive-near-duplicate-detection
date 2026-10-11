#include "simhash.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

void check(bool condition, const std::string& test_name, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << test_name << ": " << message << "\n";
        std::exit(1);
    }
    std::cout << "[PASS] " << test_name << "\n";
}

void test_hamming_distance_identical() {
    uint64_t a = 0xAAAAAAAAAAAAAAAAULL;
    int dist = hamming_distance(a, a);
    check(dist == 0, "test_hamming_distance_identical", "Identical fingerprints must have distance 0.");

    dist = hamming_distance(0ULL, 0ULL);
    check(dist == 0, "test_hamming_distance_zeros", "Distance between 0 and 0 must be 0.");

    dist = hamming_distance(~0ULL, ~0ULL);
    check(dist == 0, "test_hamming_distance_all_ones", "Distance between ~0 and ~0 must be 0.");
}

void test_hamming_distance_one_bit() {
    for (int bit = 0; bit < 64; ++bit) {
        uint64_t a = 0ULL;
        uint64_t b = (1ULL << bit);
        int dist = hamming_distance(a, b);
        check(dist == 1, "test_hamming_distance_one_bit", "Single bit flip must result in distance 1.");
    }
}

void test_hamming_distance_all_64_bits() {
    uint64_t a = 0ULL;
    uint64_t b = ~0ULL; // 0xFFFFFFFFFFFFFFFFULL
    int dist = hamming_distance(a, b);
    check(dist == 64, "test_hamming_distance_all_64_bits", "Inverted fingerprints must have distance 64.");
}

void test_hamming_distance_known_cases() {
    // 0b1010 vs 0b0101 -> 4 bits differ
    check(hamming_distance(0b1010ULL, 0b0101ULL) == 4,
          "test_hamming_known_1", "Expected 4 bits.");
    // 0x0F0F vs 0xF0F0 -> 16 bits differ (0x0F0F ^ 0xF0F0 = 0xFFFF)
    check(hamming_distance(0x0F0FULL, 0xF0F0ULL) == 16,
          "test_hamming_known_2", "Expected 16 bits.");
    // Alternating patterns: 0x5555555555555555 vs 0xAAAAAAAAAAAAAAAA -> all 64 bits differ
    check(hamming_distance(0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL) == 64,
          "test_hamming_known_3", "Expected 64 bits.");
}

void test_simhash_reproducibility() {
    std::string text = "high performance computing multicore parallel loop scheduling openmp";
    uint64_t h1 = compute_simhash(text);
    uint64_t h2 = compute_simhash(text);
    check(h1 == h2, "test_simhash_reproducibility", "Repeated fingerprinting must produce identical results.");
    check(h1 != 0ULL, "test_simhash_nonzero", "Non-empty text must produce non-zero fingerprint.");
}

void test_simhash_identical_non_empty_texts() {
    std::string text1 = "algorithm cache locality thread synchronization";
    std::string text2 = "algorithm cache locality thread synchronization";
    uint64_t h1 = compute_simhash(text1);
    uint64_t h2 = compute_simhash(text2);
    check(h1 == h2, "test_simhash_identical_texts", "Identical texts must produce identical SimHash.");
}

void test_simhash_empty_text() {
    uint64_t h1 = compute_simhash("");
    uint64_t h2 = compute_simhash("    \t\n   ");
    check(h1 == 0ULL, "test_simhash_empty_text", "Empty text must produce deterministic 0ULL.");
    check(h2 == 0ULL, "test_simhash_whitespace_only", "Whitespace-only text must produce deterministic 0ULL.");
    check(h1 == h2, "test_simhash_empty_equality", "Empty and whitespace-only must match.");
}

void test_simhash_short_text() {
    // 1. One-token document (tokens < shingle_size)
    std::string one_token = "algorithm";
    auto tokens_1 = tokenize_text(one_token);
    check(tokens_1.size() == 1, "test_tokens_1_count", "1 token expected.");
    auto shingles_1 = create_shingles(tokens_1, 3);
    check(shingles_1.size() == 1 && shingles_1[0] == "algorithm",
          "test_shingles_1_fallback", "Fallback shingle must equal single token.");

    uint64_t h1 = compute_simhash(one_token, 3);
    check(h1 != 0ULL, "test_simhash_short_text_1", "1-word text must produce valid non-zero fingerprint.");
    check(h1 == compute_simhash(one_token, 3), "test_simhash_short_deterministic_1", "Must be deterministic.");
    // For a single shingle, SimHash bit i equals bit i of FNV-1a hash
    check(h1 == fnv1a_hash64("algorithm"), "test_simhash_1_token_equals_hash",
          "Single-token SimHash must match raw shingle hash directly.");

    // 2. Two-token document (tokens < shingle_size)
    std::string two_tokens = "parallel algorithm";
    auto tokens_2 = tokenize_text(two_tokens);
    check(tokens_2.size() == 2, "test_tokens_2_count", "2 tokens expected.");
    auto shingles_2 = create_shingles(tokens_2, 3);
    check(shingles_2.size() == 1 && shingles_2[0] == "parallel algorithm",
          "test_shingles_2_fallback", "Fallback shingle must join available tokens.");

    uint64_t h2 = compute_simhash(two_tokens, 3);
    check(h2 != 0ULL, "test_simhash_short_text_2", "2-word text must produce valid non-zero fingerprint.");
    check(h2 == compute_simhash(two_tokens, 3), "test_simhash_short_deterministic_2", "Must be deterministic.");
    check(h2 == fnv1a_hash64("parallel algorithm"), "test_simhash_2_token_equals_hash",
          "Two-token fallback SimHash must match raw shingle hash.");

    // 3. Exactly three-token document (tokens == shingle_size)
    std::string three_tokens = "parallel algorithm design";
    auto tokens_3 = tokenize_text(three_tokens);
    auto shingles_3 = create_shingles(tokens_3, 3);
    check(shingles_3.size() == 1 && shingles_3[0] == "parallel algorithm design",
          "test_shingles_3_exact", "Exactly 3 tokens with k=3 must yield exactly 1 standard shingle.");
    uint64_t h3 = compute_simhash(three_tokens, 3);
    check(h3 != 0ULL && h3 == fnv1a_hash64("parallel algorithm design"),
          "test_simhash_3_tokens_exact", "3-token shingle SimHash must match shingle hash.");
}

void test_simhash_exact_copy() {
    std::string original = "multicore cpu near duplicate detection simhash openmp scheduling workload skew";
    std::string copy = original; // Exact verbatim copy

    uint64_t orig_hash = compute_simhash(original);
    uint64_t copy_hash = compute_simhash(copy);

    check(orig_hash == copy_hash, "test_simhash_exact_copy", "Exact copy must produce identical fingerprint.");
    check(hamming_distance(orig_hash, copy_hash) == 0, "test_simhash_exact_copy_distance", "Hamming distance must be 0.");
}

void test_tokenization_and_shingling_consistency() {
    std::string text = "apple banana cherry date elderberry";
    auto tokens = tokenize_text(text);
    check(tokens.size() == 5, "test_tokenize_count", "Expected 5 tokens.");
    check(tokens[0] == "apple" && tokens[4] == "elderberry", "test_tokenize_tokens", "Token contents mismatch.");

    // Test shingle size = 3
    auto shingles = create_shingles(tokens, 3);
    check(shingles.size() == 3, "test_shingle_count", "Expected 3 shingles (5 - 3 + 1).");
    check(shingles[0] == "apple banana cherry", "test_shingle_0", "First shingle mismatch.");
    check(shingles[1] == "banana cherry date", "test_shingle_1", "Second shingle mismatch.");
    check(shingles[2] == "cherry date elderberry", "test_shingle_2", "Third shingle mismatch.");

    // Multiple whitespace and tab normalization
    std::string messy_text = "  apple   \t\n  banana  \t cherry   date  elderberry   ";
    auto messy_tokens = tokenize_text(messy_text);
    check(tokens == messy_tokens, "test_whitespace_normalization", "Whitespace normalization must match tokens.");
}

void test_fnv1a_hash_determinism() {
    uint64_t hash1 = fnv1a_hash64("test shingle string");
    uint64_t hash2 = fnv1a_hash64("test shingle string");
    check(hash1 == hash2, "test_fnv1a_determinism", "FNV-1a must be strictly deterministic.");

    uint64_t hash3 = fnv1a_hash64("different shingle string");
    check(hash1 != hash3, "test_fnv1a_different", "Different inputs should produce different hashes.");
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << " Running Phase 2 Test Suite: SimHash & Hamming Distance\n";
    std::cout << "=========================================================\n";

    test_hamming_distance_identical();
    test_hamming_distance_one_bit();
    test_hamming_distance_all_64_bits();
    test_hamming_distance_known_cases();
    test_simhash_reproducibility();
    test_simhash_identical_non_empty_texts();
    test_simhash_empty_text();
    test_simhash_short_text();
    test_simhash_exact_copy();
    test_tokenization_and_shingling_consistency();
    test_fnv1a_hash_determinism();

    std::cout << "\n=========================================================\n";
    std::cout << " All Phase 2 Test Suites Passed Successfully!\n";
    std::cout << "=========================================================\n";
    return 0;
}
