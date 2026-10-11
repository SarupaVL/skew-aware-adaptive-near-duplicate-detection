#ifndef SIMHASH_HPP
#define SIMHASH_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief Default shingle size (k-shingles) for document tokenization.
 *
 * A shingle of size 3 groups 3 consecutive words (word 3-grams) to capture local
 * word-order context.
 */
inline constexpr size_t kDefaultShingleSize = 3;

/**
 * @brief Splits text into clean, whitespace-delimited tokens.
 *
 * Normalization rule:
 * - Leading and trailing whitespace is trimmed.
 * - Words separated by spaces, tabs, or newlines are extracted.
 *
 * @param text The input raw document string.
 * @return std::vector<std::string> Ordered list of word tokens.
 */
std::vector<std::string> tokenize_text(std::string_view text);

/**
 * @brief Constructs k-token shingles (word n-grams) from an ordered list of tokens.
 *
 * For tokens [w0, w1, w2, w3] and shingle_size = 3:
 * Produces: ["w0 w1 w2", "w1 w2 w3"]
 *
 * Edge cases:
 * - If tokens.empty(), returns an empty vector.
 * - If tokens.size() < shingle_size, joins all available tokens as a single fallback shingle.
 *
 * @param tokens The ordered list of word tokens.
 * @param shingle_size The number of consecutive tokens in each shingle (default: 3).
 * @return std::vector<std::string> The constructed shingle strings.
 */
std::vector<std::string> create_shingles(const std::vector<std::string>& tokens,
                                         size_t shingle_size = kDefaultShingleSize);

/**
 * @brief Computes a deterministic 64-bit FNV-1a hash for an arbitrary string.
 *
 * FNV-1a (Fowler-Noll-Vo) provides uniform bit dispersion, zero external
 * dependencies, and guaranteed identical results across all compilers,
 * architectures, and platforms (unlike std::hash which is implementation-dependent).
 *
 * @param input String view to hash.
 * @return uint64_t 64-bit hash value.
 */
uint64_t fnv1a_hash64(std::string_view input) noexcept;

/**
 * @brief Computes a 64-bit SimHash locality-sensitive fingerprint of a document.
 *
 * Algorithm (Moses Charikar, 2002):
 * 1. Tokenize document text into word tokens.
 * 2. Form k-token shingles (default: 3 tokens per shingle).
 * 3. Initialize a 64-dimensional weight accumulator vector V[0..63] with zeros.
 * 4. For each shingle:
 *    a. Compute deterministic 64-bit hash H = fnv1a_hash64(shingle).
 *    b. For bit i in 0..63:
 *       If bit i of H is 1, V[i] += 1 (positive vote).
 *       If bit i of H is 0, V[i] -= 1 (negative vote).
 * 5. Construct final 64-bit fingerprint:
 *    Bit i is set to 1 if V[i] > 0, otherwise 0.
 *
 * Edge cases:
 * - Empty text: Returns 0ULL deterministically.
 * - Text with tokens < shingle_size: Treated using the fallback shingle containing
 *   all available tokens.
 *
 * @param text The input document text.
 * @param shingle_size The shingle size k (default: 3).
 * @return uint64_t 64-bit locality-sensitive fingerprint.
 */
uint64_t compute_simhash(std::string_view text, size_t shingle_size = kDefaultShingleSize);

/**
 * @brief Calculates the bitwise Hamming distance between two 64-bit fingerprints.
 *
 * The Hamming distance is the number of bit positions in which the two binary
 * representations differ (0 to 64).
 *
 * Implementation:
 * Computes bitwise XOR (differing bits become 1) and counts set bits.
 * Uses GCC/Clang built-in __builtin_popcountll or portable fallback.
 *
 * @param a First 64-bit fingerprint.
 * @param b Second 64-bit fingerprint.
 * @return int Number of differing bits in [0, 64].
 */
int hamming_distance(uint64_t a, uint64_t b) noexcept;

#endif // SIMHASH_HPP
