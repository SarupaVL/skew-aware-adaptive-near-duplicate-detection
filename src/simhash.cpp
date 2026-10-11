#include "simhash.hpp"

#include <array>
#include <sstream>

namespace {

// Standard 64-bit FNV-1a parameters
// Offset basis: 14695981039346656037ULL (0xcbf29ce484222325ULL)
// Prime: 1099511628211ULL (0x100000001b3ULL)
constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime       = 0x100000001b3ULL;

// Portable popcount fallback in case a compiler lacks __builtin_popcountll
inline int count_set_bits(uint64_t n) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(n);
#else
    // Brian Kernighan's algorithm as portable fallback
    int count = 0;
    while (n != 0) {
        n &= (n - 1);
        ++count;
    }
    return count;
#endif
}

} // namespace

std::vector<std::string> tokenize_text(std::string_view text) {
    std::vector<std::string> tokens;
    // Fast space-delimited tokenizer
    size_t start = 0;
    while (start < text.size()) {
        // Skip whitespace
        while (start < text.size() && (text[start] == ' ' || text[start] == '\t' ||
                                       text[start] == '\n' || text[start] == '\r')) {
            ++start;
        }
        if (start >= text.size()) {
            break;
        }
        // Find token end
        size_t end = start;
        while (end < text.size() && text[end] != ' ' && text[end] != '\t' &&
               text[end] != '\n' && text[end] != '\r') {
            ++end;
        }
        tokens.emplace_back(text.substr(start, end - start));
        start = end;
    }
    return tokens;
}

std::vector<std::string> create_shingles(const std::vector<std::string>& tokens,
                                         size_t shingle_size) {
    if (tokens.empty() || shingle_size == 0) {
        return {};
    }

    // Edge case: if token count is less than shingle size, concatenate all
    // available tokens into a single fallback shingle so the document is still fingerprinted
    if (tokens.size() <= shingle_size) {
        std::string fallback;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) fallback += ' ';
            fallback += tokens[i];
        }
        return {std::move(fallback)};
    }

    std::vector<std::string> shingles;
    shingles.reserve(tokens.size() - shingle_size + 1);

    for (size_t i = 0; i <= tokens.size() - shingle_size; ++i) {
        std::string shingle;
        for (size_t j = 0; j < shingle_size; ++j) {
            if (j > 0) shingle += ' ';
            shingle += tokens[i + j];
        }
        shingles.push_back(std::move(shingle));
    }

    return shingles;
}

uint64_t fnv1a_hash64(std::string_view input) noexcept {
    uint64_t hash = kFnvOffsetBasis;
    for (unsigned char byte : input) {
        hash ^= static_cast<uint64_t>(byte);
        hash *= kFnvPrime;
    }
    return hash;
}

uint64_t compute_simhash(std::string_view text, size_t shingle_size) {
    std::vector<std::string> tokens = tokenize_text(text);
    if (tokens.empty()) {
        // Deterministic fingerprint for empty document
        return 0ULL;
    }

    std::vector<std::string> shingles = create_shingles(tokens, shingle_size);
    if (shingles.empty()) {
        return 0ULL;
    }

    // 64 accumulator buckets for weighted bit voting
    // V[i] tracks positive votes (bit=1) minus negative votes (bit=0)
    std::array<int64_t, 64> vote_accumulators{};

    for (const auto& shingle : shingles) {
        uint64_t h = fnv1a_hash64(shingle);
        for (size_t bit = 0; bit < 64; ++bit) {
            // Check if the bit-th bit of h is set
            if ((h >> bit) & 1ULL) {
                vote_accumulators[bit] += 1;
            } else {
                vote_accumulators[bit] -= 1;
            }
        }
    }

    // Construct final 64-bit fingerprint:
    // If accumulator > 0, set bit to 1; otherwise leave as 0
    uint64_t fingerprint = 0ULL;
    for (size_t bit = 0; bit < 64; ++bit) {
        if (vote_accumulators[bit] > 0) {
            fingerprint |= (1ULL << bit);
        }
    }

    return fingerprint;
}

int hamming_distance(uint64_t a, uint64_t b) noexcept {
    uint64_t xor_diff = a ^ b;
    return count_set_bits(xor_diff);
}
