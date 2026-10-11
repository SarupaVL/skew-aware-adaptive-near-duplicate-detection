#ifndef DOCUMENT_HPP
#define DOCUMENT_HPP

#include <cstdint>
#include <string>

/**
 * @brief Categorizes the relationship of a document to others in the corpus.
 *
 * - Original:     The primary seed document for a duplicate cluster.
 * - ExactCopy:    A verbatim duplicate of an existing Original document.
 * - ModifiedCopy: A variant of an Original document where a controlled subset
 *                 of tokens has been modified/substituted.
 * - Unrelated:    A standalone document that does not belong to any duplicate
 *                 cluster (singleton group).
 */
enum class DocumentKind {
    Original,
    ExactCopy,
    ModifiedCopy,
    Unrelated
};

/**
 * @brief Returns a readable string representation of DocumentKind.
 */
inline const char* to_string(DocumentKind kind) {
    switch (kind) {
        case DocumentKind::Original:     return "Original";
        case DocumentKind::ExactCopy:    return "ExactCopy";
        case DocumentKind::ModifiedCopy: return "ModifiedCopy";
        case DocumentKind::Unrelated:    return "Unrelated";
        default:                         return "Unknown";
    }
}

/**
 * @brief Core document representation in the dataset.
 *
 * GROUND-TRUTH LABELS:
 * - Every document has a unique `id`.
 * - Documents belonging to the same duplicate cluster share the exact same `group_id`.
 *   Specifically: an Original document and all of its ExactCopy and ModifiedCopy
 *   derivatives share the Original's `group_id`.
 * - Unrelated documents each receive their own unique `group_id` (a cluster of size 1).
 *
 * This allows exact evaluation of true positives, false positives, precision,
 * and recall in subsequent phases without needing any detection algorithm.
 */
struct Document {
    uint64_t id{0};                  ///< Unique, stable 0-indexed document ID.
    uint64_t group_id{0};            ///< Ground-truth duplicate cluster ID.
    DocumentKind kind{DocumentKind::Original}; ///< Category of this document.
    int64_t source_id{-1};           ///< ID of the parent original document (-1 if Original/Unrelated).
    std::string text;                ///< Space-separated token text of the document.
};

#endif // DOCUMENT_HPP
