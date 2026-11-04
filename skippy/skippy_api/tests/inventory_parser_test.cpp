//! The GGUF inventory parser reads untrusted bytes with manual offsets and
//! counts. These tests feed it malformed files and require a typed error —
//! never a crash, a hang, or a throw through the expected boundary. The
//! ASan suite runs them so off-by-one reads, uncapped allocations and
//! unbounded recursion are detected, not assumed away.

#include "llama/model_inventory.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <vector>

// The F32 row_size_bytes mirror this suite used to carry now lives in
// fuzz/native_free_row_size.cpp and is linked here from the CMake target —
// one stub shared by the tests and the fuzzers, not two.

namespace {

using skippy::probe::read_inventory;

/// Writes bytes to a temp file and returns its path; the caller removes it.
std::string write_temp(const std::vector<std::uint8_t>& bytes, const std::string& tag) {
    const std::string path = "/tmp/skippy-inv-" + tag + ".gguf";
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return path;
}

void put32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

void put64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

/// Mirrors the parser's data-start arithmetic (records end rounded up to the
/// default 32-byte alignment) so a fixture's declared payload fits the file.
void pad_for_payload(std::vector<std::uint8_t>& bytes, std::uint64_t stored_length,
                     std::uint64_t offset = 0) {
    const std::uint64_t data_start = (bytes.size() + 31U) / 32U * 32U;
    const std::uint64_t needed = data_start + offset + stored_length;
    if (needed > bytes.size()) bytes.resize(static_cast<std::size_t>(needed), 0);
}

skippy::probe::TensorInfo tensor(const char* name) {
    skippy::probe::TensorInfo info;
    info.name = name;
    return info;
}

/// A v3 header with the given tensor and metadata counts.
std::vector<std::uint8_t> header(std::uint64_t tensor_count, std::uint64_t metadata_count) {
    std::vector<std::uint8_t> bytes;
    bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
    put32(bytes, 3); // version
    put64(bytes, tensor_count);
    put64(bytes, metadata_count);
    return bytes;
}

std::vector<std::uint8_t> empty_header() { return header(0, 0); }

TEST(InventoryParser, AcceptsAnEmptyButWellFormedHeader) {
    const std::string path = write_temp(empty_header(), "empty");
    const auto inventory = read_inventory(path);
    std::remove(path.c_str());
    ASSERT_TRUE(inventory.has_value());
    EXPECT_TRUE(inventory->empty());
}

TEST(InventoryParser, ArchitectureRejectsWrongMagicBeforeMetadata) {
    auto bytes = empty_header();
    bytes[0] = 'X';
    const auto path = write_temp(bytes, "architecture-wrong-magic");
    const auto architecture = skippy::probe::model_architecture(path);
    std::remove(path.c_str());
    ASSERT_FALSE(architecture.has_value());
    EXPECT_EQ(architecture.error(), "not a GGUF file");
}

TEST(InventoryParser, ArchitectureRejectsUnsupportedVersionsBeforeMetadata) {
    for (const std::uint32_t version : {0U, 1U, 4U, std::numeric_limits<std::uint32_t>::max()}) {
        std::vector<std::uint8_t> bytes{'G', 'G', 'U', 'F'};
        put32(bytes, version);
        put64(bytes, 0);
        put64(bytes, 0);
        const auto path = write_temp(bytes, "architecture-version-" + std::to_string(version));
        const auto architecture = skippy::probe::model_architecture(path);
        std::remove(path.c_str());
        ASSERT_FALSE(architecture.has_value()) << version;
        EXPECT_EQ(architecture.error(), "unsupported GGUF version");
    }
}

TEST(InventoryParser, ArchitectureAcceptsSupportedVersionsWithoutMetadata) {
    for (const std::uint32_t version : {2U, 3U}) {
        std::vector<std::uint8_t> bytes{'G', 'G', 'U', 'F'};
        put32(bytes, version);
        put64(bytes, 0);
        put64(bytes, 0);
        const auto path = write_temp(bytes, "architecture-version-" + std::to_string(version));
        const auto architecture = skippy::probe::model_architecture(path);
        std::remove(path.c_str());
        ASSERT_TRUE(architecture.has_value()) << version;
        EXPECT_TRUE(architecture->empty());
    }
}

TEST(InventoryParser, RejectsEveryTruncationOfAValidFile) {
    const std::vector<std::uint8_t> bytes = empty_header();
    for (std::size_t cut = 0; cut < bytes.size(); ++cut) {
        const std::string tag = "cut-" + std::to_string(cut);
        const std::string path =
            write_temp(std::vector<std::uint8_t>(bytes.begin(), bytes.begin() + cut), tag);
        const auto inventory = read_inventory(path);
        std::remove(path.c_str());
        EXPECT_FALSE(inventory.has_value()) << "truncation at " << cut << " parsed";
    }
}

TEST(InventoryParser, RejectsAnImplausibleStringLengthWithoutAllocating) {
    std::vector<std::uint8_t> bytes = header(0, 1);
    put64(bytes, 1ULL << 40); // key length far past the cap
    bytes.push_back('k');
    const std::string path = write_temp(bytes, "huge-string");
    const auto inventory = read_inventory(path);
    std::remove(path.c_str());
    ASSERT_FALSE(inventory.has_value());
}

TEST(InventoryParser, RejectsAnUnknownMetadataType) {
    std::vector<std::uint8_t> bytes = header(0, 1);
    put64(bytes, 1); // key length
    bytes.push_back('k');
    put32(bytes, 99); // no such GGUF type
    const std::string path = write_temp(bytes, "bad-type");
    const auto inventory = read_inventory(path);
    std::remove(path.c_str());
    ASSERT_FALSE(inventory.has_value());
}

/// Arrays of arrays are legal GGUF; unbounded recursion on hostile nesting
/// is a stack overflow. Depth is capped and reported.
TEST(InventoryParser, RejectsDeeplyNestedMetadataArrays) {
    constexpr std::uint32_t nesting = 64; // far past kMaxArrayDepth
    std::vector<std::uint8_t> bytes = header(0, 1);
    put64(bytes, 1); // key length
    bytes.push_back('k');
    for (std::uint32_t depth = 0; depth < nesting; ++depth) {
        put32(bytes, 9); // array
        put32(bytes, 9); // of arrays
        put64(bytes, 1); // one element
    }
    put32(bytes, 0); // innermost: one u8 element
    put64(bytes, 1);
    bytes.push_back(7);
    const std::string path = write_temp(bytes, "deep-array");
    const auto inventory = read_inventory(path);
    std::remove(path.c_str());
    ASSERT_FALSE(inventory.has_value());
}

TEST(InventoryParser, RejectsAnImplausibleArrayLength) {
    std::vector<std::uint8_t> bytes = header(0, 1);
    put64(bytes, 1); // key length
    bytes.push_back('k');
    put32(bytes, 9);          // array
    put32(bytes, 0);          // of u8
    put64(bytes, 1ULL << 40); // element count past the cap
    const std::string path = write_temp(bytes, "huge-array");
    const auto inventory = read_inventory(path);
    std::remove(path.c_str());
    ASSERT_FALSE(inventory.has_value());
}

TEST(InventoryParser, RejectsAnImplausibleTensorCountBeforeReserving) {
    std::vector<std::uint8_t> bytes;
    bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
    put32(bytes, 3);
    put64(bytes, 1ULL << 40); // tensor count past the cap
    put64(bytes, 0);          // metadata count
    const std::string path = write_temp(bytes, "huge-tensors");
    const auto inventory = read_inventory(path);
    std::remove(path.c_str());
    ASSERT_FALSE(inventory.has_value());
}

TEST(InventoryParser, RejectsAnImplausibleMetadataCount) {
    std::vector<std::uint8_t> bytes;
    bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
    put32(bytes, 3);
    put64(bytes, 0);          // tensor count
    put64(bytes, 1ULL << 40); // metadata count past the cap
    const std::string path = write_temp(bytes, "huge-metadata");
    const auto inventory = read_inventory(path);
    std::remove(path.c_str());
    ASSERT_FALSE(inventory.has_value());
}

TEST(InventoryParser, RejectsTensorsOfRankZeroAndAboveFour) {
    for (const std::uint32_t rank : {0U, 5U}) {
        std::vector<std::uint8_t> bytes = header(1, 0);
        put64(bytes, 2); // name length
        bytes.push_back('t');
        bytes.push_back('0');
        put32(bytes, rank);
        const std::string path = write_temp(bytes, "rank-" + std::to_string(rank));
        const auto inventory = read_inventory(path);
        std::remove(path.c_str());
        EXPECT_FALSE(inventory.has_value()) << "rank " << rank << " parsed";
    }
}

TEST(InventoryParser, ParsesATensorWithAllFourDimensions) {
    std::vector<std::uint8_t> bytes = header(1, 0);
    put64(bytes, 2); // name length
    bytes.push_back('t');
    bytes.push_back('0');
    put32(bytes, 4); // rank
    for (int dim = 0; dim < 4; ++dim)
        put64(bytes, 2);
    put32(bytes, 0);                           // ggml type F32
    put64(bytes, 0);                           // offset
    pad_for_payload(bytes, 4 * 2 * 2 * 2 * 2); // mocked F32 payload fits the file
    const std::string path = write_temp(bytes, "rank4");
    const auto inventory = read_inventory(path);
    std::remove(path.c_str());
    ASSERT_TRUE(inventory.has_value());
    ASSERT_EQ(inventory->size(), 1U);
    EXPECT_EQ((*inventory)[0].name, "t0");
}

TEST(InventoryParser, RejectsDimensionsWhoseProductWrapsToZero) {
    // 2^32 * 2^32 * 2^16 * 2^16 = 2^96 wraps a 64-bit element product to
    // zero. Upstream's gguf.cpp accepts this today because its
    // representability guard multiplies first and checks second
    // (CVE-2026-33298 family, see UPSTREAM.md); our parser must reject it.
    std::vector<std::uint8_t> bytes = header(1, 0);
    put64(bytes, 2); // name length
    bytes.push_back('t');
    bytes.push_back('0');
    put32(bytes, 4); // rank
    for (const std::uint64_t dim : {1ULL << 32, 1ULL << 32, 1ULL << 16, 1ULL << 16}) {
        put64(bytes, dim);
    }
    put32(bytes, 0); // ggml type F32
    put64(bytes, 0); // offset
    const std::string path = write_temp(bytes, "wrap-to-zero");
    const auto inventory = read_inventory(path);
    std::remove(path.c_str());
    ASSERT_FALSE(inventory.has_value());
}

TEST(InventoryParser, RejectsADimensionPastInt64) {
    // A stored extent must survive the int64_t cast; 2^63 would alias to a
    // negative dimension.
    std::vector<std::uint8_t> bytes = header(1, 0);
    put64(bytes, 2); // name length
    bytes.push_back('t');
    bytes.push_back('0');
    put32(bytes, 1); // rank
    put64(bytes, 1ULL << 63);
    put32(bytes, 0); // ggml type F32
    put64(bytes, 0); // offset
    const std::string path = write_temp(bytes, "dim-past-int64");
    const auto inventory = read_inventory(path);
    std::remove(path.c_str());
    ASSERT_FALSE(inventory.has_value());
}

TEST(InventoryParser, RejectsZeroAndNonPowerOfTwoAlignments) {
    // Zero divides by zero at the data-start round-up; three misaligns every
    // tensor offset the native loader computes.
    for (const std::uint32_t alignment : {0U, 3U}) {
        std::vector<std::uint8_t> bytes = header(0, 1);
        const std::string key = "general.alignment";
        put64(bytes, key.size());
        bytes.insert(bytes.end(), key.begin(), key.end());
        put32(bytes, 4); // GGUF u32
        put32(bytes, alignment);
        const std::string path = write_temp(bytes, "align-" + std::to_string(alignment));
        const auto inventory = read_inventory(path);
        std::remove(path.c_str());
        ASSERT_FALSE(inventory.has_value()) << "alignment " << alignment << " accepted";
        EXPECT_EQ(inventory.error(), "invalid general.alignment");
    }
}

TEST(InventoryParser, RejectsATensorWhosePayloadLiesOutsideTheFile) {
    // Three branches: the aligned data start itself lies past EOF; the
    // declared offset runs past the data section; the stored length does.
    struct Case {
        std::uint64_t offset;
        bool data_start_only;
        std::string tag;
    };
    const std::array<Case, 3> cases = {Case{0, false, "no-payload"}, Case{64, true, "offset-past"},
                                       Case{0, true, "length-past"}};
    for (const Case& entry : cases) {
        std::vector<std::uint8_t> bytes = header(1, 0);
        put64(bytes, 2); // name length
        bytes.push_back('t');
        bytes.push_back('0');
        put32(bytes, 2); // rank
        put64(bytes, 2);
        put64(bytes, 2);
        put32(bytes, 0); // ggml type F32
        put64(bytes, entry.offset);
        if (entry.data_start_only) pad_for_payload(bytes, 0);
        const std::string path = write_temp(bytes, "payload-" + entry.tag);
        const auto inventory = read_inventory(path);
        std::remove(path.c_str());
        ASSERT_FALSE(inventory.has_value()) << entry.tag << " accepted";
        EXPECT_EQ(inventory.error(), "tensor data outside file");
    }
}

TEST(InventoryParser, LayerCountTakesTheHighestBlockIndex) {
    const std::vector<skippy::probe::TensorInfo> tensors = {
        tensor("token_embd.weight"),      tensor("blk.0.attn_q.weight"),
        tensor("blk.12.mlp_down.weight"), tensor("blk.3.output_norm.weight"),
        tensor("root.blk.7.norm.weight"), // counts: after a separator
    };
    EXPECT_EQ(skippy::probe::layer_count_from(tensors), 13);
}

TEST(InventoryParser, LayerCountSkipsMalformedNames) {
    const std::vector<skippy::probe::TensorInfo> tensors = {
        tensor("blk..weight"),                     // empty index
        tensor("blk.abc.weight"),                  // non-numeric
        tensor("blk.-2.weight"),                   // negative
        tensor("blk.99999999999999999999.weight"), // past int64
        tensor("xxblk.9.weight"),                  // no separator before blk.
        tensor("blk.9223372036854775807.weight"),  // INT64_MAX: +1 would overflow
    };
    EXPECT_EQ(skippy::probe::layer_count_from(tensors), 0);
}

/// Parses a GGUF whose only tensors are `names` (rank-1, one-element F32,
/// all at data offset zero — only the names matter here). The parser sorts
/// by name, so the fixture may list them in any order.
std::expected<std::vector<skippy::probe::TensorInfo>, std::string>
parse_tensors(std::string_view tag, std::initializer_list<std::string_view> names) {
    std::vector<std::uint8_t> bytes = header(names.size(), 0);
    for (const std::string_view name : names) {
        put64(bytes, name.size());
        bytes.insert(bytes.end(), name.begin(), name.end());
        put32(bytes, 1); // rank
        put64(bytes, 1); // extent
        put32(bytes, 0); // ggml type F32
        put64(bytes, 0); // offset
    }
    pad_for_payload(bytes, 4); // every fixture tensor is one F32 at offset 0
    const std::string path = write_temp(bytes, std::string{tag});
    auto inventory = read_inventory(path);
    std::remove(path.c_str());
    return inventory;
}

TEST(InventoryParser, ClassifierHeadPresenceIsTheHeadTensorNotTheFamily) {
    // The workload gate reads `llama_model_n_cls_out` only when the inventory
    // carries the head tensor the native loader names, `cls.output.weight`.
    // Presence is the tensor, not the family: a decoder family that loads it
    // still classifies, and an unknown architecture without it must not read
    // the native default of one output as a head that does not exist.
    // `cls` and `cls_norm.weight` bracket the head's name on either side
    // ('.' sorts before '_'), so the search cannot pass on a prefix.
    const auto with_head = parse_tensors("head-yes", {"token_embd.weight", "cls.output.weight"});
    ASSERT_TRUE(with_head.has_value());
    EXPECT_TRUE(skippy::probe::has_classifier_head(*with_head));

    const auto without_head =
        parse_tensors("head-no", {"cls", "cls_norm.weight", "token_embd.weight"});
    ASSERT_TRUE(without_head.has_value());
    EXPECT_FALSE(skippy::probe::has_classifier_head(*without_head));

    EXPECT_FALSE(skippy::probe::has_classifier_head(std::vector<skippy::probe::TensorInfo>{}));
}

TEST(InventoryParser, CheckedTensorBytesMultipliesOnlyTheRemainingExtents) {
    const std::array<std::int64_t, 4> dimensions{999, 4, 5, 7};
    // The first dimension already sized the row; only 4 * 5 counts at rank 3.
    EXPECT_EQ(skippy::probe::checked_tensor_bytes(6, dimensions, 3), std::uint64_t{120});
    EXPECT_EQ(skippy::probe::checked_tensor_bytes(6, dimensions, 1), std::uint64_t{6});
}

TEST(InventoryParser, CheckedTensorBytesRejectsWrappingAndNegativeExtents) {
    const std::array<std::int64_t, 4> wraps{0, 16, 0, 0};
    EXPECT_FALSE(skippy::probe::checked_tensor_bytes(1ULL << 60, wraps, 2).has_value());
    const std::array<std::int64_t, 4> negative{0, -1, 0, 0};
    EXPECT_FALSE(skippy::probe::checked_tensor_bytes(6, negative, 2).has_value());
    const std::array<std::int64_t, 4> zero{0, 0, 0, 0};
    EXPECT_EQ(skippy::probe::checked_tensor_bytes(6, zero, 2), std::uint64_t{0});
}

TEST(InventoryParser, RejectsATensorWhoseByteSizeWouldWrap) {
    // The element product (2^62 * 1) passes the read_extents checks; the
    // byte total (4 * 2^62) does not. The parser must surface that as a
    // typed error, never a wrapped stored_length.
    std::vector<std::uint8_t> bytes = header(1, 0);
    put64(bytes, 2); // name length
    bytes.push_back('t');
    bytes.push_back('0');
    put32(bytes, 2); // rank
    put64(bytes, 1ULL << 62);
    put64(bytes, 1);
    put32(bytes, 0); // ggml type F32
    put64(bytes, 0); // offset
    const std::string path = write_temp(bytes, "byte-wrap");
    const auto inventory = read_inventory(path);
    std::remove(path.c_str());
    ASSERT_FALSE(inventory.has_value());
    EXPECT_EQ(inventory.error(), "invalid tensor type or size");
}

TEST(InventoryParser, Sha256MatchesReferenceDigests) {
    // Package identity hashes the whole shard through the same mapping the
    // header parser uses. Digests are pinned against an independent hashlib
    // run, covering both SHA-256 padding shapes (56- and 63-byte tails) and
    // multi-megabyte files that dwarf every other fixture in this suite.
    const auto pattern = [](std::size_t size) {
        std::vector<std::uint8_t> bytes(size);
        for (std::size_t index = 0; index < size; ++index) {
            bytes[index] = static_cast<std::uint8_t>(index % 251U);
        }
        return bytes;
    };
    struct Case {
        std::vector<std::uint8_t> bytes;
        const char* digest;
    };
    const std::array cases = {
        Case{std::vector<std::uint8_t>{},
             "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        Case{{'a', 'b', 'c'},
             "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        Case{pattern(56),
             "sha256:da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"},
        Case{pattern(1U << 20U),
             "sha256:631b84027d6b9e52b539c4e8373622d23032dfadc64d60af87339c9037e4f769"},
        Case{pattern((1U << 20U) + 63U),
             "sha256:f9605ee85813a4764ead84bf52043de51690ab7f08835b190fb3447f28f3a7aa"},
        Case{pattern(2U * (1U << 20U) + 55U),
             "sha256:03ed671717bf4fb9fe4e709e566c5ebd5531c7303864904539d384fc7729a4bd"},
    };
    for (const Case& entry : cases) {
        const std::string path = write_temp(entry.bytes, "sha256-stream");
        const auto digest = skippy::probe::sha256_of(path);
        ASSERT_TRUE(digest.has_value()) << digest.error();
        EXPECT_EQ(*digest, entry.digest) << "size " << entry.bytes.size();
        std::remove(path.c_str());
    }
}

} // namespace
