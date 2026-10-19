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
#include <string>
#include <vector>

// The parser only needs the byte-size seam; ggml's real implementation is
// linked into the llama side and exercised by the probes. The mocked suite
// supplies its own so this target stays native-free.
namespace skippy::probe {
std::uint64_t row_size_bytes(std::uint32_t, const std::array<std::int64_t, 4>& dimensions,
                             std::uint32_t rank) {
    std::uint64_t bytes = 4; // F32 row element
    for (std::uint32_t dim = 0; dim < rank; ++dim) {
        bytes *= static_cast<std::uint64_t>(dimensions[dim]);
    }
    return bytes;
}
} // namespace skippy::probe

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
    put32(bytes, 0); // ggml type F32
    put64(bytes, 0); // offset
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

} // namespace
