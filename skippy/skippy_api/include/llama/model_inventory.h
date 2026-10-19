#ifndef SKIPPY_API_TOOLS_GGUF_INVENTORY_H
#define SKIPPY_API_TOOLS_GGUF_INVENTORY_H

/// Reads a GGUF's tensor inventory straight from the header.
///
/// The planner is configured from an exact tensor closure, so anything that
/// plans a real model needs this map first. It lives beside the llama backend
/// because the byte size of a tensor is a ggml type property. This header
/// stays free of native includes: row_size_bytes is declared here and
/// defined once in src/llama/model_inventory.cpp against ggml, so the
/// mocked tests can link a stub and stay native-free.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace skippy::probe {

/// Identity of a file version: a plan is only reusable while these hold.
inline std::string file_stamp(const std::string& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) return path;
    const auto modified = std::filesystem::last_write_time(path, error);
    const auto bytes = std::to_string(static_cast<std::uintmax_t>(size));
    if (error) return path + ":" + bytes;
    const auto ticks =
        std::to_string(static_cast<std::int64_t>(modified.time_since_epoch().count()));
    return path + ":" + bytes + ":" + ticks;
}

/// Metadata strings are model names; anything larger is a malformed file.
constexpr std::uint64_t kMaxMetadataString = 1048576;

struct TensorInfo {
    std::string name;
    std::uint32_t type = 0;
    std::uint32_t dimension_count = 0;
    std::array<std::int64_t, 4> dimensions{};
    std::uint64_t offset = 0;
    std::uint64_t stored_length = 0;
};

/// Owns a read-only mapping of the whole file. Only the pages the parser
/// touches fault in, so mapping a multi-gigabyte shard costs the header.
class Mapping {
public:
    Mapping() = default;
    explicit Mapping(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return;
        struct stat status{};
        if (::fstat(fd, &status) == 0 && status.st_size > 0) {
            size_ = static_cast<std::size_t>(status.st_size);
            data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
            if (data_ == MAP_FAILED) data_ = nullptr;
        }
        ::close(fd);
    }
    ~Mapping() {
        if (data_ != nullptr) ::munmap(data_, size_);
    }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping(Mapping&& other) noexcept { adopt(other); }
    Mapping& operator=(Mapping&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) ::munmap(data_, size_);
            adopt(other);
        }
        return *this;
    }
    [[nodiscard]] bool valid() const { return data_ != nullptr; }
    [[nodiscard]] const void* data() const { return data_; }
    [[nodiscard]] std::size_t size() const { return size_; }

private:
    void adopt(Mapping& other) {
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, std::size_t{0});
    }
    void* data_ = nullptr;
    std::size_t size_ = 0;
};

/// A bounds-checked cursor over the mapped bytes. Every operation is one
/// comparison and a pointer bump: no stream machinery, no per-field
/// syscalls, no allocation. Instruments blamed the per-element seekg (one
/// flockfile/fseeko/ftello each) for 62% of parse time; this is the fix.
class Reader {
public:
    explicit Reader(const Mapping& mapping)
        : base_(static_cast<const std::uint8_t*>(mapping.data())), pos_(base_),
          end_(base_ + mapping.size()) {}

    [[nodiscard]] std::uint64_t tell() const { return static_cast<std::uint64_t>(pos_ - base_); }
    [[nodiscard]] std::uint64_t remaining() const {
        return static_cast<std::uint64_t>(end_ - pos_);
    }

    template <typename T> std::expected<void, std::string> read(T& value) {
        if (remaining() < sizeof(T)) return std::unexpected("truncated header");
        std::memcpy(&value, pos_, sizeof(T));
        pos_ += sizeof(T);
        return {};
    }

    std::expected<void, std::string> skip(std::uint64_t count) {
        if (remaining() < count) return std::unexpected("truncated header");
        pos_ += count;
        return {};
    }

    /// Views the next length-prefixed string without copying it. The view
    /// borrows the mapping and must not outlive it.
    std::expected<std::string_view, std::string> read_string_view() {
        std::uint64_t length = 0;
        if (auto ok = read(length); !ok) return std::unexpected(ok.error());
        if (length > kMaxMetadataString) return std::unexpected("implausible string length");
        if (remaining() < length) return std::unexpected("truncated string");
        const auto* begin = reinterpret_cast<const char*>(pos_);
        pos_ += length;
        return std::string_view{begin, static_cast<std::size_t>(length)};
    }

    /// Advances past one length-prefixed string without viewing its bytes.
    std::expected<void, std::string> skip_string() {
        std::uint64_t length = 0;
        if (auto ok = read(length); !ok) return std::unexpected(ok.error());
        if (length > kMaxMetadataString) return std::unexpected("implausible string length");
        return skip(length);
    }

private:
    const std::uint8_t* base_;
    const std::uint8_t* pos_;
    const std::uint8_t* end_;
};

/// Counts come from the file: cap them so a malformed GGUF is a typed
/// error, not a length_error through the expected boundary or a
/// stack-overflow via recursive arrays. Real GGUFs carry at most tens of
/// thousands of entries; these bounds are far past that.
inline constexpr std::uint64_t kMaxInventoryEntries = 1ULL * 1024U * 1024U;
inline constexpr std::uint32_t kMaxArrayDepth = 8;

/// Bytes a tensor of `type` occupies given its extents. Declared here,
/// defined against ggml in src/llama/model_inventory.cpp so this header
/// stays includable from native-free test targets.
[[nodiscard]] std::uint64_t row_size_bytes(std::uint32_t type,
                                           const std::array<std::int64_t, 4>& dimensions,
                                           std::uint32_t rank);

/// Fixed widths of the scalar GGUF value types; anything else is a string,
/// an array, or unknown.
inline std::optional<std::uint64_t> scalar_width(std::uint32_t type) {
    switch (type) {
    case 0:                           // u8
    case 1:                           // i8
    case 7: return std::uint64_t{1};  // bool
    case 2:                           // u16
    case 3: return std::uint64_t{2};  // i16
    case 4:                           // u32
    case 5:                           // i32
    case 6: return std::uint64_t{4};  // f32
    case 10:                          // u64
    case 11:                          // i64
    case 12: return std::uint64_t{8}; // f64
    default: return std::nullopt;
    }
}

inline std::expected<void, std::string> skip_value(Reader& reader, std::uint32_t type,
                                                   std::uint32_t depth);

/// Skips one array: scalar arrays collapse to a single bulk skip, string
/// arrays walk lengths without reading bytes, nested arrays recurse inside
/// the depth cap.
inline std::expected<void, std::string> skip_array(Reader& reader, std::uint32_t depth) {
    if (depth >= kMaxArrayDepth) return std::unexpected("metadata array nesting too deep");
    std::uint32_t element_type = 0;
    std::uint64_t count = 0;
    if (auto ok = reader.read(element_type); !ok) return ok;
    if (auto ok = reader.read(count); !ok) return ok;
    if (count > kMaxInventoryEntries) return std::unexpected("implausible array length");
    if (const auto width = scalar_width(element_type)) {
        // One bounds check replaces count per-element reads.
        if (count > 0 && *width > reader.remaining() / count) {
            return std::unexpected("truncated metadata array");
        }
        return reader.skip(count * *width);
    }
    if (element_type == 8) { // string array: walk lengths, not bytes
        for (std::uint64_t index = 0; index < count; ++index) {
            if (auto ok = reader.skip_string(); !ok) return ok;
        }
        return {};
    }
    if (element_type != 9) {
        return std::unexpected("unknown metadata type " + std::to_string(element_type));
    }
    for (std::uint64_t index = 0; index < count; ++index) {
        if (auto ok = skip_value(reader, element_type, depth + 1); !ok) return ok;
    }
    return {};
}

// Skips one metadata value of the given GGUF type. Depth bounds nested
// arrays: a metadata value may contain arrays of arrays, and unbounded
// recursion is a stack overflow on hostile input. Scalar arrays skip in
// bulk and string arrays walk lengths only; neither ever materialises the
// bytes, because the tokenizer arrays carry hundreds of thousands of
// entries and their contents are not the parser's business.
inline std::expected<void, std::string> skip_value(Reader& reader, std::uint32_t type,
                                                   std::uint32_t depth = 0) {
    if (const auto width = scalar_width(type)) return reader.skip(*width);
    switch (type) {
    case 8: return reader.skip_string(); // string
    case 9: return skip_array(reader, depth);
    default: return std::unexpected("unknown metadata type " + std::to_string(type));
    }
}

inline std::expected<void, std::string> skip_metadata(Reader& reader, std::uint64_t count,
                                                      std::uint64_t& alignment) {
    if (count > kMaxInventoryEntries) return std::unexpected("implausible metadata count");
    for (std::uint64_t index = 0; index < count; ++index) {
        auto key = reader.read_string_view();
        if (!key) return std::unexpected(key.error());
        std::uint32_t type = 0;
        if (auto ok = reader.read(type); !ok) return std::unexpected(ok.error());
        if (*key == "general.alignment" && type == 4) {
            std::uint32_t value = 0;
            if (auto ok = reader.read(value); !ok) return std::unexpected(ok.error());
            alignment = value;
            continue;
        }
        if (auto ok = skip_value(reader, type); !ok) return std::unexpected(ok.error());
    }
    return {};
}

/// Reads one tensor extent. A value past INT64_MAX would alias to a
/// negative dimension once stored (the upstream ggml_nelements overflow
/// shape, CVE-2026-33298 — see UPSTREAM.md), so it is rejected here.
inline std::expected<std::int64_t, std::string> read_extent(Reader& reader) {
    std::uint64_t extent = 0;
    if (auto ok = reader.read(extent); !ok) return std::unexpected(ok.error());
    if (extent > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return std::unexpected("tensor dimension not representable");
    }
    return static_cast<std::int64_t>(extent);
}

/// A parse record borrows the mapping: sorting happens over trivially
/// movable views and the owning TensorInfo vector is materialised once,
/// already in order, with no string moves in flight.
struct TensorRecord {
    std::string_view name;
    std::uint32_t type = 0;
    std::uint32_t dimension_count = 0;
    std::array<std::int64_t, 4> dimensions{};
    std::uint64_t offset = 0;
};

/// Reads a record's extents and keeps their product representable: a
/// product that wraps silently disables every downstream size check.
inline std::expected<void, std::string> read_extents(Reader& reader, TensorRecord& record) {
    std::uint64_t elements = 1;
    for (std::uint32_t dim = 0; dim < record.dimension_count; ++dim) {
        auto extent = read_extent(reader);
        if (!extent) return std::unexpected(extent.error());
        if (*extent != 0 && elements > std::numeric_limits<std::uint64_t>::max() /
                                           static_cast<std::uint64_t>(*extent)) {
            return std::unexpected("tensor extent product overflows");
        }
        elements *= static_cast<std::uint64_t>(*extent);
        record.dimensions[dim] = *extent;
    }
    return {};
}

inline std::expected<std::vector<TensorInfo>, std::string> read_tensors(Reader& reader,
                                                                        std::uint64_t count) {
    if (count > kMaxInventoryEntries) return std::unexpected("implausible tensor count");
    std::vector<TensorRecord> records;
    records.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        TensorRecord record;
        auto name = reader.read_string_view();
        if (!name) return std::unexpected(name.error());
        record.name = *name;
        if (auto ok = reader.read(record.dimension_count); !ok) {
            return std::unexpected(ok.error());
        }
        if (record.dimension_count > 4) return std::unexpected("tensor rank above four");
        if (record.dimension_count == 0) return std::unexpected("tensor rank below one");
        if (auto ok = read_extents(reader, record); !ok) return std::unexpected(ok.error());
        if (auto ok = reader.read(record.type); !ok) return std::unexpected(ok.error());
        if (auto ok = reader.read(record.offset); !ok) return std::unexpected(ok.error());
        records.push_back(record);
    }
    std::ranges::sort(records, {}, &TensorRecord::name);
    std::vector<TensorInfo> tensors;
    tensors.reserve(records.size());
    for (const TensorRecord& record : records) {
        TensorInfo info;
        info.name.assign(record.name);
        info.type = record.type;
        info.dimension_count = record.dimension_count;
        info.dimensions = record.dimensions;
        info.offset = record.offset;
        info.stored_length = row_size_bytes(record.type, record.dimensions, record.dimension_count);
        tensors.push_back(std::move(info));
    }
    return tensors;
}

inline std::expected<std::vector<TensorInfo>, std::string> read_inventory(const std::string& path) {
    const Mapping mapping(path);
    if (!mapping.valid()) return std::unexpected("cannot open " + path);
    Reader reader(mapping);

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t metadata_count = 0;
    if (auto ok = reader.read(magic); !ok) return std::unexpected(ok.error());
    if (magic != 0x46554747U) return std::unexpected("not a GGUF file");
    if (auto ok = reader.read(version); !ok) return std::unexpected(ok.error());
    if (version < 2 || version > 3) return std::unexpected("unsupported GGUF version");
    if (auto ok = reader.read(tensor_count); !ok) return std::unexpected(ok.error());
    if (auto ok = reader.read(metadata_count); !ok) return std::unexpected(ok.error());

    std::uint64_t alignment = 32;
    if (auto ok = skip_metadata(reader, metadata_count, alignment); !ok) {
        return std::unexpected(ok.error());
    }

    auto tensors = read_tensors(reader, tensor_count);
    if (!tensors) return std::unexpected(tensors.error());

    // The native loader reports absolute file offsets: data section start
    // plus the tensor record's relative offset.
    const std::uint64_t data_start = (reader.tell() + alignment - 1) / alignment * alignment;
    for (TensorInfo& tensor : *tensors) {
        tensor.offset += data_start;
    }
    // Package tensor ids must be strictly increasing.
    std::ranges::sort(*tensors, {}, &TensorInfo::name);
    return tensors;
}

// Package identity is the content hash, so the probe hashes the shard.
inline std::string sha256_of(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
    static constexpr std::array<std::uint32_t, 64> k = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    };
    std::array<std::uint32_t, 8> h{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    const auto rotr = [](std::uint32_t value, int bits) {
        const auto width = static_cast<std::uint32_t>(32 - bits);
        return (value >> static_cast<std::uint32_t>(bits)) | (value << width);
    };
    const std::uint64_t bit_length = static_cast<std::uint64_t>(data.size()) * 8U;
    data.push_back(0x80);
    while (data.size() % 64 != 56)
        data.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8) {
        data.push_back(static_cast<std::uint8_t>(bit_length >> static_cast<unsigned>(shift)));
    }
    for (std::size_t block = 0; block < data.size(); block += 64) {
        std::array<std::uint32_t, 64> w{};
        for (std::size_t index = 0; index < 16; ++index) {
            const std::size_t base = block + (index * 4);
            w[index] = (static_cast<std::uint32_t>(data[base]) << 24U) |
                       (static_cast<std::uint32_t>(data[base + 1]) << 16U) |
                       (static_cast<std::uint32_t>(data[base + 2]) << 8U) |
                       static_cast<std::uint32_t>(data[base + 3]);
        }
        for (std::size_t index = 16; index < 64; ++index) {
            const auto s0 =
                rotr(w[index - 15], 7) ^ rotr(w[index - 15], 18) ^ (w[index - 15] >> 3U);
            const auto s1 = rotr(w[index - 2], 17) ^ rotr(w[index - 2], 19) ^ (w[index - 2] >> 10U);
            w[index] = w[index - 16] + s0 + w[index - 7] + s1;
        }
        auto [a, b, c, d, e, f, g, hh] = h;
        for (std::size_t index = 0; index < 64; ++index) {
            const auto s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const auto ch = (e & f) ^ (~e & g);
            const auto t1 = hh + s1 + ch + k[index] + w[index];
            const auto s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const auto maj = (a & b) ^ (a & c) ^ (b & c);
            const auto t2 = s0 + maj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }
    static constexpr std::string_view kHex = "0123456789abcdef";
    std::string digest = "sha256:";
    for (const std::uint32_t word : h) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            digest.push_back(kHex[(word >> static_cast<unsigned>(shift)) & 0xfU]);
        }
    }
    return digest;
}

/// Reads `general.architecture`, the one metadata value the facade types.
inline std::expected<std::string, std::string> model_architecture(const std::string& path) {
    const Mapping mapping(path);
    if (!mapping.valid()) return std::unexpected("cannot open " + path);
    Reader reader(mapping);

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    [[maybe_unused]] std::uint64_t tensor_count = 0;
    std::uint64_t metadata_count = 0;
    if (auto ok = reader.read(magic); !ok) return std::unexpected(ok.error());
    if (auto ok = reader.read(version); !ok) return std::unexpected(ok.error());
    if (auto ok = reader.read(tensor_count); !ok) return std::unexpected(ok.error());
    if (auto ok = reader.read(metadata_count); !ok) return std::unexpected(ok.error());
    if (metadata_count > kMaxInventoryEntries) {
        return std::unexpected("implausible metadata count");
    }

    std::string architecture;
    for (std::uint64_t index = 0; index < metadata_count; ++index) {
        auto key = reader.read_string_view();
        if (!key) return std::unexpected(key.error());
        std::uint32_t type = 0;
        if (auto ok = reader.read(type); !ok) return std::unexpected(ok.error());
        if (*key == "general.architecture" && type == 8) {
            auto value = reader.read_string_view();
            if (!value) return std::unexpected(value.error());
            architecture.assign(*value);
            continue;
        }
        if (auto ok = skip_value(reader, type); !ok) return std::unexpected(ok.error());
    }
    return architecture;
}

inline std::int64_t layer_count_from(const std::vector<TensorInfo>& tensors) {
    std::int64_t layers = 0;
    for (const TensorInfo& tensor : tensors) {
        const auto position = tensor.name.rfind("blk.");
        if (position == std::string::npos) continue;
        const auto dot = tensor.name.find('.', position);
        if (dot == std::string::npos) continue;
        layers =
            std::max(layers, std::stoll(tensor.name.substr(position + 4, dot - position - 4)) + 1);
    }
    return layers;
}

/// Memoises a per-file-version computation without a mutex: readers snapshot
/// an atomic index of results and never block; a miss computes outside any
/// critical section and publishes copy-on-write with a compare-and-swap. The
/// shared_ptr atomics use a short internal spin on libc++ rather than being
/// truly lock-free, which is fine here — reads dominate and a file version is
/// hashed and parsed exactly once.
///
/// `compute` returns `std::optional<Value>`: an engaged result is published,
/// an absent one is *not*, so a transient failure (a busy file, a corrupt
/// shard) is retried on the next call instead of being cached forever. The
/// absence rule lives here, once, rather than at every call site.
template <typename Value, typename Compute>
inline std::optional<Value> memoised_file_result(const std::string& stamp, Compute compute) {
    // static in a template: one index per Value type, shared across TUs. These
    // are properties of the file version, not of any runtime or backend, so
    // two backends sharing them is correct.
    static std::shared_ptr<const std::unordered_map<std::string, Value>> index =
        std::make_shared<const std::unordered_map<std::string, Value>>();
    std::shared_ptr<const std::unordered_map<std::string, Value>> current =
        std::atomic_load_explicit(&index, std::memory_order_acquire);
    if (const auto found = current->find(stamp); found != current->end()) {
        return std::optional<Value>{found->second};
    }

    std::optional<Value> computed = compute(); // outside any lock; may be slow
    if (!computed) return std::nullopt;
    for (;;) {
        auto next = std::make_shared<std::unordered_map<std::string, Value>>(*current);
        next->emplace(stamp, *computed);
        if (std::atomic_compare_exchange_weak_explicit(
                &index, &current,
                std::static_pointer_cast<const std::unordered_map<std::string, Value>>(next),
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            return computed;
        }
        // Lost the race to a newer generation: if it already carries our
        // stamp return theirs, otherwise merge into that generation.
        if (const auto found = current->find(stamp); found != current->end()) {
            return std::optional<Value>{found->second};
        }
    }
}

/// Reads the inventory once per file version.
///
/// Parsing a 468 MB shard costs tens of milliseconds and hashing it costs
/// seconds, so both are memoised on the file's size and modification time.
inline std::expected<std::vector<TensorInfo>, std::string>
cached_inventory(const std::string& path) {
    std::string failure;
    auto tensors = memoised_file_result<std::vector<TensorInfo>>(
        file_stamp(path), [&] -> std::optional<std::vector<TensorInfo>> {
            auto read = read_inventory(path);
            if (!read) {
                failure = read.error();
                return std::nullopt;
            }
            return std::move(read).value();
        });
    if (!tensors.has_value()) return std::unexpected(std::move(failure));
    return {std::move(tensors).value()};
}

/// Hashes the shard once per file version.
inline std::string cached_sha256(const std::string& path) {
    // sha256_of cannot fail, so the optional is always engaged here.
    auto digest =
        memoised_file_result<std::string>(file_stamp(path), [&] { return sha256_of(path); });
    return digest.value_or(std::string{});
}

} // namespace skippy::probe

#endif // SKIPPY_API_TOOLS_GGUF_INVENTORY_H
