#ifndef SKIPPY_API_TOOLS_GGUF_INVENTORY_H
#define SKIPPY_API_TOOLS_GGUF_INVENTORY_H

/// Reads a GGUF's tensor inventory straight from the header.
///
/// The planner is configured from an exact tensor closure, so anything that
/// plans a real model needs this map first. It lives beside the llama backend
/// because the byte size of a tensor is a ggml type property.

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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

class Reader {
public:
    explicit Reader(const std::string& path) : file_(path, std::ios::binary) {}

    [[nodiscard]] bool okay() const { return file_.good(); }
    [[nodiscard]] std::uint64_t tell() { return static_cast<std::uint64_t>(file_.tellg()); }

    template <typename T> std::expected<void, std::string> read(T& value) {
        file_.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!file_) return std::unexpected("truncated header");
        return {};
    }

    std::expected<void, std::string> skip(std::uint64_t count) {
        file_.seekg(static_cast<std::streamoff>(count), std::ios::cur);
        if (!file_) return std::unexpected("truncated header");
        return {};
    }

    std::expected<std::string, std::string> read_string() {
        std::uint64_t length = 0;
        if (auto ok = read(length); !ok) return std::unexpected(ok.error());
        if (length > kMaxMetadataString) return std::unexpected("implausible string length");
        std::string value(length, '\0');
        file_.read(value.data(), static_cast<std::streamsize>(length));
        if (!file_) return std::unexpected("truncated string");
        return value;
    }

private:
    std::ifstream file_;
};

// Skips one metadata value of the given GGUF type.
inline std::expected<void, std::string> skip_value(Reader& reader, std::uint32_t type) {
    switch (type) {
    case 0:                         // u8
    case 1:                         // i8
    case 7: return reader.skip(1);  // bool
    case 2:                         // u16
    case 3: return reader.skip(2);  // i16
    case 4:                         // u32
    case 5:                         // i32
    case 6: return reader.skip(4);  // f32
    case 10:                        // u64
    case 11:                        // i64
    case 12: return reader.skip(8); // f64
    case 8: {                       // string
        auto value = reader.read_string();
        if (!value) return std::unexpected(value.error());
        return {};
    }
    case 9: { // array
        std::uint32_t element_type = 0;
        std::uint64_t count = 0;
        if (auto ok = reader.read(element_type); !ok) return ok;
        if (auto ok = reader.read(count); !ok) return ok;
        for (std::uint64_t index = 0; index < count; ++index) {
            if (auto ok = skip_value(reader, element_type); !ok) return ok;
        }
        return {};
    }
    default: return std::unexpected("unknown metadata type " + std::to_string(type));
    }
}

inline std::expected<void, std::string> skip_metadata(Reader& reader, std::uint64_t count,
                                                      std::uint64_t& alignment) {
    for (std::uint64_t index = 0; index < count; ++index) {
        auto key = reader.read_string();
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

inline std::expected<std::vector<TensorInfo>, std::string> read_tensors(Reader& reader,
                                                                        std::uint64_t count) {
    std::vector<TensorInfo> tensors;
    tensors.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        auto name = reader.read_string();
        if (!name) return std::unexpected(name.error());
        TensorInfo info;
        info.name = std::move(*name);
        if (auto ok = reader.read(info.dimension_count); !ok) return std::unexpected(ok.error());
        if (info.dimension_count > 4) return std::unexpected("tensor rank above four");
        for (std::uint32_t dim = 0; dim < info.dimension_count; ++dim) {
            std::uint64_t extent = 0;
            if (auto ok = reader.read(extent); !ok) return std::unexpected(ok.error());
            info.dimensions[dim] = static_cast<std::int64_t>(extent);
        }
        if (auto ok = reader.read(info.type); !ok) return std::unexpected(ok.error());
        if (auto ok = reader.read(info.offset); !ok) return std::unexpected(ok.error());
        info.stored_length = ggml_row_size(static_cast<ggml_type>(info.type), info.dimensions[0]);
        for (std::uint32_t dim = 1; dim < info.dimension_count; ++dim) {
            info.stored_length *= static_cast<std::uint64_t>(info.dimensions[dim]);
        }
        tensors.push_back(std::move(info));
    }
    return tensors;
}

inline std::expected<std::vector<TensorInfo>, std::string> read_inventory(const std::string& path) {
    Reader reader(path);
    if (!reader.okay()) return std::unexpected("cannot open " + path);

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
    Reader reader(path);
    if (!reader.okay()) return std::unexpected("cannot open " + path);

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    [[maybe_unused]] std::uint64_t tensor_count = 0;
    std::uint64_t metadata_count = 0;
    if (auto ok = reader.read(magic); !ok) return std::unexpected(ok.error());
    if (auto ok = reader.read(version); !ok) return std::unexpected(ok.error());
    if (auto ok = reader.read(tensor_count); !ok) return std::unexpected(ok.error());
    if (auto ok = reader.read(metadata_count); !ok) return std::unexpected(ok.error());

    std::string architecture;
    for (std::uint64_t index = 0; index < metadata_count; ++index) {
        auto key = reader.read_string();
        if (!key) return std::unexpected(key.error());
        std::uint32_t type = 0;
        if (auto ok = reader.read(type); !ok) return std::unexpected(ok.error());
        if (*key == "general.architecture" && type == 8) {
            auto value = reader.read_string();
            if (!value) return std::unexpected(value.error());
            architecture = *value;
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
    auto digest = memoised_file_result<std::string>(file_stamp(path),
                                                    [&] { return sha256_of(path); });
    return digest.value_or(std::string{});
}

} // namespace skippy::probe

#endif // SKIPPY_API_TOOLS_GGUF_INVENTORY_H
