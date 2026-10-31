// Fuzzes the GGUF inventory parser with bytes the facade did not produce:
// hand-written files, a directory path, a missing path and an empty file. The
// mapping, reader, tensor-bounds and digest paths are all in the target.
#include "llama/model_inventory.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

const std::filesystem::path& scratch_dir() {
    static const std::filesystem::path dir = [] {
        const auto path = std::filesystem::temp_directory_path() / "skippy_fuzz_inventory";
        std::filesystem::create_directories(path);
        return path;
    }();
    return dir;
}

void write_file(const std::filesystem::path& path, const std::uint8_t* data, std::size_t size) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size == 0) return 0;
    const unsigned shape = data[0] & 0x03U;
    ++data;
    --size;

    const auto file = scratch_dir() / "input.gguf";
    const auto directory = scratch_dir() / "adirectory";
    std::filesystem::create_directory(directory);
    std::string path = file.string();
    switch (shape) {
    case 0U: write_file(file, data, size); break;
    case 1U:
        path = directory.string(); // a directory path
        break;
    case 2U:
        path = (scratch_dir() / "missing.gguf").string(); // a missing path
        break;
    default:
        write_file(file, data, 0U); // an empty file
        break;
    }

    const auto inventory = skippy::probe::read_inventory(path);
    [[maybe_unused]] const auto architecture = skippy::probe::model_architecture(path);
    [[maybe_unused]] const auto digest = skippy::probe::sha256_of(path);
    if (inventory.has_value()) {
        (void)skippy::probe::layer_count_from(*inventory);
        (void)skippy::probe::has_classifier_head(*inventory);
    }
    return 0;
}
