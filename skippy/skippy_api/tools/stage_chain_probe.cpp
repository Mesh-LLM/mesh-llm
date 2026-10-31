/// Runs a model as N chained stages through the facade's StageChain.

#include "llama/model_inventory.h"
#include "llama/stage_chain.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {
// bounded strtol: a probe count from argv; a non-numeric argument keeps the default
int parse_int(const char* text, int fallback) noexcept {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 2147483647L) return fallback;
    return static_cast<int>(value); // bounded by the range check above
}
} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape): a probe main reads as a script; an unexpected throw terminating with the message is the diagnostic
int main(
    int argc,
    char**
        argv) { // NOLINT(bugprone-exception-escape): a probe that throws has nothing to keep alive; terminating with the message is the diagnostic
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> [tokens]\n";
        return 2;
    }
    const std::string path = argv[1];
    const auto count = argc > 2 ? parse_int(argv[2], 4) : 4;

    auto inventory = skippy::probe::cached_inventory(path);
    if (!inventory) {
        std::cerr << "inventory: " << inventory.error() << "\n";
        return 1;
    }
    const auto layers = skippy::probe::layer_count_from(*inventory);
    const std::int32_t half = layers / 2;
    const std::array ranges{
        skippy::LayerRange{.start = 0, .end = half},
        skippy::LayerRange{.start = half, .end = layers},
    };

    auto chain = skippy::llama::StageChain::open(path, ranges);
    if (!chain) {
        std::cerr << "open: " << chain.error().message << "\n";
        return 1;
    }
    std::cout << "chain of " << chain->stage_count() << " stages\n";

    std::vector<llama_token> tokens(static_cast<std::size_t>(count), 1);
    skippy::CopyFrameSink sink;
    const auto outcome = chain->prefill(tokens, sink);
    if (!outcome) {
        std::cerr << "run: " << outcome.error().message << "\n";
        return 1;
    }
    std::cout << "terminal frame: " << (sink.frame ? sink.frame->size() : 0) << " bytes\n";
    return 0;
}
