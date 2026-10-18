/// Runs a model as N chained stages through the facade's StageChain.

#include "llama/model_inventory.h"
#include "llama/stage_chain.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> [tokens]\n";
        return 2;
    }
    const std::string path = argv[1];
    const auto count = static_cast<int>(argc > 2 ? std::atoi(argv[2]) : 4);

    auto inventory = skippy::probe::cached_inventory(path);
    if (!inventory) {
        std::cerr << "inventory: " << inventory.error() << "\n";
        return 1;
    }
    const auto layers = static_cast<std::int32_t>(skippy::probe::layer_count_from(*inventory));
    const std::int32_t half = layers / 2;
    const std::array ranges{skippy::LayerRange{0, half}, skippy::LayerRange{half, layers}};

    auto chain = skippy::llama::StageChain::open(path, ranges);
    if (!chain) {
        std::cerr << "open: " << chain.error().message << "\n";
        return 1;
    }
    std::cout << "chain of " << chain->stage_count() << " stages\n";

    std::vector<llama_token> tokens(static_cast<std::size_t>(count), 1);
    skippy::CopyFrameSink sink;
    const auto outcome = chain->prefill(tokens, sink);
    if (outcome.has_value()) {
        std::cerr << "run: " << outcome->message << "\n";
        return 1;
    }
    std::cout << "terminal frame: " << (sink.frame ? sink.frame->size() : 0) << " bytes\n";
    return 0;
}
