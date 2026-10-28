#include "family.h"

#include <array>
#include <utility>

namespace skippy {
namespace {

using Entry = std::pair<std::string_view, ModelFamily>;

constexpr std::array kFamilies{
    Entry{"llama", ModelFamily::Llama},     Entry{"qwen", ModelFamily::Qwen},
    Entry{"qwen2", ModelFamily::Qwen},      Entry{"qwen2vl", ModelFamily::Qwen},
    Entry{"qwen3", ModelFamily::Qwen},      Entry{"mistral", ModelFamily::Mistral},
    Entry{"gemma", ModelFamily::Gemma},     Entry{"gemma2", ModelFamily::Gemma},
    Entry{"gemma3", ModelFamily::Gemma},    Entry{"phi2", ModelFamily::Phi},
    Entry{"phi3", ModelFamily::Phi},        Entry{"bert", ModelFamily::Bert},
    Entry{"nomic-bert", ModelFamily::Bert}, Entry{"mamba", ModelFamily::Mamba},
    Entry{"rwkv", ModelFamily::Rwkv},       Entry{"t5", ModelFamily::T5},
};

} // namespace

ModelFamily family_of(std::string_view architecture) noexcept {
    for (const auto& [name, family] : kFamilies) {
        if (name == architecture) return family;
    }
    return ModelFamily::Unknown;
}

std::string_view describe(ModelFamily family) noexcept {
    switch (family) {
    case ModelFamily::Unknown: return "unknown";
    case ModelFamily::Llama: return "llama";
    case ModelFamily::Qwen: return "qwen";
    case ModelFamily::Mistral: return "mistral";
    case ModelFamily::Gemma: return "gemma";
    case ModelFamily::Phi: return "phi";
    case ModelFamily::Bert: return "bert";
    case ModelFamily::Mamba: return "mamba";
    case ModelFamily::Rwkv: return "rwkv";
    case ModelFamily::T5: return "t5";
    }
    return "unknown";
}

} // namespace skippy
