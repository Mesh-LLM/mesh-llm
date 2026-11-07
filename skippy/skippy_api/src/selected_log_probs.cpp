#include "selected_log_probs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace skippy::detail {

bool selected_log_probabilities(std::span<const float> logits, std::span<const Token> selected,
                                std::span<float> output) noexcept {
    if (logits.empty() || selected.empty() || output.size() < selected.size() ||
        selected.size() > max_selected_tokens) {
        return false;
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (const float logit : logits) {
        const double value = logit;
        if (std::isnan(value) || value == std::numeric_limits<double>::infinity()) return false;
        maximum = std::max(value, maximum);
    }
    if (!std::isfinite(maximum)) return false;
    double sum = 0;
    for (const float logit : logits) {
        sum += std::exp(static_cast<double>(logit) - maximum);
    }
    const double normalizer = std::log(sum);
    std::array<float, max_selected_tokens> scores;
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const auto token = static_cast<std::int32_t>(selected[index]);
        if (token < 0 || static_cast<std::size_t>(token) >= logits.size()) return false;
        const double value = (static_cast<double>(logits[token]) - maximum) - normalizer;
        scores[index] = static_cast<float>(value);
        if (!std::isfinite(scores[index])) return false;
    }
    for (std::size_t index = 0; index < selected.size(); ++index)
        output[index] = scores[index];
    return true;
}

} // namespace skippy::detail
