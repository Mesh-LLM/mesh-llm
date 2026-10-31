// CHECKS: skippy-no-process-wide-state
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
namespace skippy::detail { std::uint64_t next_generation() { static std::atomic<std::uint64_t> next{1}; return next++; } }
template <typename V> V* memo(const std::string&) {
    static std::map<std::string, V> index;
    // CHECK: [[@LINE-1]]:37: warning: 'index' is mutable process-wide state
    return nullptr;
}
int* instantiate() { return memo<int>("x"); }
inline constexpr int kFixed = 3;
static std::once_flag initialised;
