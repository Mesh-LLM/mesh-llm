// CHECKS: skippy-no-void-silencing
#include <cstddef>
#include <vector>
std::size_t materialize(const std::vector<int>& into) {
    std::size_t imports = 0;
    imports += into.size();
    (void)imports;
    // CHECK: [[@LINE-1]]:5: warning: 'imports' is never read except to be discarded
    const std::size_t used = into.size();
    (void)used;
    return used;
}
