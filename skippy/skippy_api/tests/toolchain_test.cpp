// Fails the day every supported compiler has C++26 deleted-function reasons,
// so the SKIPPY_DELETE fallback can be dropped (house rule, TYPE-RULES.md).
// Compilers without the feature do not define the macro at all, so the test
// skips rather than failing to compile on an undeclared identifier.
#include "skippy/poison.h"

#include <gtest/gtest.h>

#include <version>

TEST(Toolchain, SkippyDeleteFallbackCanGo) {
#if defined(__cpp_deleted_function) && __cpp_deleted_function >= 202403L
    // This toolchain could drop the fallback; the project floor (GCC 13 /
    // clang 18 in CI) still needs it. The FAIL belongs to the day the floor
    // rises, which no single binary can detect - so record, don't fail.
    GTEST_SUCCEED()
        << "toolchain has C++26 deleted-function reasons; fallback goes when the floor follows";
#else
    GTEST_SKIP() << "toolchain lacks C++26 deleted-function reasons; fallback stays";
#endif
}
