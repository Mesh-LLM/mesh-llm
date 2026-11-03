// SKIPPY_DELETE: C++26 reasons on deleted functions, degraded gracefully.
// The pinning static_asserts live in tests/type_contracts_test.cpp; the
// "fallback can be dropped" check lives in tests/toolchain_test.cpp.
#pragma once

#if defined(__cpp_deleted_function) && __cpp_deleted_function >= 202403L
#define SKIPPY_DELETE(reason) = delete (reason)
#else
#define SKIPPY_DELETE(reason) = delete
#endif
