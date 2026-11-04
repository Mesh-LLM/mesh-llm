#pragma once
#include <string>
#include <string_view>
namespace rust {
class String {
public:
    String() = default;
    String(const std::string&);
    String(const char*);
    String(const char*, std::size_t);
    static String lossy(std::string_view) noexcept;
};
template <typename T> class Vec { public: void push_back(const T&); void reserve(std::size_t); };
} // namespace rust
