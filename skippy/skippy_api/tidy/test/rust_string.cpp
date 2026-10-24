// CHECKS: skippy-rust-string-lossy
#include <string>
#include "rust_stub.h"
rust::String hex_of(std::string_view);
rust::String from_native(const std::string& message) { return rust::String(message); }
// CHECK: [[@LINE-1]]:63: warning: rust::String(...) validates UTF-8 and throws on native bytes
rust::String from_literal() { return rust::String("a plan needs at least one stage"); }
rust::String from_hex(std::string_view bytes) { return rust::String(hex_of(bytes)); }
rust::String lossy(const std::string& message) { return rust::String::lossy(message); }
