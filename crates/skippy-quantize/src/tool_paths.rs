use std::path::{Path, PathBuf};

use crate::backend::resolve_external_tool;

const CONVERTER_ENV: &str = "SKIPPY_QUANTIZE_CONVERTER";
const LLAMA_QUANTIZE_ENV: &str = "SKIPPY_QUANTIZE_LLAMA_QUANTIZE";
const LLAMA_CLI_ENV: &str = "SKIPPY_QUANTIZE_LLAMA_CLI";
const CONVERTER_CANDIDATES: &[&str] = &[
    ".deps/llama.cpp/convert_hf_to_gguf.py",
    "../../.deps/llama.cpp/convert_hf_to_gguf.py",
];
const LLAMA_QUANTIZE_CANDIDATES: &[&str] = &[
    ".deps/llama.cpp/build/bin/llama-quantize",
    ".deps/llama.cpp/build/bin/Release/llama-quantize",
    "../../.deps/llama.cpp/build/bin/llama-quantize",
    "../../.deps/llama.cpp/build/bin/Release/llama-quantize",
];
const LLAMA_CLI_CANDIDATES: &[&str] = &[
    ".deps/llama.cpp/build-cli/bin/llama-cli",
    ".deps/llama.cpp/build-cli/bin/llama",
    ".deps/llama.cpp/build-cli/bin/llama-simple",
    ".deps/llama.cpp/build/bin/llama-cli",
    ".deps/llama.cpp/build/bin/llama",
    ".deps/llama.cpp/build/bin/llama-simple",
    ".deps/llama.cpp/build/bin/Release/llama-cli",
    ".deps/llama.cpp/build/bin/Release/llama",
    ".deps/llama.cpp/build/bin/Release/llama-simple",
    "../../.deps/llama.cpp/build-cli/bin/llama-cli",
    "../../.deps/llama.cpp/build-cli/bin/llama",
    "../../.deps/llama.cpp/build-cli/bin/llama-simple",
    "../../.deps/llama.cpp/build/bin/llama-cli",
    "../../.deps/llama.cpp/build/bin/llama",
    "../../.deps/llama.cpp/build/bin/llama-simple",
    "../../.deps/llama.cpp/build/bin/Release/llama-cli",
    "../../.deps/llama.cpp/build/bin/Release/llama",
    "../../.deps/llama.cpp/build/bin/Release/llama-simple",
];

pub(crate) fn resolve_converter(explicit: Option<&Path>) -> Option<PathBuf> {
    resolve_external_tool(explicit, CONVERTER_ENV, CONVERTER_CANDIDATES)
}

pub(crate) fn resolve_llama_quantize(explicit: Option<&Path>) -> Option<PathBuf> {
    resolve_external_tool(explicit, LLAMA_QUANTIZE_ENV, LLAMA_QUANTIZE_CANDIDATES)
}

pub(crate) fn resolve_llama_cli(explicit: Option<&Path>) -> Option<PathBuf> {
    resolve_external_tool(explicit, LLAMA_CLI_ENV, LLAMA_CLI_CANDIDATES)
}
