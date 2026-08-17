use std::path::{Path, PathBuf};

const LLAMA_CLI_ENV: &str = "SKIPPY_QUANTIZE_LLAMA_CLI";
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

pub(crate) fn resolve_llama_cli(explicit: Option<&Path>) -> Option<PathBuf> {
    resolve_tool(explicit, LLAMA_CLI_ENV, LLAMA_CLI_CANDIDATES)
}

fn resolve_tool(
    explicit: Option<&Path>,
    env_var: &str,
    relative_candidates: &[&str],
) -> Option<PathBuf> {
    if let Some(explicit) = explicit {
        return Some(explicit.to_path_buf());
    }
    if let Some(from_env) = std::env::var_os(env_var).map(PathBuf::from)
        && from_env.is_file()
    {
        return Some(from_env);
    }
    for root in candidate_roots() {
        for relative in relative_candidates {
            let candidate = root.join(relative);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn candidate_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(current) = std::env::current_dir() {
        roots.push(current);
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    roots.push(manifest_dir.clone());
    if let Some(repo_root) = manifest_dir.parent().and_then(Path::parent) {
        roots.push(repo_root.to_path_buf());
    }
    roots.dedup();
    roots
}
