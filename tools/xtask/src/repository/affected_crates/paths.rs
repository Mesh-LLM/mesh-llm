//! Path classification from `scripts/affected-crates.sh`: website inputs, UI
//! inputs, and the escalation set that selects the whole workspace.

pub(super) const FORCE_ALL: &str = "__force_all__";

/// Build/native scripts whose change can alter the Rust link line.
const ESCALATING_SCRIPTS: &[&str] = &[
    "build-llama",
    "prepare-llama",
    "build-linux",
    "build-linux-rocm",
    "build-mac",
    "build-windows",
    "skippy-ci-smoke",
    "ci-install-native-runtime",
    "ci-prepare-native-runtime",
    "ci-smoke-test",
    "ci-compat-smoke",
    "ci-client-auto-test",
    "ci-two-node-client-serving-smoke",
    "ci-two-node-split-smoke",
];

const WEBSITE_DOC_FILES: &[&str] = &[
    "index.html",
    "CNAME",
    "install.sh",
    "install.ps1",
    "mesh-llm-logo.svg",
];

const WEBSITE_DOC_DIRS: &[&str] = &["assets", "catalog", "docs", "pagefind"];

pub(super) fn is_website_input(file: &str) -> bool {
    if file.starts_with("website/") || file == "install.sh" || file == "install.ps1" {
        return true;
    }
    let Some(rest) = file.strip_prefix("docs/") else {
        return false;
    };
    WEBSITE_DOC_FILES.contains(&rest)
        || WEBSITE_DOC_DIRS.iter().any(|dir| {
            rest.strip_prefix(dir)
                .is_some_and(|tail| tail.is_empty() || tail.starts_with('/'))
        })
}

pub(super) fn is_ui_input(file: &str) -> bool {
    file.starts_with("crates/mesh-llm-ui/")
}

/// Whether one changed path forces the all-workspace selection.
pub(super) fn escalates(file: &str) -> bool {
    file == FORCE_ALL
        || file == "third_party/llama.cpp/upstream.txt"
        || file.starts_with("third_party/llama.cpp/patches/")
        || file == "Cargo.lock"
        || file == "Cargo.toml"
        || file == ".github/cache-version.txt"
        || file == "rust-toolchain"
        || file == "rust-toolchain.toml"
        || file.strip_prefix("scripts/").is_some_and(|rest| {
            ESCALATING_SCRIPTS.iter().any(|name| {
                rest.strip_prefix(name)
                    .is_some_and(|tail| tail.starts_with('.'))
            })
        })
}

/// Whether a changed path can be owned by a Rust crate.
pub(super) fn may_own_rust(file: &str) -> bool {
    (file.starts_with("crates/") || file.starts_with("tools/")) && !is_ui_input(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_repository_affected_path_classes() {
        assert!(is_website_input("docs/assets"));
        assert!(is_website_input("docs/pagefind/x.js"));
        assert!(!is_website_input("docs/assetsx/y"));
        assert!(!is_website_input("docs/MESHES.md"));
        assert!(escalates("scripts/build-llama.sh"));
        assert!(!escalates("scripts/build-llamax.sh"));
        assert!(!escalates("scripts/build-llama"));
        assert!(escalates("third_party/llama.cpp/patches/0001.patch"));
        assert!(!may_own_rust("crates/mesh-llm-ui/src/app.tsx"));
        assert!(may_own_rust("tools/xtask/src/main.rs"));
    }
}
