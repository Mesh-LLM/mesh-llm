//! Dependency-cache identity and compatibility between the trusted seed and consumers.
use super::{
    Node,
    cache_evidence::{steps, text},
    cache_predicate::requires,
};
use std::collections::{BTreeMap, BTreeSet};
fn all_steps(document: &Node) -> impl Iterator<Item = &Node> {
    document
        .get("jobs")
        .into_iter()
        .flat_map(Node::entries)
        .flat_map(|(_, job)| steps(job))
}
fn exact(node: &Node, key: &str, value: &str) -> Result<(), String> {
    if text(node, key) != value {
        return Err(format!("dependency cache {key} changed"));
    }
    Ok(())
}
// This reads only the hashFiles argument list, not the Actions expression language.
fn hashes(expression: &str) -> Result<BTreeSet<&str>, String> {
    let (_, tail) = expression
        .split_once("hashFiles(")
        .ok_or("cache hashFiles missing")?;
    let (arguments, _) = tail.split_once(')').ok_or("cache hashFiles unclosed")?;
    arguments
        .split(',')
        .map(|argument| {
            let argument = argument.trim();
            argument
                .strip_prefix('\'')
                .and_then(|v| v.strip_suffix('\''))
                .filter(|v| !v.is_empty() && !v.contains('\''))
                .ok_or_else(|| "cache hashFiles arguments must be literal paths".into())
        })
        .collect()
}
fn swift(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    let swift = workflows
        .get("swift-sdk-artifact.yml")
        .ok_or("Swift dependency cache missing")?;
    let mut swift_count = 0;
    for step in all_steps(swift).filter(|s| text(s, "uses").starts_with("Swatinem/rust-cache@")) {
        swift_count += 1;
        exact(
            step,
            "uses",
            "Swatinem/rust-cache@6323deb102c322ba6fcbdcafc7e3dddab59af2b6",
        )?;
        let values = step.get("with").ok_or("Swift cache inputs missing")?;
        if !text(values, "shared-key").starts_with("swift-sdk-") {
            return Err("Swift cache must retain target identity".into());
        }
        exact(values, "key", "${{ steps.native_toolchain.outputs.epoch }}")?;
        exact(values, "add-job-id-key", "false")?;
        for predicate in [
            "github.event_name == 'push'",
            "github.ref == 'refs/heads/main'",
        ] {
            if !requires(text(values, "save-if"), predicate) {
                return Err("Swift seed writes require main push".into());
            }
        }
    }
    if swift_count == 0 {
        return Err("Swift dependency cache absent".into());
    }
    Ok(())
}
fn rust_smoke(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    let sdk = workflows.get("sdk-smoke.yml").ok_or("SDK smoke missing")?;
    let caches: Vec<_> = all_steps(sdk)
        .filter(|s| text(s, "uses").starts_with("Swatinem/rust-cache@"))
        .collect();
    let [cache] = caches.as_slice() else {
        return Err("SDK Rust smoke requires one dependency cache".into());
    };
    if !requires(text(cache, "if"), "inputs.sdk_kind == 'rust'") {
        return Err("SDK dependency cache must select Rust".into());
    }
    let values = cache.get("with").ok_or("SDK cache inputs missing")?;
    exact(values, "shared-key", "ci-sdk-smoke-rust")?;
    exact(values, "cache-bin", "false")?;
    if !requires(text(values, "save-if"), "github.ref == 'refs/heads/main'") {
        return Err("SDK seed writes require main".into());
    }
    let prefix = text(values, "prefix-key");
    for boundary in [
        "sdk-rust-cargo-v1",
        "env.SDK_RUST_TARGET",
        "env.SDK_RUST_IMAGE_DIGEST",
        "env.SDK_RUST_TOOLCHAIN_EPOCH",
        "env.SDK_RUST_PROFILE_LINKER",
    ] {
        if !prefix.contains(boundary) {
            return Err(format!(
                "SDK cache compatibility boundary missing: {boundary}"
            ));
        }
    }
    let paths = hashes(prefix)?;
    for path in [
        "Cargo.lock",
        ".github/cache-version.txt",
        ".cargo/config.toml",
        "scripts/cargo-linker",
        "scripts/cargo-linker-linux-*",
        "scripts/lib/lld.sh",
        "**/Cargo.toml",
        "scripts/ci-rust-sdk-smoke.sh",
        "scripts/ci-sdk-fixture.sh",
        "scripts/ci-prepare-native-runtime.sh",
        "scripts/package-sdk-console-assets.sh",
        "scripts/check-sdk-contract.sh",
        "scripts/verify-sdk-console-assets.sh",
        ".github/workflows/sdk-smoke.yml",
    ] {
        if !paths.contains(path) {
            return Err(format!("SDK cache source boundary missing: {path}"));
        }
    }
    if all_steps(sdk).any(|s| text(s, "uses").starts_with("actions/cache")) {
        return Err("SDK smoke must use dependency-aware Rust cache".into());
    }
    Ok(())
}
fn seed(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    let warmer = workflows
        .get("cache-warm-sccache.yml")
        .ok_or("compiler seed warmer missing")?;
    let seed = all_steps(warmer)
        .find(|s| text(s, "id") == "seed")
        .ok_or("seed identity output missing")?;
    let assignments: Vec<_> = text(seed, "run")
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("key=\"")
                .and_then(|v| v.strip_suffix('"'))
        })
        .collect();
    let [key] = assignments.as_slice() else {
        return Err("one compiler seed key assignment required".into());
    };
    if !key.starts_with("mesh-llm-sccache-seed-linux-x86_64-img-") || !key.contains("-epoch-") {
        return Err("compiler seed image/toolchain identity missing".into());
    }
    let paths = hashes(key)?;
    for path in [
        "Cargo.lock",
        ".github/cache-version.txt",
        ".cargo/config.toml",
        "scripts/cargo-linker",
        "scripts/cargo-linker-linux-*",
        "scripts/lib/lld.sh",
        "Justfile",
        "just/**",
    ] {
        if !paths.contains(path) {
            return Err(format!(
                "compiler seed compatibility boundary missing: {path}"
            ));
        }
    }
    let mut caches = 0;
    for step in all_steps(warmer).filter(|s| text(s, "uses").starts_with("actions/cache/")) {
        caches += 1;
        exact(
            step.get("with").ok_or("warmer cache inputs missing")?,
            "key",
            "${{ steps.seed.outputs.key }}",
        )?;
    }
    if caches != 2 {
        return Err("compiler seed restore/save pair missing".into());
    }
    if !all_steps(warmer).any(|s| text(s, "run") == "just ci-sccache-seed-build") {
        return Err("bounded compiler seed build recipe missing".into());
    }
    for name in [
        "ci-quality-slice.yml",
        "ci-rust-tests-slice.yml",
        "ci-linux-host-slice.yml",
        "ci-linux-runtime-slice.yml",
    ] {
        let workflow = workflows
            .get(name)
            .ok_or("compiler seed consumer workflow missing")?;
        let restores: Vec<_> = all_steps(workflow)
            .filter(|s| text(s, "uses") == "./.github/actions/restore-sccache-seed")
            .collect();
        if restores.is_empty() {
            return Err(format!("{name}: compiler seed consumer missing"));
        }
        for restore in restores {
            let inputs = restore.get("with").ok_or("compiler seed inputs missing")?;
            exact(inputs, "cache_key", key)?;
            if name == "ci-linux-runtime-slice.yml" {
                exact(inputs, "allow_trusted_seed", "false")?;
            }
        }
    }
    Ok(())
}
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    swift(workflows)?;
    rust_smoke(workflows)?;
    seed(workflows)
}
