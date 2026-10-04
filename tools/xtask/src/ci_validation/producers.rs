mod protected_runner_policy;
use crate::command::{DynResult, ensure_contains, ensure_not_contains};

pub(super) struct ProducerInvariantSources<'a> {
    pub(super) quality: &'a str,
    pub(super) web: &'a str,
    pub(super) host: &'a str,
    pub(super) runtime_and_product: &'a str,
    pub(super) rust_tests: &'a str,
    pub(super) static_abi: &'a str,
    pub(super) native_sdk: &'a str,
    pub(super) swift_sdk: &'a str,
    pub(super) prepare_windows_host: &'a str,
    pub(super) prepare_runtime: &'a str,
    pub(super) compose_product: &'a str,
}

pub(super) fn check_producer_invariants(sources: &ProducerInvariantSources<'_>) -> DynResult<()> {
    for (workflow, context) in [
        (sources.quality, "quality slice"),
        (sources.web, "web slice"),
        (sources.host, "host slice"),
        (sources.runtime_and_product, "runtime/product slice"),
        (sources.rust_tests, "Rust test slice"),
        (sources.static_abi, "static ABI producer"),
        (sources.native_sdk, "native SDK producer"),
        (sources.swift_sdk, "Swift SDK producer"),
    ] {
        ensure_contains(
            workflow,
            "persist-credentials: false",
            &format!("{context} safe checkout"),
        )?;
    }
    ensure_contains(
        sources.quality,
        "just ci-legacy-contracts",
        "quality contract suite",
    )?;
    ensure_contains(
        sources.quality,
        "cargo run -p xtask -- repo-consistency ci-crate-lists",
        "quality crate-list consistency",
    )?;
    ensure_contains(
        sources.quality,
        "cargo run -p xtask -- repo-consistency publish-crates",
        "quality publish consistency",
    )?;
    ensure_contains(sources.web, "website:", "web website sub-slice")?;
    ensure_contains(
        sources.host,
        "uses: ./.github/actions/prepare-host-input",
        "host immutable producer",
    )?;
    ensure_contains(
        sources.host,
        "uses: ./.github/actions/prepare-windows-host-input",
        "Windows host producer",
    )?;
    ensure_contains(
        sources.runtime_and_product,
        "uses: ./.github/actions/prepare-native-runtime-input",
        "runtime immutable producer",
    )?;
    ensure_contains(
        sources.runtime_and_product,
        "uses: ./.github/actions/compose-product-input",
        "composition-only product producer",
    )?;
    ensure_contains(
        sources.runtime_and_product,
        "binary_name: mesh-llm.exe",
        "Windows product executable",
    )?;
    ensure_contains(
        sources.rust_tests,
        "cargo test --locked",
        "Rust test command",
    )?;
    ensure_contains(
        sources.prepare_windows_host,
        "-HostOnly",
        "Windows host-only builder",
    )?;
    ensure_contains(
        sources.prepare_runtime,
        "scripts/package-native-runtime.sh",
        "native runtime builder",
    )?;
    ensure_not_contains(
        sources.compose_product,
        "cargo build",
        "composition must not compile",
    )?;
    protected_runner_policy::check(
        sources.native_sdk,
        "native SDK reusable workflow",
        protected_runner_policy::Producer::NativeSdk,
    )?;
    protected_runner_policy::check(
        sources.static_abi,
        "static ABI reusable workflow",
        protected_runner_policy::Producer::StaticAbi,
    )?;

    Ok(())
}
