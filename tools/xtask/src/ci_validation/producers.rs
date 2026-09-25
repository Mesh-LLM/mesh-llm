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
        "python3 -m unittest discover -s scripts/tests -p 'test_*.py'",
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
    check_protected_reusable_runner_policy(sources.native_sdk, "native SDK reusable workflow")?;
    check_protected_reusable_runner_policy(sources.static_abi, "static ABI reusable workflow")?;

    Ok(())
}

fn check_protected_reusable_runner_policy(workflow: &str, context: &str) -> DynResult<()> {
    for (required, contract) in [
        ("runner_size:", "bounded runner-size input"),
        ("default: '8'", "bounded runner-size default"),
        ("runner_policy:", "protected runner policy job"),
        ("runs-on: ubuntu-24.04", "fixed hosted policy runner"),
        (
            "uses: ./.github/actions/select-ci-runners",
            "central protected runner selector",
        ),
        (
            "repository: ${{ github.repository }}",
            "immutable repository context",
        ),
        (
            "head_repository: ${{ github.event.pull_request.head.repo.full_name }}",
            "same-repository PR head context",
        ),
        ("ref: ${{ github.ref }}", "immutable ref context"),
        (
            "original_event_name: ${{ inputs.original_event_name }}",
            "protected original event context",
        ),
        (
            "depot_main_enabled: ${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
            "repository Depot gate",
        ),
        (
            "depot_pr_enabled: ${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
            "repository PR Depot gate",
        ),
        (
            "manual_use_depot: ${{ inputs.use_depot }}",
            "typed main-dispatch canary flag",
        ),
        (
            "runner_size must be one of: default, 4, 8, 16",
            "bounded runner-size validation",
        ),
        (
            "runs-on: ${{ needs.runner_policy.outputs.runner }}",
            "derived producer runner",
        ),
        (
            "allow_depot_remote_cache: ${{ needs.runner_policy.outputs.allow_depot_remote_cache }}",
            "derived Depot cache authority",
        ),
    ] {
        ensure_contains(workflow, required, &format!("{context} {contract}"))?;
    }
    for (forbidden, contract) in [
        ("inputs.runs_on", "caller-controlled runner label"),
        (
            "inputs.allow_depot_remote_cache",
            "caller-controlled Depot cache authority",
        ),
        ("fromJson(inputs.runs_on)", "caller-controlled runner JSON"),
    ] {
        ensure_not_contains(workflow, forbidden, &format!("{context} {contract}"))?;
    }
    Ok(())
}
