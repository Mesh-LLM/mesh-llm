use crate::command::{DynResult, ensure_contains, ensure_not_contains, workflow_job_section};

pub(super) fn check_windows_dynamic_runtime_contract(
    host_workflow: &str,
    runtime_and_product_workflows: &str,
    prepare_windows_host_action: &str,
    prepare_native_runtime_action: &str,
    compose_product_action: &str,
) -> DynResult<()> {
    ensure_contains(
        prepare_windows_host_action,
        r"& .\scripts\build-windows.ps1 -BuildProfile $profile -HostOnly",
        "shared Windows host action canonical host-only build",
    )?;
    ensure_contains(
        prepare_windows_host_action,
        r"scripts\verify-host-dependencies.py",
        "shared Windows host action import-policy verification",
    )?;
    ensure_not_contains(
        prepare_windows_host_action,
        "package-native-runtime.sh",
        "shared Windows host action must not build a native runtime",
    )?;
    ensure_contains(
        prepare_native_runtime_action,
        r#"scripts/package-native-runtime.sh "${args[@]}""#,
        "shared native-runtime action canonical runtime builder",
    )?;
    ensure_not_contains(
        prepare_native_runtime_action,
        "build-windows.ps1",
        "shared native-runtime action must not build the Windows host",
    )?;
    ensure_contains(
        compose_product_action,
        "scripts/ci-compose-product-input.sh",
        "shared product action canonical composition script",
    )?;

    let host = workflow_job_section(host_workflow, "windows_host")
        .ok_or("host slice: missing `windows_host` job")?;
    let runtime = workflow_job_section(runtime_and_product_workflows, "windows_runtime")
        .ok_or("runtime slice: missing `windows_runtime` job")?;
    let product = workflow_job_section(runtime_and_product_workflows, "windows_product")
        .ok_or("runtime slice: missing `windows_product` job")?;

    ensure_contains(
        host,
        "uses: ilammy/msvc-dev-cmd@0b201ec74fa43914dc39ae48a89fd1d8cb592756",
        "host slice persistent MSVC host environment",
    )?;
    ensure_contains(
        host,
        "uses: ./.github/actions/prepare-windows-host-input",
        "host slice shared immutable Windows host producer",
    )?;
    ensure_contains(
        runtime,
        "uses: ./.github/actions/prepare-native-runtime-input",
        "runtime slice shared Windows runtime producer",
    )?;
    ensure_contains(
        runtime,
        "target: ${{ matrix.runtime.target }}",
        "runtime slice planned Windows target",
    )?;
    ensure_contains(
        product,
        "uses: ./.github/actions/compose-product-input",
        "runtime slice shared Windows product composer",
    )?;
    ensure_contains(
        product,
        "binary_name: mesh-llm.exe",
        "runtime slice Windows product executable",
    )?;
    ensure_contains(
        product,
        "readiness_smoke: \"true\"",
        "runtime slice Windows product readiness",
    )?;

    for forbidden in [
        "cargo ",
        "dtolnay/rust-toolchain",
        "Swatinem/rust-cache",
        "mozilla-actions/sccache-action",
        "scripts/build-windows.ps1",
        "scripts/package-native-runtime.sh",
        "prepare-windows-host-input",
        "prepare-native-runtime-input",
    ] {
        ensure_not_contains(
            product,
            forbidden,
            "runtime slice Windows product composition-only contract",
        )?;
    }

    Ok(())
}
