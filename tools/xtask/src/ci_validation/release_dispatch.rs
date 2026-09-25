use crate::command::{DynResult, ensure_contains, ensure_not_contains, workflow_job_section};

pub(super) fn check_release_dispatch_version_preparation(
    release_workflow: &str,
    native_sdk_artifact_workflow: &str,
    swift_sdk_artifact_workflow: &str,
) -> DynResult<()> {
    const DISPATCH_RELEASE_JOBS: &[&str] = &[
        "build",
        "build_linux_arm64",
        "compose_linux_aarch64_cuda",
        "compose_linux_cuda",
        "compose_linux_rocm",
        "compose_linux_vulkan",
        "windows_host_input",
    ];
    const REQUIRED_STEP: &str = "Prepare dispatched release version";
    const REQUIRED_COMMAND: &str = "scripts/release-version.sh \"$RELEASE_TAG\"";

    for job_name in DISPATCH_RELEASE_JOBS {
        let job = workflow_job_section(release_workflow, job_name).ok_or_else(|| {
            format!("release workflow: missing `{job_name}` job for dispatched version check")
        })?;
        ensure_contains(
            job,
            REQUIRED_STEP,
            &format!("release workflow `{job_name}` dispatch version step"),
        )?;
        ensure_contains(
            job,
            "if: github.event_name == 'workflow_dispatch'",
            &format!("release workflow `{job_name}` dispatch version condition"),
        )?;
        ensure_contains(
            job,
            REQUIRED_COMMAND,
            &format!("release workflow `{job_name}` dispatch version command"),
        )?;
    }

    let native_sdk_caller = workflow_job_section(release_workflow, "build_native_sdk_runtime")
        .ok_or("release workflow: missing `build_native_sdk_runtime` job")?;
    for (required, context) in [
        (
            "uses: ./.github/workflows/native-sdk-artifact.yml",
            "release native SDK shared producer call",
        ),
        ("profile: release", "release native SDK producer profile"),
        (
            "artifact_name: release-native-sdk-${{ matrix.artifact_suffix }}",
            "release native SDK artifact name",
        ),
        (
            "include_runtime_crate: true",
            "release native SDK runtime crate staging",
        ),
        (
            "static_abi_artifact_name: ci-release-native-sdk-static-abi-${{ matrix.artifact_suffix }}",
            "release native SDK static ABI artifact",
        ),
        (
            "produce_static_abi: ${{ endsWith(matrix.target, '-unknown-linux-gnu') }}",
            "release native SDK per-target static ABI producer",
        ),
        ("runner_size: '8'", "release native SDK bounded runner size"),
        (
            "release_tag: ${{ needs.metadata.outputs.tag }}",
            "release native SDK producer tag input",
        ),
        (
            "prepare_release_version: ${{ github.event_name == 'workflow_dispatch' }}",
            "release native SDK dispatch version input",
        ),
    ] {
        ensure_contains(native_sdk_caller, required, context)?;
    }
    ensure_not_contains(
        native_sdk_caller,
        "runs_on:",
        "release native SDK must not supply a runner label",
    )?;
    ensure_not_contains(
        native_sdk_caller,
        "allow_depot_remote_cache:",
        "release native SDK must not supply Depot cache authority",
    )?;
    ensure_contains(
        native_sdk_artifact_workflow,
        REQUIRED_STEP,
        "shared native SDK producer dispatch version step",
    )?;
    ensure_contains(
        native_sdk_artifact_workflow,
        "if: ${{ inputs.prepare_release_version }}",
        "shared native SDK producer dispatch version condition",
    )?;
    ensure_contains(
        native_sdk_artifact_workflow,
        REQUIRED_COMMAND,
        "shared native SDK producer dispatch version command",
    )?;

    let swift_caller = workflow_job_section(release_workflow, "build_swift_sdk_artifact")
        .ok_or("release workflow: missing `build_swift_sdk_artifact` job")?;
    for (required, context) in [
        (
            "uses: ./.github/workflows/swift-sdk-artifact.yml",
            "release Swift shared producer call",
        ),
        ("mode: full", "release Swift exhaustive producer mode"),
        (
            "release_tag: ${{ needs.metadata.outputs.tag }}",
            "release Swift producer tag input",
        ),
        (
            "prepare_release_version: ${{ github.event_name == 'workflow_dispatch' }}",
            "release Swift dispatch version input",
        ),
    ] {
        ensure_contains(swift_caller, required, context)?;
    }
    ensure_contains(
        swift_sdk_artifact_workflow,
        REQUIRED_STEP,
        "shared Swift producer dispatch version step",
    )?;
    ensure_contains(
        swift_sdk_artifact_workflow,
        "if: ${{ inputs.prepare_release_version }}",
        "shared Swift producer dispatch version condition",
    )?;
    ensure_contains(
        swift_sdk_artifact_workflow,
        REQUIRED_COMMAND,
        "shared Swift producer dispatch version command",
    )?;

    Ok(())
}
