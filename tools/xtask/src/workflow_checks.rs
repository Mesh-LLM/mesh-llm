use crate::command::{
    DynResult, ensure_contains, ensure_not_contains, ensure_set_eq, workflow_job_section,
};
use crate::repo_consistency::{script_workspace_members, workspace_package_names};
use std::fs;
use std::path::Path;
use std::process::Command;

pub(crate) fn check_docs_and_workflow_invariants(repo_root: &Path) -> DynResult<()> {
    check_current_ci_invariants(repo_root)
}

fn check_current_ci_invariants(repo_root: &Path) -> DynResult<()> {
    let readme = fs::read_to_string(repo_root.join("README.md"))?;
    let contributing = fs::read_to_string(repo_root.join("CONTRIBUTING.md"))?;
    let release = fs::read_to_string(repo_root.join("RELEASE.md"))?;
    let release_package_source = fs::read_to_string(repo_root.join("just/release-bundle.just"))?;
    let release_workflow = fs::read_to_string(repo_root.join(".github/workflows/release.yml"))?;
    let ci_workflow = fs::read_to_string(repo_root.join(".github/workflows/ci.yml"))?;
    let pr_workflows = ["quality", "website", "linux", "macos", "windows"]
        .into_iter()
        .map(|lane| {
            fs::read_to_string(repo_root.join(format!(".github/workflows/pr_{lane}.yml")))
                .map(|workflow| (lane, workflow))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let main_workflows = ["quality", "website", "linux", "macos", "windows"]
        .into_iter()
        .map(|lane| {
            fs::read_to_string(repo_root.join(format!(".github/workflows/main_{lane}.yml")))
                .map(|workflow| (lane, workflow))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let controller = fs::read_to_string(repo_root.join(".github/workflows/ci-control.yml"))?;
    let lane_workflows = ["quality", "website", "linux", "macos", "windows"]
        .into_iter()
        .map(|lane| {
            fs::read_to_string(repo_root.join(format!(".github/workflows/ci-{lane}-lane.yml")))
                .map(|workflow| (lane, workflow))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let quality = fs::read_to_string(repo_root.join(".github/workflows/ci-quality-slice.yml"))?;
    let web = fs::read_to_string(repo_root.join(".github/workflows/ci-web-slice.yml"))?;
    let host = ["linux", "macos", "windows"]
        .into_iter()
        .map(|platform| {
            fs::read_to_string(
                repo_root.join(format!(".github/workflows/ci-{platform}-host-slice.yml")),
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .join("\n");
    let runtime_and_product = ["linux", "macos", "windows"]
        .into_iter()
        .flat_map(|platform| {
            ["runtime", "product"]
                .into_iter()
                .map(move |component| (platform, component))
        })
        .map(|(platform, component)| {
            fs::read_to_string(repo_root.join(format!(
                ".github/workflows/ci-{platform}-{component}-slice.yml"
            )))
        })
        .collect::<Result<Vec<_>, _>>()?
        .join("\n");
    let rust_tests =
        fs::read_to_string(repo_root.join(".github/workflows/ci-rust-tests-slice.yml"))?;
    let static_abi =
        fs::read_to_string(repo_root.join(".github/workflows/static-abi-artifact.yml"))?;
    let native_sdk =
        fs::read_to_string(repo_root.join(".github/workflows/native-sdk-artifact.yml"))?;
    let swift_sdk = fs::read_to_string(repo_root.join(".github/workflows/swift-sdk-artifact.yml"))?;
    let website_pages = fs::read_to_string(repo_root.join(".github/workflows/website-pages.yml"))?;
    let compute_changes =
        fs::read_to_string(repo_root.join(".github/actions/compute-changes/action.yml"))?;
    let prepare_windows_host = fs::read_to_string(
        repo_root.join(".github/actions/prepare-windows-host-input/action.yml"),
    )?;
    let prepare_runtime = fs::read_to_string(
        repo_root.join(".github/actions/prepare-native-runtime-input/action.yml"),
    )?;
    let compose_product =
        fs::read_to_string(repo_root.join(".github/actions/compose-product-input/action.yml"))?;
    let configure_sccache =
        fs::read_to_string(repo_root.join(".github/actions/configure-sccache-gha/action.yml"))?;
    let ci_docs = fs::read_to_string(repo_root.join("ci/ci.md"))?;
    let depot_docs = fs::read_to_string(repo_root.join("ci/DEPOT_MIGRATION.md"))?;

    check_documentation_invariants(
        &readme,
        &contributing,
        &release,
        &release_package_source,
        &ci_docs,
        &depot_docs,
    )?;
    check_workflow_invariants(
        &release_workflow,
        &ci_workflow,
        &pr_workflows,
        &main_workflows,
        &website_pages,
    )?;
    check_producer_invariants(&ProducerInvariantSources {
        quality: &quality,
        web: &web,
        host: &host,
        runtime_and_product: &runtime_and_product,
        rust_tests: &rust_tests,
        static_abi: &static_abi,
        native_sdk: &native_sdk,
        swift_sdk: &swift_sdk,
        prepare_windows_host: &prepare_windows_host,
        prepare_runtime: &prepare_runtime,
        compose_product: &compose_product,
    })?;
    check_orchestrator_invariants(
        &controller,
        &pr_workflows,
        &main_workflows,
        &lane_workflows,
        &compute_changes,
    )?;
    check_release_dispatch_version_preparation(&release_workflow, &native_sdk, &swift_sdk)?;
    check_release_container_contracts(&release_workflow, &configure_sccache)?;
    check_windows_dynamic_runtime_contract(
        &host,
        &runtime_and_product,
        &prepare_windows_host,
        &prepare_runtime,
        &compose_product,
    )
}

fn check_documentation_invariants(
    readme: &str,
    contributing: &str,
    release: &str,
    release_package_source: &str,
    ci_docs: &str,
    depot_docs: &str,
) -> DynResult<()> {
    for (text, needle, context) in [
        (
            readme,
            "mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
            "README ARM64 asset note",
        ),
        (
            readme,
            "mesh-llm-aarch64-unknown-linux-gnu-cuda.tar.gz",
            "README ARM64 CUDA asset note",
        ),
        (
            release,
            "Windows release artifacts use the `x86_64-pc-windows-msvc` target triple",
            "RELEASE Windows publish note",
        ),
        (
            release_package_source,
            "cargo run -p xtask -- repo-consistency release-targets",
            "Imported Just release consistency command",
        ),
        (
            contributing,
            "just check-release",
            "CONTRIBUTING release consistency command",
        ),
        (ci_docs, "CI · Manual Full", "CI manual-full workflow"),
        (ci_docs, "Main / Quality", "native main workflow results"),
        (ci_docs, "CI Required", "CI topology required summary"),
        (
            depot_docs,
            "Cache isolation",
            "Depot cache-isolation policy",
        ),
    ] {
        ensure_contains(text, needle, context)?;
    }
    Ok(())
}

fn check_workflow_invariants(
    release_workflow: &str,
    ci_workflow: &str,
    pr_workflows: &[(&str, String)],
    main_workflows: &[(&str, String)],
    website_pages: &str,
) -> DynResult<()> {
    for (text, needle, context) in [
        (
            release_workflow,
            "compose_windows_gpu:",
            "release Windows GPU composition",
        ),
        (
            release_workflow,
            "publish_crates_preflight:",
            "release crates.io preflight",
        ),
        (
            website_pages,
            "name: Public Website Deploy",
            "public website deploy workflow",
        ),
        (
            website_pages,
            "branches: [main]",
            "public website main trigger",
        ),
    ] {
        ensure_contains(text, needle, context)?;
    }

    ensure_contains(ci_workflow, "workflow_call:", "legacy main CI shim")?;
    ensure_not_contains(ci_workflow, "push:", "legacy main CI event trigger")?;
    for (lane, workflow) in pr_workflows {
        ensure_contains(workflow, "pull_request:", &format!("PR {lane} trigger"))?;
        ensure_contains(
            workflow,
            &format!("uses: Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main"),
            &format!("PR protected native {lane} lane call"),
        )?;
        ensure_contains(
            workflow,
            "needs: [plan, lane]",
            &format!("PR {lane} required job"),
        )?;
        ensure_not_contains(
            workflow,
            "pull_request_target",
            &format!("PR {lane} trust boundary"),
        )?;
        ensure_not_contains(workflow, "secrets:", &format!("PR {lane} secret boundary"))?;
    }
    for (lane, workflow) in main_workflows {
        ensure_contains(
            workflow,
            "push:\n    branches: [main]",
            &format!("main {lane} trigger"),
        )?;
        ensure_contains(
            workflow,
            &format!("uses: ./.github/workflows/ci-{lane}-lane.yml"),
            &format!("main same-commit {lane} lane call"),
        )?;
        ensure_contains(
            workflow,
            "needs: [plan, lane]",
            &format!("main {lane} required job"),
        )?;
        ensure_not_contains(
            workflow,
            "createWorkflowDispatch",
            &format!("main {lane} native visibility"),
        )?;
        ensure_not_contains(
            workflow,
            "concurrency:",
            &format!("main {lane} exhaustive evidence"),
        )?;
    }
    ensure_not_contains(
        ci_workflow,
        "uses: ./.github/workflows/ci-orchestrator.yml",
        "main entrypoint must not expand the monolithic bootstrap graph",
    )?;
    Ok(())
}

struct ProducerInvariantSources<'a> {
    quality: &'a str,
    web: &'a str,
    host: &'a str,
    runtime_and_product: &'a str,
    rust_tests: &'a str,
    static_abi: &'a str,
    native_sdk: &'a str,
    swift_sdk: &'a str,
    prepare_windows_host: &'a str,
    prepare_runtime: &'a str,
    compose_product: &'a str,
}

fn check_producer_invariants(sources: &ProducerInvariantSources<'_>) -> DynResult<()> {
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

fn check_orchestrator_invariants(
    controller: &str,
    pr_workflows: &[(&str, String)],
    main_workflows: &[(&str, String)],
    lanes: &[(&str, String)],
    compute_changes: &str,
) -> DynResult<()> {
    ensure_contains(
        controller,
        "name: CI · Manual Full",
        "manual-full workflow identity",
    )?;
    ensure_contains(
        controller,
        "uses: ./.github/actions/plan-ci",
        "controller canonical planner call",
    )?;
    ensure_contains(
        controller,
        "github.rest.actions.createWorkflowDispatch",
        "controller native lane dispatch",
    )?;
    ensure_contains(controller, "workflow_dispatch:", "manual-full trigger")?;
    ensure_not_contains(controller, "workflow_run:", "manual controller trigger")?;
    ensure_not_contains(controller, "\n  push:\n", "manual controller push trigger")?;
    let lane_workflow = |name: &str| {
        lanes
            .iter()
            .find_map(|(lane, workflow)| (*lane == name).then_some(workflow.as_str()))
            .unwrap_or("")
    };
    for lane in ["quality", "website", "linux", "macos", "windows"] {
        let pr_workflow = pr_workflows
            .iter()
            .find_map(|(name, workflow)| (*name == lane).then_some(workflow.as_str()))
            .unwrap_or("");
        ensure_contains(
            pr_workflow,
            &format!("uses: Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main"),
            &format!("native PR {lane} lane call"),
        )?;
        let main_workflow = main_workflows
            .iter()
            .find_map(|(name, workflow)| (*name == lane).then_some(workflow.as_str()))
            .unwrap_or("");
        ensure_contains(
            main_workflow,
            &format!("uses: ./.github/workflows/ci-{lane}-lane.yml"),
            &format!("native main {lane} lane call"),
        )?;
        ensure_contains(
            lane_workflow(lane),
            "workflow_call:",
            &format!("{lane} reusable lane trigger"),
        )?;
    }
    ensure_contains(
        lane_workflow("quality"),
        "uses: ./.github/workflows/ci-quality-slice.yml",
        "quality lane slice call",
    )?;
    for lane in ["linux", "macos", "windows"] {
        for component in ["host", "runtime", "product"] {
            ensure_contains(
                lane_workflow(lane),
                &format!("uses: ./.github/workflows/ci-{lane}-{component}-slice.yml"),
                &format!("{lane} lane {component} call"),
            )?;
        }
        for legacy in ["ci-host-slice.yml", "ci-runtime-product-slice.yml"] {
            ensure_not_contains(
                lane_workflow(lane),
                legacy,
                &format!("{lane} lane cross-platform placeholder graph"),
            )?;
        }
    }
    for (lane, component) in [
        ("linux", "product-smoke"),
        ("linux", "sdk"),
        ("macos", "product-smoke"),
        ("macos", "sdk"),
    ] {
        ensure_contains(
            lane_workflow(lane),
            &format!("uses: ./.github/workflows/ci-{lane}-{component}-slice.yml"),
            &format!("{lane} lane {component} call"),
        )?;
    }
    ensure_contains(
        controller,
        "name: 'CI Required'",
        "stable dispatched CI required check",
    )?;
    ensure_contains(
        compute_changes,
        "changed_files:",
        "changed-file planner input",
    )?;
    ensure_contains(
        compute_changes,
        "affected_crates:",
        "affected-crate planner input",
    )?;
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

fn check_release_dispatch_version_preparation(
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

fn check_release_container_contracts(
    release_workflow: &str,
    configure_sccache_action: &str,
) -> DynResult<()> {
    const REQUIRED_STEP: &str = "Trust checkout directory";
    const REQUIRED_COMMAND: &str = "git config --global --add safe.directory \"$GITHUB_WORKSPACE\"";
    const LOCAL_SCCACHE_ENV: &str = "      SCCACHE_GHA_ENABLED: \"false\"";
    const CONFIGURE_SCCACHE_ACTION: &str = "      - uses: ./.github/actions/configure-sccache-gha";
    const COMPOSE_PRODUCT_ACTION: &str = "uses: ./.github/actions/compose-product-input";
    const PREPARE_RUNTIME_ACTION: &str = "uses: ./.github/actions/prepare-native-runtime-input";
    const PINNED_GITHUB_SCRIPT: &str =
        "uses: actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd";

    for (required, context) in [
        (
            "  SCCACHE_DIR: ${{ github.workspace }}/../.sccache",
            "release workflow sccache disk cache",
        ),
        (
            "  SCCACHE_IGNORE_SERVER_IO_ERROR: \"1\"",
            "release workflow sccache compiler fallback",
        ),
        (
            "  SCCACHE_MULTILEVEL_CHAIN: disk,gha",
            "release workflow sccache cache chain",
        ),
        (
            "  SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY: ignore",
            "release workflow sccache write fallback",
        ),
    ] {
        ensure_contains(release_workflow, required, context)?;
    }

    ensure_contains(
        configure_sccache_action,
        PINNED_GITHUB_SCRIPT,
        "sccache GHA action pinned credential exporter",
    )?;
    ensure_not_contains(
        configure_sccache_action,
        "mozilla-actions/sccache-action",
        "sccache GHA action must use the baked binary",
    )?;
    for (required, context) in [
        (
            "core.exportVariable('ACTIONS_RESULTS_URL'",
            "sccache GHA action cache URL export",
        ),
        (
            "core.exportVariable('ACTIONS_RUNTIME_TOKEN'",
            "sccache GHA action runtime token export",
        ),
        (
            "core.exportVariable('SCCACHE_GHA_ENABLED', 'true')",
            "sccache GHA action remote enable",
        ),
        (
            "core.exportVariable('SCCACHE_GHA_ENABLED', 'false')",
            "sccache GHA action job-local fallback",
        ),
        (
            "core.exportVariable('SCCACHE_IGNORE_SERVER_IO_ERROR', '1')",
            "sccache GHA action compiler fallback",
        ),
        (
            "core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk,gha')",
            "sccache GHA action cache chain",
        ),
        (
            "process.env.SCCACHE_WEBDAV_ENDPOINT",
            "sccache Depot WebDAV endpoint",
        ),
        ("process.env.DEPOT_CACHE_TOKEN", "sccache Depot job token"),
        (
            "core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk,webdav')",
            "sccache Depot cache chain",
        ),
        (
            "core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk')",
            "sccache GHA action disk-only fallback",
        ),
        (
            "core.exportVariable('SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY', 'all')",
            "sccache GHA action synchronous remote writes",
        ),
        ("['--start-server']", "sccache GHA action server start"),
        ("['--stop-server']", "sccache GHA action server stop"),
    ] {
        ensure_contains(configure_sccache_action, required, context)?;
    }

    let container_jobs = release_container_job_names(release_workflow);
    if container_jobs.is_empty() {
        return Err("release workflow: expected at least one container job".into());
    }

    for job_name in container_jobs {
        let job = workflow_job_section(release_workflow, job_name).ok_or_else(|| {
            format!("release workflow: missing `{job_name}` job for container contract check")
        })?;
        ensure_contains(
            job,
            REQUIRED_STEP,
            &format!("release workflow `{job_name}` safe-directory step"),
        )?;
        ensure_contains(
            job,
            REQUIRED_COMMAND,
            &format!("release workflow `{job_name}` safe-directory command"),
        )?;
        let composition_only =
            job.contains(COMPOSE_PRODUCT_ACTION) && !job.contains(PREPARE_RUNTIME_ACTION);
        if composition_only {
            ensure_not_contains(
                job,
                CONFIGURE_SCCACHE_ACTION.trim(),
                &format!(
                    "release workflow `{job_name}` composition must not configure a compiler cache"
                ),
            )?;
            ensure_not_contains(
                job,
                "uses: actions/cache@",
                &format!(
                    "release workflow `{job_name}` composition must not restore a compiler cache"
                ),
            )?;
            continue;
        }
        if !job.lines().any(|line| line == LOCAL_SCCACHE_ENV) {
            return Err(format!(
                "release workflow `{job_name}`: missing job-level `{}`",
                LOCAL_SCCACHE_ENV.trim()
            )
            .into());
        }
        if !job.lines().any(|line| line == CONFIGURE_SCCACHE_ACTION) {
            return Err(format!(
                "release workflow `{job_name}`: missing `{}`",
                CONFIGURE_SCCACHE_ACTION.trim()
            )
            .into());
        }
    }

    Ok(())
}

fn release_container_job_names(release_workflow: &str) -> Vec<&str> {
    release_workflow
        .lines()
        .filter_map(|line| {
            let job_name = line.strip_prefix("  ")?.strip_suffix(':')?;
            if job_name.is_empty() || job_name.starts_with(' ') || job_name.contains(' ') {
                return None;
            }
            let job = workflow_job_section(release_workflow, job_name)?;
            job.lines()
                .any(|job_line| job_line == "    container:")
                .then_some(job_name)
        })
        .collect()
}

fn check_windows_dynamic_runtime_contract(
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

fn check_ci_crate_test_coverage(
    linux_lane_workflow: &str,
    quality_workflow: &str,
) -> DynResult<()> {
    ensure_contains(
        linux_lane_workflow,
        "rust_tests_matrix: ${{ toJson(fromJson(inputs.lane_plan_json).matrices.rust_tests) }}",
        "Linux lane Rust test matrix input",
    )?;
    ensure_contains(
        linux_lane_workflow,
        "uses: ./.github/workflows/ci-rust-tests-slice.yml",
        "Linux lane shared Rust test slice",
    )?;
    ensure_contains(
        quality_workflow,
        "python3 -m unittest discover -s scripts/tests -p 'test_*.py'",
        "quality CI contract test suite",
    )?;

    Ok(())
}

pub(crate) fn check_ci_crate_test_coverage_files(repo_root: &Path) -> DynResult<()> {
    let linux_lane_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/ci-linux-lane.yml"))?;
    let quality_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/ci-quality-slice.yml"))?;
    check_ci_crate_test_coverage(&linux_lane_workflow, &quality_workflow)?;
    check_test_batch_planner_covers_workspace(repo_root)
}

fn check_test_batch_planner_covers_workspace(repo_root: &Path) -> DynResult<()> {
    let output = Command::new("bash")
        .current_dir(repo_root)
        .args(["scripts/plan-test-batches.sh", "--all", "--bins", "4"])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "test batch planner failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }

    let batches: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let mut actual = std::collections::BTreeSet::new();
    for crate_name in batches
        .as_array()
        .ok_or("test batch planner output must be an array")?
        .iter()
        .flat_map(|batch| batch["crates"].as_array().into_iter().flatten())
    {
        let crate_name = crate_name
            .as_str()
            .ok_or("test batch planner crate names must be strings")?;
        if !actual.insert(crate_name.to_owned()) {
            return Err(format!("test batch planner duplicated crate `{crate_name}`").into());
        }
    }

    let expected = workspace_package_names(repo_root)?;
    ensure_set_eq(&expected, &actual, "Cargo test batch workspace coverage")
}

pub(crate) fn check_ci_script_workspace_members(repo_root: &Path) -> DynResult<()> {
    let expected = workspace_package_names(repo_root)?;
    let scripts = [
        "scripts/affected-crates.sh",
        "scripts/plan-clippy-batches.sh",
    ];

    for script in scripts {
        let actual = script_workspace_members(repo_root, script)?;
        ensure_set_eq(&expected, &actual, &format!("{script} WORKSPACE_MEMBERS"))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_release_container_contracts;

    const VALID_SCCACHE_ACTION: &str = r#"
uses: actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd
core.exportVariable('ACTIONS_RESULTS_URL'
core.exportVariable('ACTIONS_RUNTIME_TOKEN'
core.exportVariable('SCCACHE_GHA_ENABLED', 'true')
core.exportVariable('SCCACHE_GHA_ENABLED', 'false')
core.exportVariable('SCCACHE_IGNORE_SERVER_IO_ERROR', '1')
core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk,gha')
process.env.SCCACHE_WEBDAV_ENDPOINT
process.env.DEPOT_CACHE_TOKEN
core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk,webdav')
core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk')
core.exportVariable('SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY', 'all')
['--start-server']
['--stop-server']
"#;

    const VALID_CONTAINER_WORKFLOW: &str = r#"env:
  SCCACHE_DIR: ${{ github.workspace }}/../.sccache
  SCCACHE_IGNORE_SERVER_IO_ERROR: "1"
  SCCACHE_MULTILEVEL_CHAIN: disk,gha
  SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY: ignore
jobs:
  build_linux_cuda:
    container:
      image: example.invalid/runner@sha256:digest
    env:
      SCCACHE_GHA_ENABLED: "false"
    steps:
      - uses: actions/checkout@v5
      - name: Trust checkout directory
        run: git config --global --add safe.directory "$GITHUB_WORKSPACE"
      - uses: ./.github/actions/configure-sccache-gha
  publish:
    runs-on: ubuntu-24.04
"#;

    const VALID_COMPOSITION_CONTAINER_WORKFLOW: &str = r#"env:
  SCCACHE_DIR: ${{ github.workspace }}/../.sccache
  SCCACHE_IGNORE_SERVER_IO_ERROR: "1"
  SCCACHE_MULTILEVEL_CHAIN: disk,gha
  SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY: ignore
jobs:
  compose_linux_cuda:
    container:
      image: example.invalid/runner@sha256:digest
    steps:
      - uses: actions/checkout@v5
      - name: Trust checkout directory
        run: git config --global --add safe.directory "$GITHUB_WORKSPACE"
      - uses: ./.github/actions/compose-product-input
  publish:
    runs-on: ubuntu-24.04
"#;

    #[test]
    fn release_container_contract_accepts_remote_sccache_with_local_fallback() {
        check_release_container_contracts(VALID_CONTAINER_WORKFLOW, VALID_SCCACHE_ACTION).unwrap();
    }

    #[test]
    fn release_container_contract_accepts_cache_free_product_composition() {
        check_release_container_contracts(
            VALID_COMPOSITION_CONTAINER_WORKFLOW,
            VALID_SCCACHE_ACTION,
        )
        .unwrap();
    }

    #[test]
    fn release_container_contract_requires_safe_checkout() {
        let workflow = VALID_CONTAINER_WORKFLOW.replace(
            "      - name: Trust checkout directory\n        run: git config --global --add safe.directory \"$GITHUB_WORKSPACE\"\n",
            "",
        );

        let error = check_release_container_contracts(&workflow, VALID_SCCACHE_ACTION).unwrap_err();
        assert!(error.to_string().contains("safe-directory"));
    }

    #[test]
    fn release_container_contract_requires_job_local_sccache() {
        let workflow =
            VALID_CONTAINER_WORKFLOW.replace("      SCCACHE_GHA_ENABLED: \"false\"\n", "");

        let error = check_release_container_contracts(&workflow, VALID_SCCACHE_ACTION).unwrap_err();
        assert!(error.to_string().contains("SCCACHE_GHA_ENABLED"));
    }

    #[test]
    fn release_container_contract_requires_sccache_gha_configuration() {
        let workflow = VALID_CONTAINER_WORKFLOW.replace(
            "      - uses: ./.github/actions/configure-sccache-gha\n",
            "",
        );

        let error = check_release_container_contracts(&workflow, VALID_SCCACHE_ACTION).unwrap_err();
        assert!(error.to_string().contains("configure-sccache-gha"));
    }

    #[test]
    fn release_container_contract_requires_sccache_job_local_fallback() {
        let action =
            VALID_SCCACHE_ACTION.replace("core.exportVariable('SCCACHE_GHA_ENABLED', 'false')", "");

        let error =
            check_release_container_contracts(VALID_CONTAINER_WORKFLOW, &action).unwrap_err();
        assert!(error.to_string().contains("job-local fallback"));
    }

    #[test]
    fn release_container_contract_requires_fail_open_sccache_writes() {
        let workflow = VALID_CONTAINER_WORKFLOW
            .replace("  SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY: ignore\n", "");

        let error = check_release_container_contracts(&workflow, VALID_SCCACHE_ACTION).unwrap_err();
        assert!(error.to_string().contains("write fallback"));
    }

    #[test]
    fn release_container_contract_requires_disk_first_sccache_chain() {
        let action = VALID_SCCACHE_ACTION.replace(
            "core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk,gha')",
            "",
        );

        let error =
            check_release_container_contracts(VALID_CONTAINER_WORKFLOW, &action).unwrap_err();
        assert!(error.to_string().contains("cache chain"));
    }
}
