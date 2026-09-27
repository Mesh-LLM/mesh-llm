use crate::command::{
    DynResult, ensure_contains, ensure_not_contains, ensure_set_eq, workflow_job_section,
};
use crate::repo_consistency::{script_workspace_members, workspace_package_names};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

pub(crate) fn check_docs_and_workflow_invariants(repo_root: &Path) -> DynResult<()> {
    check_current_ci_invariants(repo_root)
}

fn check_current_ci_invariants(repo_root: &Path) -> DynResult<()> {
    let readme = fs::read_to_string(repo_root.join("README.md"))?;
    let contributing = fs::read_to_string(repo_root.join("CONTRIBUTING.md"))?;
    let release = fs::read_to_string(repo_root.join("RELEASE.md"))?;
    let release_package_source = fs::read_to_string(repo_root.join("just/release-bundle.just"))?;
    let release_workflow = fs::read_to_string(repo_root.join(".github/workflows/release.yml"))?;
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
    check_release_sccache_initialization(&release_workflow)?;
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

/// Release jobs that only compose immutable producer artifacts. They consume
/// an already-versioned source and finished host/runtime inputs, so they must
/// never rewrite the release version and must never invoke cargo.
const RELEASE_COMPOSITION_ONLY_JOBS: &[&str] = &[
    "compose_cpu_products",
    "compose_linux_arm64_cpu",
    "compose_linux_aarch64_cuda",
    "compose_linux_cuda",
    "compose_linux_rocm",
    "compose_linux_vulkan",
    "compose_windows_cpu",
    "compose_windows_gpu",
];

/// Text that proves a release job can reach cargo, either directly or through
/// the release scripts that own version rewriting and crates.io packaging.
const RELEASE_CARGO_INVOCATIONS: &[&str] = &[
    "cargo ",
    "scripts/package-release.sh",
    "scripts/release-version.sh",
    "scripts/publish-crates.sh",
];

/// Steps that can initialize sccache before the first compiler probe, listed in
/// the order a job that uses both must run them: the installer puts the sccache
/// binary on `PATH`, and the repository action starts a server from that binary.
/// Some pinned runner images already include the binary, so those jobs need only
/// the repository action that configures its cache backend.
const RELEASE_SCCACHE_INITIALIZATION: &[&str] = &[
    "uses: mozilla-actions/sccache-action",
    "uses: ./.github/actions/configure-sccache-gha",
];

/// A hosted Ubuntu job starts without the repository's pinned runner image, so
/// it must install sccache before configuring its cache backend. Derive this
/// from the job instead of maintaining a name allowlist that a new job can
/// silently fall outside.
fn release_job_requires_hosted_sccache_setup(job: &str) -> bool {
    job.lines()
        .any(|line| line.trim() == "runs-on: ubuntu-24.04")
        && !job
            .lines()
            .any(|line| line.trim_start().starts_with("container:"))
}

/// Names of the jobs declared under `jobs:` in a workflow, in file order.
fn workflow_job_names(workflow: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_jobs = false;

    for raw_line in workflow.lines() {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if !in_jobs {
            in_jobs = line == "jobs:";
            continue;
        }
        if line.is_empty() {
            continue;
        }
        // A column-zero comment is still inside the `jobs:` mapping and must
        // not truncate discovery of the jobs declared after it.
        if line.trim_start().starts_with('#') {
            continue;
        }
        // Leaving the two-space indentation ends the `jobs:` mapping.
        if !line.starts_with(' ') {
            break;
        }
        let Some(candidate) = line
            .strip_prefix("  ")
            .and_then(|rest| rest.strip_suffix(':'))
        else {
            continue;
        };
        if candidate.is_empty()
            || candidate.starts_with(' ')
            || !candidate
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
        {
            continue;
        }
        names.push(candidate.to_string());
    }

    names
}

/// Every job that can invoke cargo must start sccache first, and a
/// composition-only job must never invoke cargo at all.
///
/// This is the static half of the release CI contract: it catches a cargo
/// caller that was never given the wrapper initialization, which a green
/// canary cannot catch for the canary-skipped jobs. Steps reached through local
/// composite actions or reusable workflows are outside this scan; only the
/// steps spelled out in `release.yml` are visible here.
fn check_release_sccache_initialization(release_workflow: &str) -> DynResult<()> {
    for job_name in workflow_job_names(release_workflow) {
        let job = workflow_job_section(release_workflow, &job_name)
            .ok_or_else(|| format!("release workflow: unable to read `{job_name}` job"))?;
        // Comments and step names may mention cargo without running it.
        let steps = job
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let mut cargo_invocations = RELEASE_CARGO_INVOCATIONS
            .iter()
            .filter(|invocation| steps.contains(**invocation))
            .copied()
            .collect::<Vec<_>>();
        if cargo_invocations.is_empty() {
            continue;
        }
        if RELEASE_COMPOSITION_ONLY_JOBS.contains(&job_name.as_str()) {
            if cargo_invocations.contains(&"scripts/package-release.sh") {
                for guard in [
                    "MESH_RELEASE_HOST_PRESTAMPED: \"1\"",
                    "MESH_RELEASE_ATTESTATION_PREVERIFIED: \"1\"",
                ] {
                    if !steps.contains(guard) {
                        return Err(format!(
                            "release workflow `{job_name}` runs scripts/package-release.sh without the cargo-bypass guard `{guard}`"
                        )
                        .into());
                    }
                }
                cargo_invocations.retain(|invocation| *invocation != "scripts/package-release.sh");
            }
            if !cargo_invocations.is_empty() {
                let invocation_list = cargo_invocations
                    .iter()
                    .map(|invocation| invocation.trim())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(format!(
                    "release workflow `{job_name}` is composition-only but invokes {invocation_list}"
                )
                .into());
            }
            continue;
        }
        let invocation_list = cargo_invocations
            .iter()
            .map(|invocation| invocation.trim())
            .collect::<Vec<_>>()
            .join(", ");
        let first_cargo_invocation = RELEASE_CARGO_INVOCATIONS
            .iter()
            .filter_map(|invocation| steps.find(invocation))
            .min()
            .expect("a cargo invocation was found above");
        let initialized_before_cargo = RELEASE_SCCACHE_INITIALIZATION
            .iter()
            .filter_map(|marker| {
                steps
                    .find(marker)
                    .filter(|position| *position < first_cargo_invocation)
                    .map(|position| (*marker, position))
            })
            .collect::<Vec<_>>();
        if release_job_requires_hosted_sccache_setup(job) {
            // Presence is not enough for a hosted job: the installer has to run
            // before the action that starts a server from the installed binary.
            let mut previous: Option<(&str, usize)> = None;
            for marker in RELEASE_SCCACHE_INITIALIZATION {
                let found = initialized_before_cargo
                    .iter()
                    .find(|(initialized_marker, _)| *initialized_marker == *marker);
                let Some((_, position)) = found else {
                    return Err(format!(
                        "release workflow `{job_name}` invokes {invocation_list} before required sccache initialization `{marker}`"
                    )
                    .into());
                };
                if let Some((previous_marker, previous_position)) = previous
                    && *position < previous_position
                {
                    return Err(format!(
                        "release workflow `{job_name}` runs `{marker}` before `{previous_marker}`; a hosted job installs sccache before it configures the cache backend"
                    )
                    .into());
                }
                previous = Some((*marker, *position));
            }
        } else if initialized_before_cargo.is_empty() {
            let marker = RELEASE_SCCACHE_INITIALIZATION.join("` or `");
            return Err(format!(
                "release workflow `{job_name}` invokes {invocation_list} before any required sccache initialization (`{marker}`)"
            )
            .into());
        }
    }

    Ok(())
}

fn check_release_dispatch_version_preparation(
    release_workflow: &str,
    native_sdk_artifact_workflow: &str,
    swift_sdk_artifact_workflow: &str,
) -> DynResult<()> {
    const SOURCE_BUILD_JOBS: &[&str] = &["build", "build_linux_arm64", "windows_host_input"];
    const REQUIRED_STEP: &str = "Prepare dispatched release version";
    const REQUIRED_COMMAND: &str = "scripts/release-version.sh \"$RELEASE_TAG\"";

    for job_name in SOURCE_BUILD_JOBS {
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

    for job_name in RELEASE_COMPOSITION_ONLY_JOBS {
        let job = workflow_job_section(release_workflow, job_name).ok_or_else(|| {
            format!("release workflow: missing `{job_name}` job for composition-only check")
        })?;
        ensure_not_contains(
            job,
            REQUIRED_STEP,
            &format!("release workflow `{job_name}` composition-only version step"),
        )?;
        ensure_not_contains(
            job,
            REQUIRED_COMMAND,
            &format!("release workflow `{job_name}` composition-only version command"),
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
    check_main_plan_test_coverage(repo_root)
}

fn check_main_plan_test_coverage(repo_root: &Path) -> DynResult<()> {
    let input = r#"{"profile":"main","event_name":"push","source_sha":"0000000000000000000000000000000000000000","base_sha":"","changed_files":[]}"#;
    let mut child = Command::new("python3")
        .current_dir(repo_root)
        .args(["scripts/plan-ci.py"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or("failed to open planner stdin")?
        .write_all(input.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(format!(
            "main CI planner failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }

    let plan: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let batches = plan
        .get("matrices")
        .and_then(|matrices| matrices.get("rust_tests"))
        .ok_or("main CI plan is missing matrices.rust_tests")?;
    let mut actual = std::collections::BTreeSet::new();
    for crate_name in batches
        .as_array()
        .ok_or("main CI rust-test matrix must be an array")?
        .iter()
        .flat_map(|batch| batch["crates"].as_array().into_iter().flatten())
    {
        let crate_name = crate_name
            .as_str()
            .ok_or("main CI rust-test crate names must be strings")?;
        if !actual.insert(crate_name.to_owned()) {
            return Err(format!("main CI rust-test matrix duplicated crate `{crate_name}`").into());
        }
    }

    let expected = workspace_package_names(repo_root)?;
    ensure_set_eq(&expected, &actual, "main CI rust-test workspace coverage")
}

pub(crate) fn check_ci_script_workspace_members(repo_root: &Path) -> DynResult<()> {
    let expected = workspace_package_names(repo_root)?;
    let scripts = ["scripts/affected-crates.sh"];

    for script in scripts {
        let actual = script_workspace_members(repo_root, script)?;
        ensure_set_eq(&expected, &actual, &format!("{script} WORKSPACE_MEMBERS"))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_release_container_contracts, check_release_sccache_initialization};

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

    const VALID_SCCACHE_INITIALIZATION_WORKFLOW: &str = r#"jobs:
  metadata:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v5
      - uses: mozilla-actions/sccache-action@v0
      - uses: ./.github/actions/configure-sccache-gha
      - name: Prepare canonical release source
        run: |
          scripts/release-version.sh "$RELEASE_TAG"
          cargo fmt --all -- --check
  publish:
    runs-on: ubuntu-24.04
    steps:
      - uses: mozilla-actions/sccache-action@v0
      - uses: ./.github/actions/configure-sccache-gha
      - name: Prepare dispatched release tag
        run: scripts/release-version.sh "$RELEASE_TAG"
  build_native_runtime_linux_aarch64_cuda:
    runs-on: ubuntu-24.04
    container:
      image: ghcr.io/example/runner
    steps:
      - uses: ./.github/actions/configure-sccache-gha
      - name: Build native runtime
        run: cargo build -p skippy-runtime
  windows_host_input:
    runs-on: windows-2025
    steps:
      - uses: mozilla-actions/sccache-action@v0
      - name: Build host input
        run: cargo build -p mesh-llm
  compose_linux_cuda:
    runs-on: ubuntu-24.04
    steps:
      - uses: ./.github/actions/compose-product-input
  release_notes:
    runs-on: ubuntu-24.04
    steps:
      - name: Regroup published release notes
        # The script decides what the release it just published needs.
        run: scripts/release-notes-generate.sh
"#;

    #[test]
    fn release_sccache_initialization_accepts_supported_setup_variants() {
        check_release_sccache_initialization(VALID_SCCACHE_INITIALIZATION_WORKFLOW).unwrap();
    }

    #[test]
    fn release_sccache_initialization_rejects_swapped_hosted_setup() {
        let workflow = VALID_SCCACHE_INITIALIZATION_WORKFLOW.replace(
            "      - uses: mozilla-actions/sccache-action@v0\n      - uses: ./.github/actions/configure-sccache-gha\n",
            "      - uses: ./.github/actions/configure-sccache-gha\n      - uses: mozilla-actions/sccache-action@v0\n",
        );

        let error = check_release_sccache_initialization(&workflow).unwrap_err();
        assert!(
            error.to_string().contains(
                "`metadata` runs `uses: ./.github/actions/configure-sccache-gha` before `uses: mozilla-actions/sccache-action`"
            ),
            "{error}"
        );
    }

    #[test]
    fn release_sccache_initialization_requires_both_setup_steps() {
        let workflow = VALID_SCCACHE_INITIALIZATION_WORKFLOW.replace(
            "      - uses: ./.github/actions/configure-sccache-gha\n",
            "",
        );

        let error = check_release_sccache_initialization(&workflow).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("`metadata` invokes cargo, scripts/release-version.sh before required sccache initialization `uses: ./.github/actions/configure-sccache-gha`"),
            "{error}"
        );
    }

    #[test]
    fn release_sccache_initialization_requires_the_hosted_server_step() {
        let workflow = VALID_SCCACHE_INITIALIZATION_WORKFLOW
            .replace("      - uses: mozilla-actions/sccache-action@v0\n", "");

        let error = check_release_sccache_initialization(&workflow).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("`metadata` invokes cargo, scripts/release-version.sh before required sccache initialization `uses: mozilla-actions/sccache-action`"),
            "{error}"
        );
    }

    #[test]
    fn release_sccache_initialization_rejects_setup_after_cargo() {
        let workflow = VALID_SCCACHE_INITIALIZATION_WORKFLOW.replace(
            "      - uses: mozilla-actions/sccache-action@v0\n      - uses: ./.github/actions/configure-sccache-gha\n",
            "",
        ).replace(
            "          cargo fmt --all -- --check\n",
            "          cargo fmt --all -- --check\n      - uses: mozilla-actions/sccache-action@v0\n      - uses: ./.github/actions/configure-sccache-gha\n",
        );

        let error = check_release_sccache_initialization(&workflow).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("`metadata` invokes cargo, scripts/release-version.sh before required sccache initialization"),
            "{error}"
        );
    }

    #[test]
    fn release_sccache_initialization_rejects_cargo_in_composition_only_jobs() {
        let workflow = VALID_SCCACHE_INITIALIZATION_WORKFLOW.replace(
            "      - uses: ./.github/actions/compose-product-input\n",
            "      - uses: ./.github/actions/compose-product-input\n      - name: Prepare dispatched release version\n        run: scripts/release-version.sh \"$RELEASE_TAG\"\n",
        );

        let error = check_release_sccache_initialization(&workflow).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("`compose_linux_cuda` is composition-only but invokes"),
            "{error}"
        );
    }

    #[test]
    fn release_sccache_initialization_tracks_package_release_cargo() {
        let workflow = VALID_SCCACHE_INITIALIZATION_WORKFLOW.replace(
            "      - uses: ./.github/actions/compose-product-input\n",
            "      - uses: ./.github/actions/compose-product-input\n      - name: Package release\n        run: scripts/package-release.sh dist/product\n",
        );

        let error = check_release_sccache_initialization(&workflow).unwrap_err();
        assert!(
            error.to_string().contains(
                "`compose_linux_cuda` runs scripts/package-release.sh without the cargo-bypass guard"
            ),
            "{error}"
        );
    }

    #[test]
    fn release_sccache_initialization_scans_past_top_level_comments() {
        let workflow = VALID_SCCACHE_INITIALIZATION_WORKFLOW
            .replace("  publish:\n", "# release publication follows\n  publish:\n")
            .replace(
                "      - uses: mozilla-actions/sccache-action@v0\n      - uses: ./.github/actions/configure-sccache-gha\n      - name: Prepare dispatched release tag",
                "      - name: Prepare dispatched release tag",
            );

        let error = check_release_sccache_initialization(&workflow).unwrap_err();
        assert!(error.to_string().contains("`publish` invokes"), "{error}");
    }
}
