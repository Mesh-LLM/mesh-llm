use crate::command::{
    DynResult, ensure_contains, ensure_contains_normalized, ensure_not_contains, ensure_set_eq,
    workflow_job_section,
};
use crate::repo_consistency::{script_workspace_members, workspace_package_names};
use std::fs;
use std::path::Path;

pub(crate) fn check_docs_and_workflow_invariants(repo_root: &Path) -> DynResult<()> {
    let readme = fs::read_to_string(repo_root.join("README.md"))?;
    let contributing = fs::read_to_string(repo_root.join("CONTRIBUTING.md"))?;
    let release = fs::read_to_string(repo_root.join("RELEASE.md"))?;
    let justfile = fs::read_to_string(repo_root.join("Justfile"))?;
    let release_workflow = fs::read_to_string(repo_root.join(".github/workflows/release.yml"))?;
    let ci_workflow = fs::read_to_string(repo_root.join(".github/workflows/ci.yml"))?;
    let pr_builds_workflow = fs::read_to_string(repo_root.join(".github/workflows/pr_builds.yml"))?;
    let pr_quality_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/pr_quality.yml"))?;
    let pr_website_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/pr_website.yml"))?;
    let website_pages_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/website-pages.yml"))?;
    let compute_changes_action =
        fs::read_to_string(repo_root.join(".github/actions/compute-changes/action.yml"))?;
    let affected_crates_script = fs::read_to_string(repo_root.join("scripts/affected-crates.sh"))?;
    let ci_docs = fs::read_to_string(repo_root.join("ci/ci.md"))?;
    let pr_cleanup_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/pr_cleanup.yml"))?;
    let windows_warm_caches_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/windows-warm-caches.yml"))?;

    ensure_contains(
        &readme,
        "mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
        "README Linux ARM64 asset note",
    )?;
    ensure_contains(
        &readme,
        "mesh-llm-aarch64-unknown-linux-gnu-cuda.tar.gz",
        "README Linux ARM64 CUDA asset note",
    )?;
    ensure_contains(
        &release,
        "mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
        "RELEASE Linux ARM64 asset note",
    )?;
    ensure_contains(
        &release,
        "mesh-llm-aarch64-unknown-linux-gnu-cuda.tar.gz",
        "RELEASE Linux ARM64 CUDA asset note",
    )?;
    ensure_contains_normalized(
        &readme,
        "Windows CPU, Windows CUDA, Windows ROCm, and Windows Vulkan bundles",
        "README Windows publish note",
    )?;
    ensure_contains(
        &release,
        "Windows release artifacts use the `x86_64-pc-windows-msvc` target triple",
        "RELEASE Windows publish note",
    )?;
    ensure_contains(
        &release_workflow,
        "runs-on: ubuntu-24.04-arm",
        "release workflow ARM64 runner",
    )?;
    ensure_contains(
        &release_workflow,
        "name: release-linux-arm64",
        "release workflow ARM64 artifact",
    )?;
    ensure_contains(
        &release_workflow,
        "name: release-linux-aarch64-cuda-${{ matrix.cuda_version }}",
        "release workflow aarch64 CUDA artifact (matrix)",
    )?;
    ensure_contains(
        &release_workflow,
        "- build_linux_aarch64_cuda",
        "release workflow aarch64 CUDA publish need",
    )?;
    ensure_contains(
        &release_workflow,
        "build_windows_cpu:",
        "release workflow Windows CPU build",
    )?;
    ensure_contains(
        &release_workflow,
        "build_windows_gpu:",
        "release workflow Windows GPU build",
    )?;
    ensure_contains(
        &release_workflow,
        "- build_windows_cpu",
        "release workflow Windows CPU publish need",
    )?;
    ensure_contains(
        &release_workflow,
        "- build_windows_gpu",
        "release workflow Windows GPU publish need",
    )?;
    ensure_contains(
        &justfile,
        "check-release:",
        "Justfile release consistency wrapper",
    )?;
    ensure_contains(
        &justfile,
        "release-build-aarch64-cuda",
        "Justfile aarch64 CUDA build recipe",
    )?;
    ensure_contains(
        &justfile,
        "release-bundle-aarch64-cuda",
        "Justfile aarch64 CUDA bundle recipe",
    )?;
    ensure_contains(
        &justfile,
        "cargo run -p xtask -- repo-consistency release-targets",
        "Justfile xtask command",
    )?;
    ensure_contains(
        &contributing,
        "just check-release",
        "CONTRIBUTING release consistency command",
    )?;
    ensure_contains(
        &contributing,
        "On native Windows, `just check-release` runs the host-safe Rust/doc invariant subset and skips the Bash-only `install.sh` / `package-release.sh` parity checks",
        "CONTRIBUTING Windows check-release note",
    )?;
    ensure_contains(
        &release,
        "On native Windows, `just check-release` still runs the Rust/docs/workflow invariant checks, but it skips the Bash-only `install.sh` and `scripts/package-release.sh` parity checks",
        "RELEASE Windows check-release note",
    )?;
    ensure_contains(
        &pr_builds_workflow,
        "cargo run -p xtask -- repo-consistency release-targets",
        "PR Builds xtask release-target check",
    )?;
    ensure_contains(
        &pr_quality_workflow,
        "name: PR Quality Checks",
        "PR quality workflow display name",
    )?;
    ensure_contains(
        &pr_quality_workflow,
        "cargo run -p xtask -- repo-consistency ci-crate-lists",
        "PR quality CI crate-list drift check",
    )?;
    ensure_not_contains(
        &pr_quality_workflow,
        "website-build:",
        "PR quality should not own public website builds",
    )?;
    ensure_contains(
        &compute_changes_action,
        "website_changed",
        "compute-changes public website change output",
    )?;
    ensure_contains(
        &compute_changes_action,
        "website_docs_changed",
        "compute-changes public website docs output",
    )?;
    ensure_contains(
        &compute_changes_action,
        "cli_surface_changed",
        "compute-changes CLI surface output",
    )?;
    ensure_contains(
        &compute_changes_action,
        "inference_artifact_required",
        "compute-changes inference artifact output",
    )?;
    ensure_contains(
        &compute_changes_action,
        "backend_recipe_changed",
        "compute-changes backend Justfile recipe output",
    )?;
    ensure_contains(
        &compute_changes_action,
        "windows_cpu_build_required",
        "compute-changes Windows CPU build output",
    )?;
    ensure_contains(
        &compute_changes_action,
        "windows_gpu_build_required",
        "compute-changes Windows GPU build output",
    )?;
    ensure_contains(
        &compute_changes_action,
        "build-linux-rocm",
        "compute-changes Linux ROCm build script route",
    )?;
    ensure_contains(
        &affected_crates_script,
        "is_website_input",
        "affected-crates public website input classifier",
    )?;
    ensure_contains(
        &pr_website_workflow,
        "name: PR Website Checks",
        "PR website workflow display name",
    )?;
    ensure_contains(
        &pr_website_workflow,
        "./.github/actions/compute-changes",
        "PR website compute-changes route",
    )?;
    ensure_contains(
        &pr_website_workflow,
        "website_changed",
        "PR website public website change gate",
    )?;
    ensure_contains(
        &pr_website_workflow,
        "website-build:",
        "PR website public website build gate",
    )?;
    ensure_contains(
        &pr_website_workflow,
        "npm run build",
        "PR website public website build command",
    )?;
    ensure_contains(
        &pr_website_workflow,
        "PR Website Checks",
        "PR website Markdown summary output",
    )?;
    ensure_contains(
        &pr_quality_workflow,
        "cli-docs-sync:",
        "PR quality CLI docs sync gate",
    )?;
    ensure_contains(
        &pr_quality_workflow,
        "GITHUB_STEP_SUMMARY",
        "PR quality Markdown summary output",
    )?;
    ensure_contains(
        &pr_builds_workflow,
        "website_changed",
        "PR Builds public website change output",
    )?;
    ensure_contains(
        &pr_builds_workflow,
        "inference_artifact_required",
        "PR Builds inference artifact gate",
    )?;
    ensure_contains(
        &pr_builds_workflow,
        "backend_recipe_changed",
        "PR Builds backend recipe route",
    )?;
    ensure_contains(
        &pr_builds_workflow,
        "steps.compute.outputs.windows_cpu_build_required",
        "PR Builds Windows CPU compute route",
    )?;
    ensure_contains(
        &pr_builds_workflow,
        "steps.compute.outputs.windows_gpu_build_required",
        "PR Builds Windows GPU compute route",
    )?;
    ensure_contains(
        &ci_docs,
        "website_changed?",
        "CI topology public website route",
    )?;
    ensure_contains(
        &ci_docs,
        "inference_artifact_required?",
        "CI topology inference artifact route",
    )?;
    ensure_contains(
        &ci_docs,
        "backend_recipe_changed?",
        "CI topology backend Justfile recipe route",
    )?;
    ensure_contains(
        &ci_docs,
        "windows_cpu_build_required?",
        "CI topology Windows CPU compute route",
    )?;
    ensure_contains(
        &ci_docs,
        "windows_gpu_build_required?",
        "CI topology Windows GPU compute route",
    )?;
    ensure_contains(&ci_docs, "cli-docs-sync", "CI topology CLI docs sync gate")?;
    ensure_contains(
        &ci_docs,
        "pr_website.yml",
        "CI topology PR website workflow",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "name: Public Website Deploy",
        "public website deploy workflow name",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "branches: [main]",
        "public website deploy main trigger",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "workflow_dispatch:",
        "public website manual deploy trigger",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "github.event_name != 'workflow_dispatch' || github.ref == 'refs/heads/main'",
        "public website manual deploy main-ref guard",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "npm run clean",
        "public website clean generated output step",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "public-website-artifact",
        "public website staged artifact directory",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "path: public-website-artifact",
        "public website staged Pages artifact upload",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "actions/upload-pages-artifact@v3",
        "public website Pages artifact upload",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "actions/deploy-pages@v4",
        "public website Pages deploy action",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "pages: write",
        "public website deploy Pages permission",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "id-token: write",
        "public website deploy OIDC permission",
    )?;
    ensure_contains(
        &website_pages_workflow,
        "name: Public Website",
        "public website custom environment",
    )?;
    ensure_contains(
        &ci_docs,
        "website-pages.yml",
        "CI topology public website deploy workflow",
    )?;
    ensure_contains(
        &pr_cleanup_workflow,
        "pull_request_target:",
        "PR cache cleanup trigger",
    )?;
    ensure_contains(
        &ci_workflow,
        "push:\n    branches: [main]",
        "main CI push trigger",
    )?;
    check_windows_abi_cache_key_alignment(
        &ci_workflow,
        &pr_builds_workflow,
        &windows_warm_caches_workflow,
    )?;
    check_release_dispatch_version_preparation(&release_workflow)?;
    check_release_container_safe_directories(&release_workflow)?;
    check_ci_crate_test_coverage(&pr_builds_workflow)?;

    Ok(())
}

fn check_release_dispatch_version_preparation(release_workflow: &str) -> DynResult<()> {
    const DISPATCH_RELEASE_JOBS: &[&str] = &[
        "build",
        "build_native_sdk_runtime",
        "build_swift_sdk_artifact",
        "build_linux_arm64",
        "build_linux_aarch64_cuda",
        "build_linux_cuda",
        "build_linux_rocm",
        "build_linux_vulkan",
        "build_windows_cpu",
        "build_windows_gpu",
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

    Ok(())
}

fn check_release_container_safe_directories(release_workflow: &str) -> DynResult<()> {
    const CONTAINER_RELEASE_JOBS: &[&str] = &[
        "build_linux_aarch64_cuda",
        "build_linux_cuda",
        "build_linux_rocm",
    ];
    const REQUIRED_STEP: &str = "Trust checkout directory";
    const REQUIRED_COMMAND: &str = "git config --global --add safe.directory \"$GITHUB_WORKSPACE\"";

    for job_name in CONTAINER_RELEASE_JOBS {
        let job = workflow_job_section(release_workflow, job_name).ok_or_else(|| {
            format!("release workflow: missing `{job_name}` job for container safe-directory check")
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
    }

    Ok(())
}

fn check_windows_abi_cache_key_alignment(
    ci_workflow: &str,
    pr_builds_workflow: &str,
    windows_warm_caches_workflow: &str,
) -> DynResult<()> {
    const WINDOWS_ABI_CACHE_HASH_INPUTS: &str = concat!(
        "hashFiles('scripts/build-windows.ps1', 'scripts/install-windows-sdk.ps1', ",
        "'.github/actions/setup-windows-rocm-sdk/action.yml', ",
        "'third_party/llama.cpp/upstream.txt', 'third_party/llama.cpp/patches/**', ",
        "'Justfile', '.github/cache-version.txt')",
    );
    let windows_cpu_abi_cache_key =
        format!("windows-2022-skippy-abi-cpu--cpu-${{{{ {WINDOWS_ABI_CACHE_HASH_INPUTS} }}}}");

    ensure_contains(
        ci_workflow,
        &windows_cpu_abi_cache_key,
        "main CI Windows CPU ABI cache key",
    )?;
    ensure_contains(
        windows_warm_caches_workflow,
        &windows_cpu_abi_cache_key,
        "Windows warm-cache CPU ABI cache key",
    )?;
    ensure_contains(
        pr_builds_workflow,
        "windows-2022-skippy-abi-${{ matrix.backend }}-${{ matrix.build_args }}-",
        "PR Builds Windows ABI cache key template",
    )?;
    ensure_contains(
        pr_builds_workflow,
        "|| 'cpu' }}-${{ hashFiles(",
        "PR Builds Windows CPU ABI cache discriminator",
    )?;
    ensure_contains(
        pr_builds_workflow,
        WINDOWS_ABI_CACHE_HASH_INPUTS,
        "PR Builds Windows ABI cache hash inputs",
    )?;

    Ok(())
}

fn check_ci_crate_test_coverage(ci_workflow: &str) -> DynResult<()> {
    const REQUIRED_TEST_CRATES: &[(&str, &str)] = &[
        ("mesh-llm-client", "mesh client crate tests"),
        ("mesh-llm-api-client", "mesh LLM client API crate tests"),
        ("mesh-llm-api-server", "mesh LLM API crate tests"),
        ("mesh-llm-config", "mesh LLM config crate tests"),
        ("mesh-llm-commands", "mesh LLM command handler crate tests"),
        ("mesh-llm-events", "mesh LLM events crate tests"),
        (
            "mesh-llm-hardware-profile",
            "mesh LLM hardware profile crate tests",
        ),
        (
            "mesh-llm-runtime-install",
            "mesh LLM runtime install crate tests",
        ),
        (
            "mesh-llm-native-runtime",
            "mesh LLM native runtime crate tests",
        ),
        ("mesh-llm-routing", "mesh LLM routing crate tests"),
        ("mesh-llm-types", "mesh LLM shared types crate tests"),
        ("mesh-llm-cli", "mesh LLM CLI crate tests"),
        ("mesh-llm-tui", "mesh LLM TUI crate tests"),
        (
            "mesh-llm-embedded-runtime",
            "mesh LLM embedded runtime crate tests",
        ),
        ("mesh-llm-sdk", "mesh LLM Rust SDK crate tests"),
        (
            "mesh-llm-console-server",
            "mesh LLM console server crate tests",
        ),
        ("mesh-llm-ffi", "mesh LLM FFI crate tests"),
        ("mesh-llm-nodejs", "mesh LLM Node.js crate tests"),
        ("skippy-protocol", "skippy protocol crate tests"),
        ("skippy-server", "skippy server crate tests"),
        ("openai-frontend", "OpenAI frontend crate tests"),
        ("skippy-runtime", "skippy runtime crate tests"),
        ("skippy-topology", "skippy topology crate tests"),
        ("skippy-model-package", "skippy model-package crate tests"),
        ("skippy-prompt", "skippy prompt crate tests"),
        ("metrics-server", "metrics server crate tests"),
    ];
    const LIB_ONLY_CRATE_PATTERN: &str = "skippy-protocol|skippy-server|openai-frontend)";

    ensure_contains(
        ci_workflow,
        "cargo test -p \"$c\"",
        "CI dynamic crate test command",
    )?;
    ensure_contains(
        ci_workflow,
        "for c in mesh-llm-client mesh-llm-api-client mesh-llm-api-server mesh-llm-config mesh-llm-commands mesh-llm-events mesh-llm-hardware-profile mesh-llm-runtime-install mesh-llm-native-runtime mesh-llm-routing mesh-llm-types mesh-llm-cli mesh-llm-tui mesh-llm-embedded-runtime mesh-llm-sdk mesh-llm-console-server mesh-llm-ffi mesh-llm-nodejs; do",
        "CI SDK/API crate test loop",
    )?;
    ensure_contains(
        ci_workflow,
        "for c in skippy-protocol skippy-server openai-frontend skippy-runtime skippy-topology skippy-model-package skippy-prompt metrics-server; do",
        "CI Skippy crate test loop",
    )?;
    ensure_contains(
        ci_workflow,
        LIB_ONLY_CRATE_PATTERN,
        "CI lib-only crate test flag selector",
    )?;
    ensure_contains(ci_workflow, "--lib", "CI lib-only crate test flag")?;

    for (crate_name, context) in REQUIRED_TEST_CRATES {
        ensure_contains(ci_workflow, crate_name, &format!("CI {context}"))?;
    }

    Ok(())
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
