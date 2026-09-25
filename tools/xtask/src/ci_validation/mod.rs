//! Repository CI documentation and workflow contract validation.

mod crate_coverage;
mod documentation;
mod entrypoints;
pub(crate) mod lane_results;
mod producers;
mod release_containers;
mod release_dispatch;
mod windows_runtime;

use crate::command::DynResult;
use std::fs;
use std::path::Path;

pub(crate) use crate_coverage::{
    check_ci_crate_test_coverage_files, check_ci_script_workspace_members,
};
use documentation::check_documentation_invariants;
use entrypoints::{check_orchestrator_invariants, check_workflow_invariants};
use producers::{ProducerInvariantSources, check_producer_invariants};
use release_containers::check_release_container_contracts;
use release_dispatch::check_release_dispatch_version_preparation;
use windows_runtime::check_windows_dynamic_runtime_contract;

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
    check_release_container_contracts(&release_workflow, &configure_sccache)?;
    check_windows_dynamic_runtime_contract(
        &host,
        &runtime_and_product,
        &prepare_windows_host,
        &prepare_runtime,
        &compose_product,
    )
}
