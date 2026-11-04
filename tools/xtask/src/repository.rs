mod affected_crates;
mod bundled_plugins;
pub(crate) mod cargo_packages;
mod cargo_projection;
pub(crate) mod check_args;
pub(crate) mod check_report;
mod docker_precheck;
mod env_census;
mod publish_order;
pub(crate) mod publish_roster;
mod selected_ref;
pub(crate) mod text;
mod upstream_release;

use crate::cli::RepositoryCheck;
use crate::command::DynResult;
use ci_controller::repository::{CheckRoots, conventional_commit, upstream_pin};

/// Runs a ported check with the shared `CheckRoots` discovery semantics.
pub(crate) fn run_check(
    check: RepositoryCheck,
    args: &[String],
    explicit_root: Option<RepositoryRoot>,
) -> DynResult<()> {
    let roots = CheckRoots::new(explicit_root)?;
    let cwd = roots.cwd().to_path_buf();
    let default_root = roots.default_root();
    match check {
        RepositoryCheck::CargoTargetDirectory => cargo_projection::run(args),
        RepositoryCheck::PublishOrder => publish_order::run(args),
        RepositoryCheck::AffectedCrates => affected_crates::run(&cwd, args),
        RepositoryCheck::CargoPackages => cargo_packages::run(&cwd, args),
        RepositoryCheck::ConventionalCommits => conventional_commit::run(&cwd, args),
        RepositoryCheck::EnvMutationCensus => env_census::run(args, default_root),
        RepositoryCheck::LlamaUpstreamPin => upstream_pin::run(args, default_root),
        RepositoryCheck::LlamaUpstreamRelease => upstream_release::run(args),
        RepositoryCheck::SelectedRef => selected_ref::run(args, default_root),
        RepositoryCheck::DockerPrecheck => docker_precheck::run(args, default_root),
        RepositoryCheck::BundledPlugins => bundled_plugins::list(args, default_root),
        RepositoryCheck::BundledPluginsCheck => bundled_plugins::bundle_check(args),
    }
}

pub(crate) use ci_controller::repository::RepositoryRoot;
