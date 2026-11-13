use ci_controller::repository::{RepositoryRoot, parse_root_prefix};
use ci_controller::{DynResult, bootstrap};
use runner_cleanup::bootstrap::RUNNER_CLEANUP;

const USAGE: &str = "usage: runner-cleanup [--repo-root DIR] <command> ...
  ci-ops runner-cleanup ...
  automation bootstrap";

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> DynResult<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (root, command) = parse_root_prefix(&args)?;
    let (bootstrap, rest) = match command {
        [domain, verb, rest @ ..] if domain == "ci-ops" && verb == "runner-cleanup" => {
            (false, rest)
        }
        [domain, verb, rest @ ..] if domain == "automation" && verb == "bootstrap" => (true, rest),
        _ => usage(),
    };
    // xtask resolves an explicit root before dispatch, so an invalid one fails the same way.
    let explicit_root = root
        .as_deref()
        .map(|path| RepositoryRoot::resolve(Some(path)))
        .transpose()?;
    if !bootstrap {
        return runner_cleanup::command::run(rest);
    }
    let root = match explicit_root {
        Some(root) => root,
        None => RepositoryRoot::resolve(None)?,
    };
    bootstrap::run(root.as_path(), rest, &RUNNER_CLEANUP)
}

fn usage() -> ! {
    eprintln!("{USAGE}");
    std::process::exit(2);
}
