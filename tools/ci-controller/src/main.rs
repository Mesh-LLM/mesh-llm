use ci_controller::repository::{
    CheckRoots, RepositoryRoot, affected_crates, conventional_commit, parse_root_prefix,
    upstream_pin,
};
use ci_controller::{DynError, DynResult, bootstrap, lane_results, plan};

const USAGE: &str = "usage: ci-controller [--repo-root DIR] <command> ...
  ci plan [--manifest-root MANIFEST_ROOT]
  ci validate-lane ...
  repository affected-crates ...
  repository conventional-commits ...
  repository llama-upstream-pin ...
  automation bootstrap";

enum Command<'a> {
    Plan(&'a [String]),
    ValidateLane(&'a [String]),
    AffectedCrates(&'a [String]),
    ConventionalCommits(&'a [String]),
    LlamaUpstreamPin(&'a [String]),
    Bootstrap(&'a [String]),
}

impl<'a> Command<'a> {
    fn parse(command: &'a [String]) -> Option<Self> {
        let [domain, verb, rest @ ..] = command else {
            return None;
        };
        Some(match (domain.as_str(), verb.as_str()) {
            ("ci", "plan") => Self::Plan(rest),
            ("ci", "validate-lane") => Self::ValidateLane(rest),
            ("repository", "affected-crates") => Self::AffectedCrates(rest),
            ("repository", "conventional-commits") => Self::ConventionalCommits(rest),
            ("repository", "llama-upstream-pin") => Self::LlamaUpstreamPin(rest),
            ("automation", "bootstrap") => Self::Bootstrap(rest),
            _ => return None,
        })
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> DynResult<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (root, command) = parse_root_prefix(&args)?;
    let Some(command) = Command::parse(command) else {
        usage();
    };
    let explicit_root = root
        .as_deref()
        .map(|path| RepositoryRoot::resolve(Some(path)))
        .transpose()?;
    let checkout = |explicit: Option<RepositoryRoot>| -> Result<RepositoryRoot, DynError> {
        match explicit {
            Some(root) => Ok(root),
            None => RepositoryRoot::resolve(None),
        }
    };
    match command {
        Command::Plan(rest) => plan::run(checkout(explicit_root)?.as_path(), rest),
        Command::ValidateLane(rest) => lane_results::run(rest),
        Command::Bootstrap(rest) => bootstrap::run(
            checkout(explicit_root)?.as_path(),
            rest,
            &bootstrap::CONTROLLER,
        ),
        Command::AffectedCrates(rest) => {
            affected_crates::run(CheckRoots::new(explicit_root)?.cwd(), rest)
        }
        Command::ConventionalCommits(rest) => {
            conventional_commit::run(CheckRoots::new(explicit_root)?.cwd(), rest)
        }
        Command::LlamaUpstreamPin(rest) => {
            upstream_pin::run(rest, CheckRoots::new(explicit_root)?.default_root())
        }
    }
}

fn usage() -> ! {
    eprintln!("{USAGE}");
    std::process::exit(2);
}
