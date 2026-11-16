use authority_automation::bootstrap::AUTHORITY_AUTOMATION;
use authority_automation::ci_operations::{authority_command, pr_authority};
use authority_automation::file_projection;
use authority_automation::model_registry::{resolve, restore_inputs};
use ci_controller::repository::check_report::CheckReport;
use ci_controller::repository::{RepositoryRoot, parse_root_prefix};
use ci_controller::{DynResult, bootstrap};

const USAGE: &str = "usage: authority-automation [--repo-root DIR] <command> ...
  models resolve ...
  models restore-inputs ...
  ci-ops authority-audit ...
  ci-ops pr-authority-audit
  artifact file-projection ...
  automation bootstrap";

enum Command<'a> {
    Report(fn(&[String]) -> CheckReport, &'a [String]),
    Bootstrap(&'a [String]),
}

impl<'a> Command<'a> {
    fn parse(command: &'a [String]) -> Option<Self> {
        let [domain, verb, rest @ ..] = command else {
            return None;
        };
        let report: fn(&[String]) -> CheckReport = match (domain.as_str(), verb.as_str()) {
            ("models", "resolve") => resolve::run,
            ("models", "restore-inputs") => restore_inputs::run,
            ("ci-ops", "authority-audit") => authority_command::run,
            ("ci-ops", "pr-authority-audit") => pr_authority::run,
            ("artifact", "file-projection") => file_projection::run,
            ("automation", "bootstrap") => return Some(Self::Bootstrap(rest)),
            _ => return None,
        };
        Some(Self::Report(report, rest))
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
    match command {
        Command::Report(report, rest) => report(rest).emit(),
        Command::Bootstrap(rest) => {
            let root = match explicit_root {
                Some(root) => root,
                None => RepositoryRoot::resolve(None)?,
            };
            bootstrap::run(root.as_path(), rest, &AUTHORITY_AUTOMATION)
        }
    }
}

fn usage() -> ! {
    eprintln!("{USAGE}");
    std::process::exit(2);
}
