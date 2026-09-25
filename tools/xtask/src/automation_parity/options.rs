//! `automation parity` arguments. The legacy interpreter comes only from an
//! explicit flag or the documented opt-in variable; `--rust-only` compares
//! Rust with the frozen goldens and never starts an interpreter.

use crate::command::DynResult;
use std::path::PathBuf;

pub(super) const INTERPRETER_ENV: &str = "MIGRATION_CI_PLAN_LEGACY_PYTHON";
pub(super) const BASH_ENV: &str = "MIGRATION_CI_PLAN_LEGACY_BASH";

pub(super) const USAGE: &str = "usage: cargo xtool automation parity --suite ci [--evidence <dir>] \
[--interpreter <path> | --rust-only] [--bash <path>] [--planner <path>] [--fixtures <dir>] \
[--source-repo <dir>] [--source-sha <sha>]";

/// Which side(s) run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Mode {
    /// Rust against the frozen goldens and recorded action outputs only.
    RustOnly,
    /// Also run the legacy planner with this interpreter.
    Legacy { interpreter: PathBuf },
}

#[derive(Debug)]
pub(super) struct Options {
    pub(super) mode: Mode,
    pub(super) bash: Option<PathBuf>,
    pub(super) evidence: Option<PathBuf>,
    pub(super) planner: Option<PathBuf>,
    pub(super) fixtures: Option<PathBuf>,
    pub(super) source_repo: Option<PathBuf>,
    pub(super) source_sha: Option<String>,
}

#[derive(Default)]
struct Raw {
    suite: Option<String>,
    rust_only: bool,
    interpreter: Option<PathBuf>,
    bash: Option<PathBuf>,
    evidence: Option<PathBuf>,
    planner: Option<PathBuf>,
    fixtures: Option<PathBuf>,
    source_repo: Option<PathBuf>,
    source_sha: Option<String>,
}

fn value<'a>(rest: &mut impl Iterator<Item = &'a String>, flag: &str) -> DynResult<String> {
    rest.next()
        .filter(|value| !value.starts_with("--"))
        .cloned()
        .ok_or_else(|| format!("missing value for {flag}\n{USAGE}").into())
}

fn parse_raw(args: &[String]) -> DynResult<Raw> {
    let mut raw = Raw::default();
    let mut rest = args.iter();
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--rust-only" => raw.rust_only = true,
            "--suite" => raw.suite = Some(value(&mut rest, flag)?),
            "--interpreter" => raw.interpreter = Some(value(&mut rest, flag)?.into()),
            "--bash" => raw.bash = Some(value(&mut rest, flag)?.into()),
            "--evidence" => raw.evidence = Some(value(&mut rest, flag)?.into()),
            "--planner" => raw.planner = Some(value(&mut rest, flag)?.into()),
            "--fixtures" => raw.fixtures = Some(value(&mut rest, flag)?.into()),
            "--source-repo" => raw.source_repo = Some(value(&mut rest, flag)?.into()),
            "--source-sha" => raw.source_sha = Some(value(&mut rest, flag)?),
            other => return Err(format!("unknown argument {other}\n{USAGE}").into()),
        }
    }
    Ok(raw)
}

fn nonempty_env(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn mode(raw: &Raw) -> DynResult<Mode> {
    if raw.rust_only && raw.interpreter.is_some() {
        return Err(format!("--rust-only and --interpreter are exclusive\n{USAGE}").into());
    }
    if raw.rust_only {
        return Ok(Mode::RustOnly);
    }
    match raw
        .interpreter
        .clone()
        .or_else(|| nonempty_env(INTERPRETER_ENV))
    {
        Some(interpreter) => Ok(Mode::Legacy { interpreter }),
        None => Err(format!(
            "no legacy interpreter: pass --interpreter <path> or set {INTERPRETER_ENV} \
             (a 3.13 interpreter; the macOS system one is too old), and {BASH_ENV} or --bash \
             for a bash 5; or pass --rust-only to compare Rust with the frozen goldens only"
        )
        .into()),
    }
}

impl Options {
    pub(super) fn parse(args: &[String]) -> DynResult<Self> {
        let raw = parse_raw(args)?;
        if raw.suite.as_deref() != Some("ci") {
            return Err(format!("only --suite ci is supported\n{USAGE}").into());
        }
        let mode = mode(&raw)?;
        Ok(Self {
            mode,
            bash: raw.bash.or_else(|| nonempty_env(BASH_ENV)),
            evidence: raw.evidence,
            planner: raw.planner,
            fixtures: raw.fixtures,
            source_repo: raw.source_repo,
            source_sha: raw.source_sha,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn migration_ci_shadow_options_prefer_the_explicit_interpreter() {
        let parsed = Options::parse(&strings(&["--suite", "ci", "--interpreter", "/x/py"]));
        let mode = parsed.map(|options| options.mode).ok();
        assert_eq!(
            mode,
            Some(Mode::Legacy {
                interpreter: "/x/py".into()
            })
        );
    }

    #[test]
    fn migration_ci_shadow_options_reject_conflicts_and_unknown_suites() {
        let conflict = strings(&["--suite", "ci", "--rust-only", "--interpreter", "/x"]);
        assert!(Options::parse(&conflict).is_err());
        assert!(Options::parse(&strings(&["--suite", "release", "--rust-only"])).is_err());
        assert!(Options::parse(&strings(&["--suite", "ci", "--evidence"])).is_err());
        let rust = Options::parse(&strings(&["--suite", "ci", "--rust-only"]));
        assert_eq!(rust.map(|options| options.mode).ok(), Some(Mode::RustOnly));
    }
}
