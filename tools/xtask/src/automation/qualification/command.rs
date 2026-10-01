use super::schema::{Platform, Scenario};
use super::{Error, load, validation};
use crate::command::DynResult;
use std::path::{Path, PathBuf};

pub(crate) const USAGE: &str = "cargo xtool automation qualify --platform <linux|macos|windows> --evidence <dir>\ncargo xtool automation qualify-replay --receipt <file> --scenario <product-readiness|protocol-pair|corrupt-runtime|readiness-timeout> --evidence <dir>";

pub(crate) fn run(verb: &str, args: &[String], selected_root: Option<&Path>) -> DynResult<()> {
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let options = parse(verb, args)?;
    match options {
        Options::Qualify { platform, evidence } => {
            let _requested = (platform, evidence);
            Err(Error::ExecutionPending("task28 acceptance and frozen contracts.json qualification matrix required; no receipt emitted").into())
        }
        Options::Replay {
            receipt,
            scenario,
            evidence,
        } => {
            let receipt = load(&receipt)?;
            let root = match selected_root {
                Some(root) => root.to_path_buf(),
                None => std::env::current_dir()?,
            };
            let source = super::source::head(&root)?;
            validation::validate(&receipt, source.trim())?;
            let contracts = root.join("ci/automation-migration/contracts.json");
            let expected_digest = crate::product::digest::file_sha256(&contracts)
                .map_err(|failure| Error::Io(failure.error))?;
            if receipt.contracts.path != contracts || receipt.contracts.sha256 != expected_digest {
                return Err(
                    Error::Invalid("receipt does not bind candidate frozen contracts").into(),
                );
            }
            if receipt.platform != current_platform()? {
                return Err(Error::Invalid("receipt platform differs from replay host").into());
            }
            if receipt.scenarios.iter().all(|row| row.scenario != scenario) {
                return Err(Error::Invalid("requested scenario absent from receipt").into());
            }
            let _evidence = evidence;
            Err(Error::ExecutionPending("validated inputs only; product composition/client-readiness, task19 process and task21 smoke execution adapters pending; no success evidence emitted").into())
        }
    }
}

enum Options {
    Qualify {
        platform: Platform,
        evidence: PathBuf,
    },
    Replay {
        receipt: PathBuf,
        scenario: Scenario,
        evidence: PathBuf,
    },
}

fn parse(verb: &str, args: &[String]) -> Result<Options, Error> {
    let mut values = std::collections::BTreeMap::new();
    let (pairs, remainder) = args.as_chunks::<2>();
    for pair in pairs {
        if pair[1].is_empty()
            || pair[1].starts_with("--")
            || values.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err(Error::Invalid("missing or duplicate argument"));
        }
    }
    if !remainder.is_empty() {
        return Err(Error::Invalid("missing argument value"));
    }
    let evidence = values
        .remove("--evidence")
        .ok_or(Error::Invalid("--evidence required"))?;
    let evidence = Path::new(evidence).to_path_buf();
    let options = match verb {
        "qualify" => {
            let platform = match values.remove("--platform") {
                Some("linux") => Platform::Linux,
                Some("macos") => Platform::Macos,
                Some("windows") => Platform::Windows,
                _ => return Err(Error::Invalid("--platform must be linux, macos or windows")),
            };
            Options::Qualify { platform, evidence }
        }
        "qualify-replay" => {
            let receipt = values
                .remove("--receipt")
                .ok_or(Error::Invalid("--receipt required"))?;
            let scenario = match values.remove("--scenario") {
                Some("product-readiness") => Scenario::ProductReadiness,
                Some("protocol-pair") => Scenario::ProtocolPair,
                Some("corrupt-runtime") => Scenario::CorruptRuntime,
                Some("readiness-timeout") => Scenario::ReadinessTimeout,
                _ => return Err(Error::Invalid("unsupported replay scenario")),
            };
            Options::Replay {
                receipt: PathBuf::from(receipt),
                scenario,
                evidence,
            }
        }
        _ => return Err(Error::Invalid(USAGE)),
    };
    if !values.is_empty() {
        return Err(Error::Invalid("unknown qualification argument"));
    }
    Ok(options)
}

fn current_platform() -> Result<Platform, Error> {
    match std::env::consts::OS {
        "linux" => Ok(Platform::Linux),
        "macos" => Ok(Platform::Macos),
        "windows" => Ok(Platform::Windows),
        _ => Err(Error::Invalid("unsupported replay host")),
    }
}
