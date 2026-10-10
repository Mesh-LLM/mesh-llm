use super::{Error, Git, Options, Profile};
use crate::{command::DynResult, repository::check_report::CheckReport};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    io::{self, Write},
    path::PathBuf,
};

const USAGE: &str = "ci-ops runner-cleanup --job {build|family|replay|cuda-release|smoke|runner-contract|canary-preflight} --evidence-uploaded {true|false} [--package-uploaded {true|false}] [--git <absolute-executable>]";

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        return CheckReport::success(format!("usage: {USAGE}\n")).emit();
    }
    let mut values = BTreeMap::new();
    let mut arguments = args.iter();
    while let Some(flag) = arguments.next() {
        if !matches!(
            flag.as_str(),
            "--job" | "--evidence-uploaded" | "--package-uploaded" | "--git"
        ) {
            return CheckReport::usage(USAGE, "unknown argument").emit();
        }
        let Some(value) = arguments.next().filter(|value| !value.starts_with("--")) else {
            return CheckReport::usage(USAGE, "missing argument value").emit();
        };
        values.insert(flag.as_str(), value.as_str());
    }
    let (Some(job), Some(evidence)) = (values.get("--job"), values.get("--evidence-uploaded"))
    else {
        return CheckReport::usage(USAGE, "job and evidence-uploaded are required").emit();
    };
    let options = match Options::parse(job, evidence, values.get("--package-uploaded").copied()) {
        Ok(options) => options,
        Err(error) => return CheckReport::usage(USAGE, &error.to_string()).emit(),
    };
    let environment: BTreeMap<OsString, OsString> = std::env::vars_os().collect();
    let git = match options.profile {
        Profile::Replay => {
            let executable = values
                .get("--git")
                .ok_or(Error::Input("replay requires --git"))?;
            let child_environment = [
                "PATH",
                "HOME",
                "GIT_CONFIG_NOSYSTEM",
                "GIT_CONFIG_GLOBAL",
                "GIT_TERMINAL_PROMPT",
                "SYSTEMROOT",
                "WINDIR",
                "TEMP",
                "TMP",
            ]
            .into_iter()
            .filter_map(|name| {
                environment
                    .get(std::ffi::OsStr::new(name))
                    .map(|value| (name.into(), value.clone()))
            })
            .collect();
            Some(Git::new(PathBuf::from(executable), child_environment)?)
        }
        Profile::Build
        | Profile::Family
        | Profile::CudaRelease
        | Profile::Smoke
        | Profile::RunnerContract => None,
        Profile::CanaryPreflight => None,
    };
    Ok(super::run(
        &options,
        &environment,
        (git.as_ref(), &mut Output(Vec::new())),
    )?)
}

struct Output(Vec<u8>);

impl Write for Output {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let text = String::from_utf8(std::mem::take(&mut self.0)).map_err(io::Error::other)?;
        CheckReport::success(text)
            .emit()
            .map_err(|error| io::Error::other(error.to_string()))
    }
}
