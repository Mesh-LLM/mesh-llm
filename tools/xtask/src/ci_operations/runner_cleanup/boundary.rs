use super::{Error, Plan, paths};
use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Profile {
    Build,
    Family,
    Replay,
    CudaRelease,
    Smoke,
    RunnerContract,
}

impl Profile {
    pub(crate) fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "build" => Ok(Self::Build),
            "family" => Ok(Self::Family),
            "replay" => Ok(Self::Replay),
            "cuda-release" => Ok(Self::CudaRelease),
            "smoke" => Ok(Self::Smoke),
            "runner-contract" => Ok(Self::RunnerContract),
            _ => Err(Error::Input("unknown cleanup profile")),
        }
    }
}

pub(crate) struct Options {
    pub(crate) profile: Profile,
    pub(crate) evidence_uploaded: bool,
    pub(crate) package_uploaded: bool,
}

impl Options {
    pub(crate) fn parse(
        profile: &str,
        evidence: &str,
        package: Option<&str>,
    ) -> Result<Self, Error> {
        let profile = Profile::parse(profile)?;
        let evidence_uploaded = uploaded(evidence)?;
        let package_uploaded = uploaded(package.unwrap_or("false"))?;
        Ok(Self {
            profile,
            evidence_uploaded,
            package_uploaded,
        })
    }
}

fn uploaded(value: &str) -> Result<bool, Error> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(Error::Input("upload outcome must be true or false")),
    }
}

pub(super) struct Roots {
    pub workspace: PathBuf,
    pub temporary: PathBuf,
}
pub(super) struct Canary {
    pub root: PathBuf,
    pub key: String,
    pub pass: String,
}

fn required<'a>(
    env: &'a BTreeMap<OsString, OsString>,
    key: &'static str,
) -> Result<&'a OsStr, Error> {
    env.get(OsStr::new(key))
        .map(OsString::as_os_str)
        .ok_or(Error::Missing(key))
}

fn text(env: &BTreeMap<OsString, OsString>, key: &'static str) -> Result<String, Error> {
    required(env, key)?
        .to_str()
        .map(str::to_owned)
        .ok_or(Error::Input("identity must be ASCII"))
}

fn digits(value: &str, message: &'static str) -> Result<(), Error> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(Error::Input(message));
    }
    Ok(())
}

fn canary(env: &BTreeMap<OsString, OsString>, roots: &Roots) -> Result<Canary, Error> {
    let root = paths::absolute(Path::new(required(env, "CANARY_SOURCE_ROOT")?))?;
    if root != roots.workspace && root != roots.workspace.join("canary-source") {
        return Err(Error::Input(
            "source root must be the controller or selected-source checkout",
        ));
    }
    let run = text(env, "GITHUB_RUN_ID")?;
    let attempt = text(env, "GITHUB_RUN_ATTEMPT")?;
    let pass = text(env, "CANARY_PASS_ID")?;
    digits(&run, "invalid run identity")?;
    digits(&attempt, "invalid run identity")?;
    if !matches!(
        pass.as_str(),
        "repair-1" | "repair-2" | "repair-3" | "verify-1" | "verify-2" | "verify-3"
    ) {
        return Err(Error::Input("invalid pass identity"));
    }
    Ok(Canary {
        root,
        key: format!("{run}-{attempt}"),
        pass,
    })
}

pub(crate) fn from_environment(
    options: &Options,
    env: &BTreeMap<OsString, OsString>,
) -> Result<Plan, Error> {
    let workspace = paths::resolve(Path::new(required(env, "GITHUB_WORKSPACE")?))?;
    let temporary = paths::resolve(Path::new(required(env, "RUNNER_TEMP")?))?;
    let roots = Roots {
        workspace,
        temporary,
    };
    match options.profile {
        Profile::Build => Ok(super::plan::build(&roots, &canary(env, &roots)?, options)),
        Profile::Family => {
            let canary = canary(env, &roots)?;
            let shard = text(env, "CANARY_SHARD_INDEX")?;
            digits(&shard, "invalid shard identity")?;
            Ok(super::plan::family(
                &roots,
                &canary,
                (&shard, options.evidence_uploaded),
            ))
        }
        Profile::Replay => Ok(super::plan::replay(&roots, options.evidence_uploaded)),
        Profile::CudaRelease => Ok(super::plan::cuda(&roots, options.evidence_uploaded)),
        Profile::Smoke => {
            let artifact = required(env, "CLEANUP_ARTIFACT_PATH")?;
            super::plan::smoke_path(&roots.workspace, Path::new(artifact), false)?;
            let binary = required(env, "CLEANUP_BINARY_PATH")?;
            super::plan::smoke(&roots, Path::new(artifact), Path::new(binary))
        }
        Profile::RunnerContract => Ok(Plan::files(vec![super::Target::new(
            &roots.workspace,
            roots.workspace.join("target"),
        )])),
    }
}
