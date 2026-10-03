//! Typed UI build options and environment projection.
use super::policy::{Environment, Profile};
use crate::command::DynResult;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    time::Duration,
};

pub(super) struct Options {
    pub ui: PathBuf,
    pub logs: Option<PathBuf>,
    pub executable: Option<PathBuf>,
    pub pnpm_script: Option<PathBuf>,
    pub timeout: Duration,
    pub profile: Option<Profile>,
}

impl Options {
    pub(super) fn parse(args: &[String]) -> DynResult<Self> {
        let mut ui = None;
        let mut profile = None;
        let mut logs = None;
        let mut executable = None;
        let mut pnpm_script = None;
        let mut timeout = Duration::from_secs(1800);
        let mut seen = BTreeSet::new();
        let mut arguments = args.iter();
        while let Some(flag) = arguments.next() {
            if !seen.insert(flag) {
                return Err(format!("repeated UI build option: {flag}").into());
            }
            let value = arguments.next().ok_or("UI build option requires a value")?;
            if value.is_empty() || value.starts_with("--") {
                return Err("UI build option requires a nonempty value".into());
            }
            match flag.as_str() {
                "--profile" => profile = Some(Profile::parse(value)?),
                "--ui-dir" => ui = Some(PathBuf::from(value)),
                "--logs-dir" => logs = Some(PathBuf::from(value)),
                "--pnpm-command" => executable = Some(PathBuf::from(value)),
                "--pnpm-script" => pnpm_script = Some(PathBuf::from(value)),
                "--timeout-secs" => {
                    let seconds: u64 = value.parse()?;
                    if !(1..=3600).contains(&seconds) {
                        return Err("UI build timeout must be 1..3600 seconds".into());
                    }
                    timeout = Duration::from_secs(seconds);
                }
                _ => return Err(format!("unknown UI build option: {flag}").into()),
            }
        }
        if pnpm_script.is_some() && executable.is_none() {
            return Err("--pnpm-script requires an explicit native Node --pnpm-command".into());
        }
        Ok(Self {
            ui: ui.ok_or("UI build requires --ui-dir")?,
            logs,
            executable,
            pnpm_script,
            timeout,
            profile,
        })
    }
}

pub(super) struct BuildEnvironment {
    build: Environment,
}

pub(super) fn is_build_setting(name: &str) -> bool {
    name.starts_with("VITE_")
        || matches!(
            name,
            "TANSTACK_FILE_ROUTER" | "MESH_UI_API_ORIGIN" | "NODE_ENV"
        )
}

impl BuildEnvironment {
    pub(super) fn from_environment(override_profile: Option<Profile>) -> DynResult<Self> {
        let profile = match override_profile {
            Some(profile) => profile,
            None => match std::env::var("MESH_LLM_BUILD_PROFILE") {
                Ok(value) => Profile::parse(&value)?,
                Err(std::env::VarError::NotPresent) => Profile::Debug,
                Err(std::env::VarError::NotUnicode(_)) => {
                    return Err("MESH_LLM_BUILD_PROFILE must be UTF-8".into());
                }
            },
        };
        let mut variables = BTreeMap::new();
        for (name, value) in std::env::vars_os() {
            let Some(name) = name.to_str() else {
                continue;
            };
            if !is_build_setting(name) {
                continue;
            }
            // Release owns the debug setting and does not read an inactive override.
            if profile == Profile::Release && name == "VITE_MESH_LLM_DEBUG_UI" {
                continue;
            }
            let value = value
                .into_string()
                .map_err(|_| format!("{name} must be UTF-8"))?;
            variables.insert(name.to_owned(), value);
        }
        Ok(Self {
            build: Environment { profile, variables },
        })
    }

    pub(super) fn projection(&self) -> &Environment {
        &self.build
    }
}
