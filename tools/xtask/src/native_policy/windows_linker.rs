//! Keep nonshipping proc-macro DLL metadata stable without changing product links.
use crate::command::DynResult;
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, Readiness, Value,
};
use std::{collections::BTreeMap, ffi::OsString, fs, path::Path, time::Duration};

const ENABLE: &str = "MESH_CI_STABLE_PROC_MACRO_METADATA";

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [flag, executable, separator, arguments @ ..] = args else {
        return Err("windows-linker requires --lld PATH -- LINKER_ARGUMENTS".into());
    };
    if flag != "--lld" || separator != "--" || std::env::var(ENABLE).as_deref() != Ok("1") {
        return Err("windows-linker requires its explicit CI opt-in and --lld PATH --".into());
    }
    let environment = std::env::vars_os().collect::<BTreeMap<_, _>>();
    let mut arguments = arguments.to_vec();
    if proc_macro_dependency(&environment)? {
        // Append after any response file so this wins over rustc's unconditional /DEBUG.
        // Leave the response file's bytes and all original arguments untouched.
        arguments.push("/DEBUG:NONE".into());
    }
    let spec = ProcessSpec {
        executable: executable.into(),
        arguments: arguments
            .into_iter()
            .map(|arg| Value::Public(arg.into()))
            .collect(),
        cwd: std::env::current_dir()?,
        environment: environment
            .into_iter()
            .map(|(key, value)| (key, Value::Secret(value)))
            .collect(),
    };
    let report = process::supervise_inherited(
        &spec,
        &Limits {
            execution: Duration::from_secs(3000),
            graceful_shutdown: Duration::from_secs(5),
            forced_shutdown: Duration::from_secs(5),
            retained_bytes_per_stream: 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
    )?;
    if report.outcome != Outcome::Exited
        || report.failure.is_some()
        || !report.cleanup.complete
        || report.cleanup.failure.is_some()
    {
        return Err("Windows linker did not exit with complete process cleanup".into());
    }
    let code = report
        .status
        .and_then(|status| status.code())
        .ok_or("Windows linker returned no exit code")?;
    if code != 0 {
        std::process::exit(code);
    }
    Ok(())
}

fn proc_macro_dependency(environment: &BTreeMap<OsString, OsString>) -> DynResult<bool> {
    if environment.contains_key(std::ffi::OsStr::new("CARGO_PRIMARY_PACKAGE")) {
        return Ok(false);
    }
    let value = |key| {
        environment
            .get(std::ffi::OsStr::new(key))
            .filter(|value| !value.is_empty())
    };
    let (Some(directory), Some(name)) = (value("CARGO_MANIFEST_DIR"), value("CARGO_CRATE_NAME"))
    else {
        return Ok(false);
    };
    let path = Path::new(directory).join("Cargo.toml");
    if fs::metadata(&path)?.len() > 1024 * 1024 {
        return Err("Cargo manifest exceeds the Windows linker metadata limit".into());
    }
    let manifest: toml::Value = toml::from_str(&fs::read_to_string(path)?)?;
    let Some(library) = manifest.get("lib") else {
        return Ok(false);
    };
    let is_macro = library
        .get("proc-macro")
        .or_else(|| library.get("proc_macro"))
        .and_then(toml::Value::as_bool)
        .unwrap_or(false);
    if !is_macro {
        return Ok(false);
    }
    let library_name = library
        .get("name")
        .or_else(|| manifest.get("package")?.get("name"))
        .and_then(toml::Value::as_str)
        .ok_or("proc-macro library has no name")?;
    Ok(name == &OsString::from(library_name.replace('-', "_")))
}

#[cfg(test)]
#[path = "windows_linker_tests.rs"]
mod tests;
