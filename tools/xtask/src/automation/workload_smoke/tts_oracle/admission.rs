use super::options::Options;
use crate::{automation::canary_package_closure, command::DynResult};
use std::{
    fs,
    path::{Path, PathBuf},
};
pub(super) enum Candidate {
    Prebuilt(PathBuf),
    Standalone(PathBuf),
}
pub(super) struct Admission {
    pub patched: String,
    pub candidate: Candidate,
}
fn stamp(path: &Path, patched: &str, tools: bool) -> DynResult<()> {
    let bytes = super::regular_input::read(path, 65536, "TTS native stamp")?;
    let lines = std::str::from_utf8(&bytes)?
        .lines()
        .collect::<std::collections::BTreeSet<_>>();
    let mut policy = vec![
        format!("patched-sha={patched}"),
        "backend=cpu".into(),
        "link-mode=static".into(),
        "ggml-native=OFF".into(),
        "cmake-arg=-DGGML_NATIVE=OFF".into(),
        "cmake-arg=-DGGML_METAL=OFF".into(),
    ];
    if tools {
        policy.push("cmake-arg=-DLLAMA_BUILD_TOOLS=ON".into());
    }
    for required in policy {
        let key = required
            .rsplit_once('=')
            .ok_or("invalid TTS policy field")?
            .0;
        let prefix = format!("{key}=");
        if !lines.contains(required.as_str())
            || lines
                .iter()
                .any(|line| line.starts_with(&prefix) && *line != required.as_str())
        {
            return Err("TTS native stamp does not match pinned static CPU policy".into());
        }
    }
    Ok(())
}
fn tool(name: &str) -> DynResult<PathBuf> {
    let search = std::env::var_os("PATH").ok_or("PATH unavailable")?;
    for directory in std::env::split_paths(&search) {
        let path = std::path::absolute(directory.join(name))?;
        if executable(&path).is_ok() {
            return Ok(path);
        }
    }
    Err(format!("required TTS tool {name} unavailable").into())
}
fn executable(path: &Path) -> DynResult<()> {
    if !path.is_file() {
        return Err("TTS native input must be an executable file".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(path)?.permissions().mode() & 0o111 == 0 {
            return Err("TTS native input lacks execute permission".into());
        }
    }
    Ok(())
}
pub(super) fn admit(options: &Options) -> DynResult<Admission> {
    if options.oracle.file_name().and_then(|n| n.to_str()) != Some("llama-tts") {
        return Err("oracle CLI must be an executable llama-tts binary".into());
    }
    executable(&options.oracle)?;
    if !options.model_path.is_file() || !options.projector.is_file() {
        return Err("TTS model and projector must be files".into());
    }
    let patched = canary_package_closure::prepared_native_head(&options.root)?;
    let native = PathBuf::from(
        std::env::var_os("LLAMA_STAGE_BUILD_DIR")
            .ok_or("LLAMA_STAGE_BUILD_DIR required for pinned TTS candidate")?,
    )
    .canonicalize()?;
    stamp(&native.join(".mesh-llm-build-stamp"), &patched, false)?;
    stamp(
        &options
            .oracle
            .parent()
            .and_then(Path::parent)
            .ok_or("oracle missing build root")?
            .join(".mesh-llm-build-stamp"),
        &patched,
        true,
    )?;
    let manifest = std::env::var_os("SKIPPY_WORKLOAD_PRODUCER_MANIFEST");
    let binary = std::env::var_os("SKIPPY_WORKLOAD_CANDIDATE_BIN_DIR");
    let native = std::env::var_os("SKIPPY_WORKLOAD_NATIVE_BUILD_DIR");
    let candidate = if manifest.is_some() || binary.is_some() || native.is_some() {
        let manifest = manifest
            .filter(|v| !v.is_empty())
            .ok_or("prebuilt TTS requires producer manifest")?;
        let binary = PathBuf::from(
            binary
                .filter(|v| !v.is_empty())
                .ok_or("prebuilt TTS requires candidate path")?,
        )
        .join("skippy-server");
        let native = PathBuf::from(
            native
                .filter(|v| !v.is_empty())
                .ok_or("prebuilt TTS requires native path")?,
        );
        Candidate::Prebuilt(canary_package_closure::verified_workload_test(
            &options.root,
            &binary,
            &native,
            Path::new(&manifest),
        )?)
    } else {
        Candidate::Standalone(tool("just")?)
    };
    Ok(Admission { patched, candidate })
}
