//! Execute the maintained PowerShell functions on native Windows, never model their path semantics.
use super::{function, script};
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}
fn execute(body: &str, cwd: &Path) -> process::ProcessReport {
    let body = format!(
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); $OutputEncoding = [Console]::OutputEncoding; {body}"
    );
    let system = std::env::var_os("SystemRoot").expect("Windows SystemRoot");
    let executable = PathBuf::from(system).join("System32/WindowsPowerShell/v1.0/powershell.exe");
    assert!(
        executable.is_file(),
        "native Windows PowerShell is required"
    );
    let environment = [
        "SystemRoot",
        "WINDIR",
        "PATH",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
    ]
    .into_iter()
    .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
    .collect::<BTreeMap<_, _>>();
    let spec = ProcessSpec {
        executable,
        arguments: ["-NoProfile", "-NonInteractive", "-Command", body.as_str()]
            .into_iter()
            .map(|arg| Value::Public(arg.into()))
            .collect(),
        cwd: cwd.into(),
        environment,
    };
    let report = process::supervise(
        &spec,
        &Limits {
            execution: Duration::from_secs(15),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 32768,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.cleanup.complete, "{report:?}");
    report
}
fn accepted(body: &str, cwd: &Path) -> String {
    let report = execute(body, cwd);
    assert!(report.success(), "{report:?}");
    String::from_utf8(report.stdout.bytes_retained)
        .unwrap()
        .trim()
        .to_owned()
}
#[cfg_attr(windows, test)]
fn native_windows_build_directory_tracks_prepared_pin_and_explicit_override() {
    let directory = tempfile::tempdir().unwrap();
    let llama = directory.path().join("llama-é-模型.cpp");
    fs::create_dir(&llama).unwrap();
    let build = directory.path().join("build-é-模型");
    let body = format!(
        "$ErrorActionPreference='Stop'; $llamaDir={}; $llamaBuildRoot={}; {}; Resolve-StageBuildDir 'cuda'",
        quote(llama.to_str().unwrap()),
        quote(build.to_str().unwrap()),
        function(&script(), "Resolve-StageBuildDir")
    );
    assert_eq!(
        PathBuf::from(accepted(&body, directory.path())),
        build.join("build-stage-abi-cuda")
    );
    for (pin, suffix) in [
        ("0123456789abcdef0123456789abcdef01234567", "0123456789ab"),
        ("fedcba9876543210fedcba9876543210fedcba98", "fedcba987654"),
    ] {
        fs::write(llama.join(".mesh-llm-patched-sha"), format!("{pin}\n")).unwrap();
        assert_eq!(
            PathBuf::from(accepted(&body, directory.path())),
            build.join(format!("build-stage-abi-cuda-{suffix}"))
        );
    }
    assert_eq!(
        accepted(
            &format!("$env:LLAMA_STAGE_BUILD_DIR='D:/abi'; {body}"),
            directory.path()
        ),
        "D:/abi"
    );
}
#[cfg_attr(windows, test)]
fn native_windows_bash_paths_preserve_repo_case_and_distinguish_git_bash_from_wsl() {
    let directory = tempfile::tempdir().unwrap();
    let prefix = format!(
        "$ErrorActionPreference='Stop'; $repoRoot='D:\\work\\mesh-llm'; {};",
        function(&script(), "ConvertTo-BashRepoPath")
    );
    for (path, switch, expected) in [
        (r"D:\work\mesh-llm\.deps\llama.cpp", "", ".deps/llama.cpp"),
        (r"d:\WORK\mesh-llm\.deps\other", "", ".deps/other"),
        (r"E:\cache\llama.cpp", "", "E:/cache/llama.cpp"),
        (r"E:\cache\llama.cpp", "-Wsl", "/mnt/e/cache/llama.cpp"),
        (
            r"D:\work\mesh-llm\.deps\llama.cpp",
            "-Wsl",
            ".deps/llama.cpp",
        ),
    ] {
        assert_eq!(
            accepted(
                &format!("{prefix} ConvertTo-BashRepoPath {} {switch}", quote(path)),
                directory.path()
            ),
            expected
        );
    }
    let rejected = execute(
        &format!("{prefix} ConvertTo-BashRepoPath '\\\\server\\share\\llama.cpp'"),
        directory.path(),
    );
    assert!(!rejected.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr.bytes_retained)
            .contains("inside the repository or on a drive-letter path")
    );
}
#[cfg_attr(windows, test)]
fn native_windows_relative_llama_override_is_bound_to_repo_not_process_directory() {
    let directory = tempfile::tempdir().unwrap();
    let source = script();
    let assignment = source
        .lines()
        .find(|line| line.starts_with("$llamaDir = "))
        .expect("llama path assignment");
    for (value, expected) in [
        (r".deps\other", r"D:\work\mesh-llm\.deps\other"),
        ("custom/llama.cpp", r"D:\work\mesh-llm\custom\llama.cpp"),
        (r"E:\cache\llama.cpp", r"E:\cache\llama.cpp"),
    ] {
        let body = format!(
            "$ErrorActionPreference='Stop'; $repoRoot='D:\\work\\mesh-llm'; $env:MESH_LLM_LLAMA_DIR={}; {assignment}; $llamaDir",
            quote(value)
        );
        assert_eq!(accepted(&body, directory.path()), expected);
    }
}
