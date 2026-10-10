//! UI package policy and actual build-script/shell behavior, without installation.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use crate::workflow_yaml::{self, Node};
use semver::{Version, VersionReq};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

mod actual_console_build_script {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../mesh/crates/mesh-llm-ui/build.rs"
    ));
    pub(super) fn run() {
        main();
    }
}

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn satisfies(pin: &str, range: &str) -> bool {
    let requirement = range.split_whitespace().collect::<Vec<_>>().join(", ");
    assert!(!requirement.is_empty(), "pnpm engine range is required");
    VersionReq::parse(&requirement)
        .unwrap()
        .matches(&Version::parse(pin).unwrap())
}
#[test]
fn ui_pnpm_pin_satisfies_lower_bound() {
    assert!(satisfies("10.30.3", ">=10"));
}
#[test]
fn ui_pnpm_pin_refuses_exclusive_upper_bound() {
    assert!(!satisfies("10.30.3", "<10"));
}
#[test]
fn ui_pnpm_pin_refuses_narrow_conjunction() {
    assert!(!satisfies("10.30.3", ">=10 <10.30.0"));
}
#[test]
fn ui_pnpm_pin_satisfies_wide_conjunction() {
    assert!(satisfies("10.30.3", ">=10 <11"));
}

fn execute(
    executable: PathBuf,
    cwd: &Path,
    arguments: Vec<Value>,
    environment: BTreeMap<std::ffi::OsString, Value>,
) -> String {
    let report = process::supervise(
        &ProcessSpec {
            executable,
            cwd: cwd.to_path_buf(),
            arguments,
            environment,
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        report.cleanup.complete && !report.stdout.truncated && !report.stderr.truncated,
        "{report:?}"
    );
    assert!(
        report.success(),
        "{}",
        String::from_utf8_lossy(&report.stderr.bytes_retained)
    );
    String::from_utf8(report.stdout.bytes_retained).unwrap()
}

#[test]
fn ui_build_script_uses_fallback_without_missing_directory_watch() {
    if std::env::var_os("MESH_UI_BUILD_SCRIPT_FIXTURE_CHILD").is_some() {
        actual_console_build_script::run();
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let ui = root.join("ui");
    let out = root.join("out");
    fs::create_dir_all(&ui).unwrap();
    fs::create_dir_all(&out).unwrap();
    for present in [false, true] {
        if present {
            fs::create_dir(ui.join("dist")).unwrap();
        }
        let expected = if present {
            ui.join("dist")
        } else {
            out.join("empty-ui-dist")
        };
        let stdout = execute(std::env::current_exe().unwrap().canonicalize().unwrap(), &root,
            ["--exact", "ui_package_contracts::ui_build_script_uses_fallback_without_missing_directory_watch", "--nocapture"].into_iter().map(|s| Value::Public(s.into())).collect(),
            BTreeMap::from([("MESH_UI_BUILD_SCRIPT_FIXTURE_CHILD".into(), Value::Public("1".into())), ("CARGO_MANIFEST_DIR".into(), Value::Public(ui.clone().into())), ("OUT_DIR".into(), Value::Public(out.clone().into()))]));
        assert!(
            stdout.contains(&format!(
                "cargo:rustc-env=MESH_LLM_UI_DIST={}",
                expected.display()
            )),
            "{stdout}"
        );
        assert!(
            !stdout.contains("cargo:rerun-if-changed=dist"),
            "missing dist must not force recurring rebuilds"
        );
        assert!(expected.is_dir());
    }
}

#[test]
fn ui_workspace_contains_root_package() {
    let workspace = workflow_yaml::parse(
        &fs::read_to_string(repository().join("mesh/crates/mesh-llm-ui/pnpm-workspace.yaml"))
            .unwrap(),
    )
    .unwrap();
    assert!(workspace.get("packages").unwrap().list().contains(&"."));
}

fn pnpm_versions(node: &Node, versions: &mut Vec<String>) {
    if node
        .get("uses")
        .and_then(Node::text)
        .is_some_and(|v| v.starts_with("pnpm/action-setup@"))
        && let Some(version) = node
            .get("with")
            .and_then(|v| v.get("version"))
            .and_then(Node::text)
    {
        versions.push(version.to_owned());
    }
    match node {
        Node::Map(entries) => {
            for (_, child) in entries {
                pnpm_versions(child, versions);
            }
        }
        Node::Seq(children) => {
            for child in children {
                pnpm_versions(child, versions);
            }
        }
        Node::Scalar(_) => {}
    }
}
#[test]
fn ui_pnpm_engine_and_exact_pin_match_all_ci_setup_majors() {
    let root = repository();
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(root.join("mesh/crates/mesh-llm-ui/package.json")).unwrap(),
    )
    .unwrap();
    let range = manifest["engines"]["pnpm"].as_str().unwrap();
    let pin = manifest["packageManager"]
        .as_str()
        .unwrap()
        .strip_prefix("pnpm@")
        .unwrap();
    let version = Version::parse(pin).unwrap();
    assert!(version.pre.is_empty() && version.build.is_empty());
    assert_eq!(
        pin,
        version.to_string(),
        "packageManager must pin a canonical exact version"
    );
    assert!(satisfies(pin, range));
    let engine =
        VersionReq::parse(&range.split_whitespace().collect::<Vec<_>>().join(", ")).unwrap();
    let declared_major = engine.comparators.first().unwrap().major;
    assert_eq!(
        version.major, declared_major,
        "engine floor and packageManager major must agree"
    );
    let mut versions = Vec::new();
    for entry in fs::read_dir(root.join(".github/workflows")).unwrap() {
        let path = entry.unwrap().path();
        if matches!(
            path.extension().and_then(|v| v.to_str()),
            Some("yml" | "yaml")
        ) {
            pnpm_versions(
                &workflow_yaml::parse(&fs::read_to_string(path).unwrap()).unwrap(),
                &mut versions,
            );
        }
    }
    assert!(
        !versions.is_empty(),
        "CI must declare at least one explicit pnpm setup version"
    );
    for ci in versions {
        assert_eq!(
            ci.split('.').next().unwrap().parse::<u64>().unwrap(),
            version.major,
            "CI pnpm version {ci}"
        );
    }
}
#[test]
fn ui_npmrc_enforces_engine_strictly() {
    let source = fs::read_to_string(repository().join("mesh/crates/mesh-llm-ui/.npmrc")).unwrap();
    let mut strict = None;
    for line in source.lines() {
        let line = line.split('#').next().unwrap().trim();
        if let Some((name, value)) = line.split_once('=')
            && name.trim() == "engine-strict"
        {
            assert!(
                strict.replace(value.trim()).is_none(),
                "duplicate engine-strict setting"
            );
        }
    }
    assert_eq!(strict, Some("true"));
}

#[test]
fn ui_actual_shell_normalizes_mixed_case_release_profile() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let ui = root.join("ui");
    fs::create_dir(&ui).unwrap();
    for name in [
        "package.json",
        "pnpm-lock.yaml",
        "vite.config.ts",
        "tsconfig.json",
        "tsconfig.app.json",
        "tsconfig.node.json",
        "biome.json",
        "index.html",
    ] {
        fs::write(ui.join(name), "{}\n").unwrap();
    }
    for name in ["src", "public", "dist"] {
        fs::create_dir(ui.join(name)).unwrap();
    }
    fs::write(ui.join("dist/asset.js"), "// built\n").unwrap();
    fs::write(ui.join("dist/.mesh-llm-ui-build-env"), r#"{"dotenv":{".env":null,".env.local":null,".env.production":null,".env.production.local":null},"profile":"release","schema":2,"variables":{"VITE_MESH_LLM_DEBUG_UI":"false"}}"#).unwrap();
    let search = std::env::var_os("PATH").unwrap();
    let bash = std::env::split_paths(&search)
        .map(|p| p.join("bash"))
        .find(|p| p.is_file())
        .unwrap()
        .canonicalize()
        .unwrap();
    let stdout = execute(
        bash,
        &root,
        vec![
            Value::Public(repository().join("mesh/scripts/build-ui.sh").into()),
            Value::Public(ui.into()),
        ],
        BTreeMap::from([
            ("PATH".into(), Value::Public(search)),
            (
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(
                    Path::new(env!("CARGO_BIN_EXE_xtask"))
                        .canonicalize()
                        .unwrap()
                        .into(),
                ),
            ),
            (
                "MESH_LLM_BUILD_PROFILE".into(),
                Value::Public("ReLeAsE".into()),
            ),
        ]),
    );
    assert!(
        stdout.contains("Skipping mesh-llm UI build"),
        "must reuse the already-current fixture without package installation: {stdout}"
    );
    assert!(stdout.contains("profile: release") && stdout.contains("debug UI: false"));
}

fn ui_cli_fixture(mode: &str, expected: i32) -> String {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    fs::write(root.join("package.json"), "{}\n").unwrap();
    fs::write(root.join("pnpm-lock.yaml"), "lock\n").unwrap();
    let script = root.join("provider.cjs");
    let provider = r#"
const fs=require('fs');
const mode=MODE;
const args=process.argv.slice(2);
if(args[0]==='install') {
  if(mode==='install-fail') process.exit(42);
  fs.mkdirSync('node_modules',{recursive:true});
} else if(args[0]==='run' && args[1]==='build') {
  fs.mkdirSync('dist',{recursive:true});
  if(mode==='empty') process.exit(0);
  fs.writeFileSync('dist/index.html','fixture');
  if(mode==='build-fail') process.exit(7);
  if(mode==='signal') fs.writeFileSync('cli.pid',String(process.ppid));
  if(mode==='deadline' || mode==='signal') setInterval(()=>{},1000);
} else process.exit(91);
"#;
    fs::write(
        &script,
        provider.replace("MODE", &serde_json::to_string(mode).unwrap()),
    )
    .unwrap();
    let search = std::env::var_os("PATH").unwrap();
    let node = std::env::split_paths(&search)
        .map(|p| p.join("node"))
        .find(|p| p.is_file())
        .unwrap()
        .canonicalize()
        .unwrap();
    let arguments = vec![
        "automation".into(),
        "ui-build".into(),
        "--ui-dir".into(),
        root.clone().into_os_string(),
        "--pnpm-command".into(),
        node.into_os_string(),
        "--pnpm-script".into(),
        script.into_os_string(),
        "--timeout-secs".into(),
        if mode == "deadline" { "2" } else { "10" }.into(),
    ];
    let signal = ui_cli_signal_sender(mode, &root);
    let report = process::supervise(
        &ProcessSpec {
            executable: Path::new(env!("CARGO_BIN_EXE_xtask"))
                .canonicalize()
                .unwrap(),
            cwd: root.clone(),
            arguments: arguments.into_iter().map(Value::Public).collect(),
            environment: BTreeMap::from([
                ("PATH".into(), Value::Public(search)),
                ("HOME".into(), Value::Public(root.clone().into())),
                (
                    "MESH_LLM_BUILD_PROFILE".into(),
                    Value::Public("release".into()),
                ),
            ]),
        },
        &Limits {
            execution: Duration::from_secs(15),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    if let Some(signal) = signal {
        signal.join().unwrap();
    }
    assert!(report.cleanup.complete, "{report:?}");
    assert_eq!(report.status.unwrap().code(), Some(expected), "{report:?}");
    assert!(
        report.stdout.bytes_retained.is_empty(),
        "failure must not announce success"
    );
    assert!(!root.join("dist/.mesh-llm-ui-build-env").exists());
    String::from_utf8(report.stderr.bytes_retained).unwrap()
}

#[test]
fn ui_actual_cli_preserves_install_and_build_failure_codes() {
    for (mode, code) in [("install-fail", 42), ("build-fail", 7)] {
        assert!(ui_cli_fixture(mode, code).contains("UI"));
    }
}
#[test]
fn ui_actual_cli_rejects_success_without_output() {
    assert!(ui_cli_fixture("empty", 1).contains("without regular output files"));
}
#[test]
fn ui_actual_cli_deadline_returns_timeout_status_after_cleanup() {
    assert!(ui_cli_fixture("deadline", 124).contains("UI"));
}

#[test]
fn ui_actual_cli_sigterm_returns_signal_status_after_cleanup() {
    assert!(ui_cli_fixture("signal", 143).contains("UI"));
}

fn ui_cli_signal_sender(mode: &str, root: &Path) -> Option<std::thread::JoinHandle<()>> {
    (mode == "signal").then(|| {
        let marker = root.join("cli.pid");
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(8);
            while std::time::Instant::now() < deadline {
                if let Ok(text) = fs::read_to_string(&marker)
                    && let Ok(pid) = text.parse::<i32>()
                {
                    assert!(pid > 1);
                    // The native fixture records its owning xtask parent.
                    assert_eq!(unsafe { libc::kill(pid, libc::SIGTERM) }, 0);
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            panic!("UI producer did not publish its parent marker");
        })
    })
}

fn ui_shell_producer_fixture(fail: bool) -> (tempfile::TempDir, process::ProcessReport) {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    fs::write(root.join("package.json"), "{}\n").unwrap();
    fs::write(root.join("pnpm-lock.yaml"), "lock\n").unwrap();
    let bin = root.join("bin");
    fs::create_dir(&bin).unwrap();
    let pnpm = bin.join("pnpm");
    fs::write(
        &pnpm,
        r#"#!/usr/bin/env node
const fs=require('fs');
const args=process.argv.slice(2);
fs.appendFileSync('commands.jsonl',JSON.stringify(args)+'\n');
if(args[0]==='install') fs.mkdirSync('node_modules',{recursive:true});
else if(args[0]==='run' && args[1]==='build') {
  fs.mkdirSync('dist',{recursive:true});
  fs.writeFileSync('dist/index.html','fixture');
  if(process.env.NODE_UI_FAIL==='yes') process.exit(47);
} else process.exit(91);
"#,
    )
    .unwrap();
    fs::set_permissions(&pnpm, fs::Permissions::from_mode(0o755)).unwrap();
    let search = std::env::var_os("PATH").unwrap();
    let bash = std::env::split_paths(&search)
        .map(|p| p.join("bash"))
        .find(|p| p.is_file())
        .unwrap()
        .canonicalize()
        .unwrap();
    let path =
        std::env::join_paths(std::iter::once(bin).chain(std::env::split_paths(&search))).unwrap();
    let report = process::supervise(
        &ProcessSpec {
            executable: bash,
            cwd: root.clone(),
            arguments: vec![
                Value::Public(repository().join("scripts/build-ui.sh").into()),
                Value::Public(root.clone().into()),
            ],
            environment: BTreeMap::from([
                ("PATH".into(), Value::Public(path)),
                ("HOME".into(), Value::Public(root.clone().into())),
                (
                    "MESH_LLM_AUTOMATION_BIN".into(),
                    Value::Public(
                        Path::new(env!("CARGO_BIN_EXE_xtask"))
                            .canonicalize()
                            .unwrap()
                            .into(),
                    ),
                ),
                (
                    "MESH_LLM_BUILD_PROFILE".into(),
                    Value::Public("release".into()),
                ),
                (
                    "NODE_UI_FAIL".into(),
                    Value::Public(if fail { "yes" } else { "no" }.into()),
                ),
            ]),
        },
        &Limits {
            execution: Duration::from_secs(20),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    (temp, report)
}

#[test]
fn ui_actual_shell_builds_and_preserves_producer_failure() {
    for fail in [false, true] {
        let (temp, report) = ui_shell_producer_fixture(fail);
        assert!(report.cleanup.complete, "{report:?}");
        assert_eq!(
            report.status.unwrap().code(),
            Some(if fail { 47 } else { 0 }),
            "{report:?}"
        );
        assert_eq!(
            temp.path().join("dist/.mesh-llm-ui-build-env").exists(),
            !fail
        );
        let trace: Vec<Vec<String>> = fs::read_to_string(temp.path().join("commands.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            trace,
            vec![vec!["install", "--frozen-lockfile"], vec!["run", "build"]]
        );
    }
}

#[test]
fn ui_actual_cli_explicit_profile_preserves_vite_environment_and_reuse() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    fs::write(root.join("package.json"), "{}\n").unwrap();
    fs::write(root.join("pnpm-lock.yaml"), "lock\n").unwrap();
    let provider = root.join("provider.cjs");
    fs::write(
        &provider,
        r#"
const fs=require('fs');
if (process.argv[2]==='install') fs.mkdirSync('node_modules',{recursive:true});
else {
  fs.mkdirSync('dist',{recursive:true});
  fs.writeFileSync('dist/index.html','fixture');
  fs.writeFileSync('dist/received.json',JSON.stringify(process.env));
  fs.appendFileSync('builds.txt','built\n');
}
"#,
    )
    .unwrap();
    let search = std::env::var_os("PATH").unwrap();
    let node = std::env::split_paths(&search)
        .map(|p| p.join("node"))
        .find(|p| p.is_file())
        .unwrap()
        .canonicalize()
        .unwrap();
    let args = [
        "automation".into(),
        "ui-build".into(),
        "--ui-dir".into(),
        root.clone().into_os_string(),
        "--profile".into(),
        "ReLeAsE".into(),
        "--pnpm-command".into(),
        node.into_os_string(),
        "--pnpm-script".into(),
        provider.into_os_string(),
    ];
    let env: BTreeMap<std::ffi::OsString, std::ffi::OsString> = BTreeMap::from([
        ("PATH".into(), search),
        ("HOME".into(), root.clone().into()),
        (
            "MESH_LLM_BUILD_PROFILE".into(),
            "invalid ambient profile".into(),
        ),
        ("VITE_MESH_LLM_DEBUG_UI".into(), "ignored\nrelease".into()),
        ("VITE_API_URL".into(), "https://api.example.invalid".into()),
        ("VITE_MANAGEMENT_API_URL".into(), "".into()),
        ("VITE_STORAGE_NAMESPACE".into(), "first\nsecond".into()),
        ("TANSTACK_FILE_ROUTER".into(), "true".into()),
    ]);
    let binary = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .canonicalize()
        .unwrap();
    for action in ["Built mesh-llm UI", "Skipping mesh-llm UI build"] {
        let output = execute(
            binary.clone(),
            &root,
            args.iter().cloned().map(Value::Public).collect(),
            env.iter()
                .map(|(name, value)| (name.clone(), Value::Public(value.clone())))
                .collect(),
        );
        assert!(output.contains(action), "{output}");
    }
    assert_eq!(
        fs::read_to_string(root.join("builds.txt")).unwrap(),
        "built\n"
    );
    let received: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("dist/received.json")).unwrap()).unwrap();
    assert!(received.get("VITE_BASE_PATH").is_none());
    assert!(received.get("VITE_ROUTER_BASE_PATH").is_none());
    assert_eq!(received["VITE_MESH_LLM_DEBUG_UI"], "false");
    assert_eq!(received["VITE_API_URL"], "https://api.example.invalid");
    assert_eq!(received["VITE_MANAGEMENT_API_URL"], "");
    assert_eq!(received["VITE_STORAGE_NAMESPACE"], "first\nsecond");
    assert_eq!(received["TANSTACK_FILE_ROUTER"], "true");
}
