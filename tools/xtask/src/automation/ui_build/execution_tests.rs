//! Native component fixtures for UI build execution and cleanup.
use super::{
    execution::{self, Tool},
    failure::Failure,
    policy::{Decision, Environment, Profile},
};
use crate::process::Cancellation;
use std::{collections::BTreeMap, ffi::OsString, fs, path::PathBuf, time::Duration};

const PROVIDER: &str = r#"
const fs=require('fs');
const path=require('path');
const args=process.argv.slice(2);
fs.appendFileSync(process.env.UI_FIXTURE_TRACE, JSON.stringify(args)+'\n');
if (args[0]==='install') {
  if (process.env.UI_FIXTURE_MODE==='install-fail') process.exit(42);
  fs.mkdirSync('node_modules', {recursive:true});
} else if (args[0]==='run' && args[1]==='build') {
  fs.mkdirSync('dist', {recursive:true});
  if (process.env.UI_FIXTURE_MODE==='empty-output') process.exit(0);
  fs.writeFileSync('dist/index.html', '<html>fixture</html>\n');
  fs.writeFileSync('dist/environment.json', JSON.stringify({debug:process.env.VITE_MESH_LLM_DEBUG_UI,cuda:process.env.ONNXRUNTIME_NODE_INSTALL_CUDA,settings:Object.fromEntries(Object.entries(process.env).filter(([name])=>name.startsWith("VITE_")||name==="TANSTACK_FILE_ROUTER"))}));
  if (process.env.UI_FIXTURE_MODE==='build-fail') process.exit(7);
  if (process.env.UI_FIXTURE_MODE==='deadline') {
    const child=require('child_process').spawn(process.execPath, ['-e', "const fs=require('fs');setInterval(()=>fs.appendFileSync(process.env.UI_FIXTURE_WRITER,'x'),10)"], {env:process.env,stdio:'inherit'});
    fs.writeFileSync(process.env.UI_FIXTURE_PID,String(child.pid));
    setInterval(()=>{},1000);
  }
  if (process.env.NPM_TOKEN) console.log('token='+process.env.NPM_TOKEN);
} else process.exit(91);
"#;

struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    script: PathBuf,
    trace: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        fs::write(root.join("package.json"), "{}\n").unwrap();
        fs::write(root.join("pnpm-lock.yaml"), "lock\n").unwrap();
        let script = root.join("pnpm-fixture.cjs");
        fs::write(&script, PROVIDER).unwrap();
        let trace = root.join("commands.jsonl");
        Self {
            _temp: temp,
            root,
            script,
            trace,
        }
    }
    fn tool(&self, mode: &str, token: Option<&str>) -> Tool {
        let name = if cfg!(windows) { "node.exe" } else { "node" };
        let search = std::env::var_os("PATH").unwrap();
        let executable = std::env::split_paths(&search)
            .map(|p| p.join(name))
            .find(|p| p.is_file())
            .unwrap()
            .canonicalize()
            .unwrap();
        let mut environment = BTreeMap::from([
            ("PATH".into(), (search, false)),
            (
                "UI_FIXTURE_TRACE".into(),
                (self.trace.clone().into(), false),
            ),
            ("UI_FIXTURE_MODE".into(), (mode.into(), false)),
            (
                "UI_FIXTURE_PID".into(),
                (self.root.join("writer.pid").into(), false),
            ),
            (
                "UI_FIXTURE_WRITER".into(),
                (self.root.join("writer.txt").into(), false),
            ),
        ]);
        if let Some(token) = token {
            environment.insert("NPM_TOKEN".into(), (OsString::from(token), true));
        }
        Tool {
            executable,
            prefix: vec![self.script.clone().into()],
            environment,
        }
    }
    fn run(
        &self,
        mode: &str,
        token: Option<&str>,
        cancellation: &Cancellation,
        seconds: u64,
    ) -> crate::command::DynResult<Decision> {
        execution::run(
            &self.root,
            &build(),
            || Ok(self.tool(mode, token)),
            Duration::from_secs(seconds),
            cancellation,
            &self.root.join("logs"),
        )
    }
    fn trace(&self) -> Vec<Vec<String>> {
        fs::read_to_string(&self.trace)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
    fn stamp(&self) -> PathBuf {
        self.root.join("dist/.mesh-llm-ui-build-env")
    }
}
fn build() -> Environment {
    Environment {
        profile: Profile::Release,
        variables: BTreeMap::from([("VITE_MESH_LLM_DEBUG_UI".into(), "true".into())]),
    }
}
fn refusal<'a>(error: &'a (dyn std::error::Error + 'static)) -> &'a Failure {
    error
        .downcast_ref::<Failure>()
        .unwrap_or_else(|| panic!("expected supervised UI failure, got {error:?}"))
}

#[test]
fn actual_install_build_order_and_release_environment_are_bound() {
    let fixture = Fixture::new();
    assert_eq!(
        fixture
            .run("success", None, &Cancellation::default(), 10)
            .unwrap(),
        Decision::InstallAndBuild
    );
    assert_eq!(
        fixture.trace(),
        vec![vec!["install", "--frozen-lockfile"], vec!["run", "build"]]
    );
    assert_eq!(
        fs::read_to_string(fixture.stamp()).unwrap(),
        build().stamp_for(&fixture.root).unwrap()
    );
    let environment: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture.root.join("dist/environment.json")).unwrap())
            .unwrap();
    assert_eq!(environment["debug"], "false");
    assert_eq!(environment["cuda"], "skip");
}
#[test]
fn actual_fresh_output_reuses_without_any_tool_discovery() {
    let fixture = Fixture::new();
    fixture
        .run("success", None, &Cancellation::default(), 10)
        .unwrap();
    let lock = fixture.root.join(".mesh-llm-ui-build.lock");
    fs::remove_file(&lock).unwrap();
    #[cfg(unix)]
    let _readonly = ReadOnlyUi::new(&fixture.root);
    let result = execution::run(
        &fixture.root,
        &build(),
        || panic!("reuse must not prepare pnpm"),
        Duration::from_secs(10),
        &Cancellation::default(),
        &fixture.root.join("logs"),
    );
    assert_eq!(result.unwrap(), Decision::Reuse);
    assert_eq!(fixture.trace().len(), 2);
    assert!(
        !lock.exists(),
        "fresh reuse must not create coordination files"
    );
}
#[test]
fn actual_install_failure_preserves_status_and_never_builds_or_stamps() {
    let fixture = Fixture::new();
    let error = fixture
        .run("install-fail", None, &Cancellation::default(), 10)
        .unwrap_err();
    let failure = refusal(error.as_ref());
    assert_eq!(failure.code(None), 42);
    assert!(failure.report.cleanup.complete);
    assert_eq!(fixture.trace(), vec![vec!["install", "--frozen-lockfile"]]);
    assert!(!fixture.stamp().exists());
}
#[test]
fn actual_partial_build_failure_cannot_reuse_prior_success_stamp() {
    let fixture = Fixture::new();
    fixture
        .run("success", None, &Cancellation::default(), 10)
        .unwrap();
    // A changed environment forces a build even if source mtimes are unchanged.
    let mut changed = build();
    changed
        .variables
        .insert("VITE_BASE_PATH".into(), "/changed/".into());
    let error = execution::run(
        &fixture.root,
        &changed,
        || Ok(fixture.tool("build-fail", None)),
        Duration::from_secs(10),
        &Cancellation::default(),
        &fixture.root.join("logs"),
    )
    .unwrap_err();
    assert_eq!(refusal(error.as_ref()).code(None), 7);
    assert!(
        refusal(error.as_ref())
            .logs
            .join("build.stderr.log")
            .is_file()
    );
    assert_eq!(fs::read_dir(fixture.root.join("logs")).unwrap().count(), 2);
    assert!(fixture.root.join("dist/index.html").is_file());
    assert!(!fixture.stamp().exists());
    assert_eq!(
        fixture
            .run("success", None, &Cancellation::default(), 10)
            .unwrap(),
        Decision::Build
    );
    assert_eq!(fs::read_dir(fixture.root.join("logs")).unwrap().count(), 3);
    assert_eq!(
        fixture.trace().last().unwrap(),
        &vec!["run".to_owned(), "build".to_owned()]
    );
}
#[test]
fn actual_deadline_cleans_native_descendant_and_leaves_no_success_stamp() {
    let fixture = Fixture::new();
    let error = fixture
        .run("deadline", None, &Cancellation::default(), 2)
        .unwrap_err();
    let failure = refusal(error.as_ref());
    assert_eq!(failure.code(None), 124);
    assert!(failure.report.cleanup.complete);
    assert!(!fixture.stamp().exists());
    let writer = fixture.root.join("writer.txt");
    let before = fs::read(&writer).unwrap();
    std::thread::sleep(Duration::from_millis(80));
    assert_eq!(
        fs::read(writer).unwrap(),
        before,
        "owned descendant must stop writing after cleanup"
    );
}
#[test]
fn cancelled_admission_never_prepares_tool_or_launches_producer() {
    let fixture = Fixture::new();
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let result = execution::run(
        &fixture.root,
        &build(),
        || panic!("cancelled admission must not prepare pnpm"),
        Duration::from_secs(10),
        &cancellation,
        &fixture.root.join("logs"),
    );
    assert!(result.is_err());
    assert!(!fixture.trace.exists());
    assert!(!fixture.stamp().exists());
}
#[test]
fn active_native_lock_prevents_second_producer() {
    let fixture = Fixture::new();
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(fixture.root.join(".mesh-llm-ui-build.lock"))
        .unwrap();
    lock.try_lock().unwrap();
    let error = fixture
        .run("success", None, &Cancellation::default(), 10)
        .unwrap_err();
    assert!(error.to_string().contains("busy"));
    assert!(!fixture.trace.exists());
}
#[test]
fn declared_fixture_secret_is_redacted_from_native_output_files() {
    let fixture = Fixture::new();
    let token = "ui-fixture-token-7dd5ca07-not-a-real-credential";
    fixture
        .run("success", Some(token), &Cancellation::default(), 10)
        .unwrap();
    let attempts = fs::read_dir(fixture.root.join("logs"))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(attempts.len(), 1);
    let logs = attempts[0].path();
    for name in [
        "build.stdout.log",
        "build.stderr.log",
        "install.stdout.log",
        "install.stderr.log",
    ] {
        let bytes = fs::read(logs.join(name)).unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains(token));
    }
}

#[cfg(unix)]
struct ReadOnlyUi {
    path: PathBuf,
    original: fs::Permissions,
}
#[cfg(unix)]
impl ReadOnlyUi {
    fn new(path: &std::path::Path) -> Self {
        use std::os::unix::fs::PermissionsExt;
        let original = fs::metadata(path).unwrap().permissions();
        fs::set_permissions(path, fs::Permissions::from_mode(0o555)).unwrap();
        Self {
            path: path.to_path_buf(),
            original,
        }
    }
}
#[cfg(unix)]
impl Drop for ReadOnlyUi {
    fn drop(&mut self) {
        let _ = fs::set_permissions(&self.path, self.original.clone());
    }
}

#[test]
fn successful_child_without_output_cannot_publish_a_success_stamp() {
    let fixture = Fixture::new();
    let error = fixture
        .run("empty-output", None, &Cancellation::default(), 10)
        .unwrap_err();
    assert!(error.to_string().contains("without regular output files"));
    assert!(!fixture.stamp().exists());
    assert_eq!(
        super::policy::decide(&fixture.root, &build()).unwrap(),
        Decision::Build
    );
}

#[test]
fn actual_producer_preserves_absence_empty_and_all_vite_settings() {
    let fixture = Fixture::new();
    let mut build = build();
    build.variables.extend([
        ("VITE_API_URL".into(), "https://api.example.invalid".into()),
        ("VITE_MANAGEMENT_API_URL".into(), "".into()),
        ("VITE_STORAGE_NAMESPACE".into(), "one\ntwo".into()),
        ("VITE_FUTURE_SETTING".into(), "future".into()),
        ("TANSTACK_FILE_ROUTER".into(), "true".into()),
    ]);
    execution::run(
        &fixture.root,
        &build,
        || Ok(fixture.tool("success", None)),
        Duration::from_secs(10),
        &Cancellation::default(),
        &fixture.root.join("logs"),
    )
    .unwrap();
    let actual: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture.root.join("dist/environment.json")).unwrap())
            .unwrap();
    let settings = &actual["settings"];
    assert!(settings.get("VITE_BASE_PATH").is_none());
    assert!(settings.get("VITE_ROUTER_BASE_PATH").is_none());
    assert_eq!(settings["VITE_MANAGEMENT_API_URL"], "");
    assert_eq!(settings["VITE_API_URL"], "https://api.example.invalid");
    assert_eq!(settings["VITE_STORAGE_NAMESPACE"], "one\ntwo");
    assert_eq!(settings["VITE_FUTURE_SETTING"], "future");
    assert_eq!(settings["TANSTACK_FILE_ROUTER"], "true");
    assert_eq!(settings["VITE_MESH_LLM_DEBUG_UI"], "false");
}
