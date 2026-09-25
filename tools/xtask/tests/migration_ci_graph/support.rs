use serde_json::{Map, Value, json};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) type TestResult = Result<(), Box<dyn Error>>;

pub(crate) const LEGACY_ENV: &str = "MIGRATION_CI_GRAPH_LEGACY_PYTHON";
const LEGACY_SCRIPT: &str = "scripts/validate-ci-lane-results.py";

pub(crate) const LANES: [&str; 5] = ["quality", "website", "linux", "macos", "windows"];

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("xtask lives under tools/")
        .to_path_buf()
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub(crate) fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Every top-level job of a lane workflow except its summary, as the
/// summary's `needs` lists them.
pub(crate) fn lane_jobs(lane: &str) -> &'static [&'static str] {
    match lane {
        "quality" => &["quality", "runner_contract"],
        "website" => &["web"],
        "linux" => &[
            "ui_artifact",
            "static_abi",
            "rust_tests",
            "hosts",
            "native_runtimes",
            "runtime_product",
            "kotlin_sdk_input",
            "sdk",
            "product_smoke",
        ],
        "macos" => &[
            "validate_plan",
            "ui_artifact",
            "hosts",
            "native_runtimes",
            "runtime_product",
            "platform_checks",
            "swift_sdk_input",
            "sdk",
            "product_smoke",
        ],
        _ => &[
            "ui_artifact",
            "hosts",
            "native_runtimes",
            "runtime_product",
            "platform_checks",
        ],
    }
}

pub(crate) fn lane_workflow(lane: &str) -> PathBuf {
    repository_root().join(format!(".github/workflows/ci-{lane}-lane.yml"))
}

/// One frozen planner golden: its `$GITHUB_OUTPUT` lines by name.
pub(crate) struct Golden(Map<String, Value>);

impl Golden {
    pub(crate) fn load(case: &str) -> Result<Self, Box<dyn Error>> {
        let path = fixtures().join(format!("ci_plan/expected/{case}.outputs.txt"));
        let mut outputs = Map::new();
        for line in fs::read_to_string(path)?.lines() {
            if let Some((name, value)) = line.split_once('=') {
                outputs.insert(name.to_owned(), Value::String(value.to_owned()));
            }
        }
        Ok(Self(outputs))
    }

    pub(crate) fn output(&self, name: &str) -> &str {
        self.0.get(name).and_then(Value::as_str).unwrap_or("")
    }

    pub(crate) fn lane_plan(&self, lane: &str) -> String {
        self.output(&format!("{lane}_lane_plan")).to_owned()
    }
}

/// Planned jobs per case and lane, derived by jq from the legacy rules.
pub(crate) fn expected_jobs() -> Result<Map<String, Value>, Box<dyn Error>> {
    let path = fixtures().join("ci_graph/lane_jobs.json");
    match serde_json::from_slice(&fs::read(path)?)? {
        Value::Object(cases) => Ok(cases),
        _ => Err("lane_jobs.json must be an object".into()),
    }
}

/// `toJson(needs)` for a lane where exactly `planned` succeeded.
pub(crate) fn needs(lane: &str, planned: &[&str]) -> String {
    let states = lane_jobs(lane)
        .iter()
        .map(|job| {
            let result = if planned.contains(job) {
                "success"
            } else {
                "skipped"
            };
            ((*job).to_owned(), json!({"outputs": {}, "result": result}))
        })
        .collect::<Map<_, _>>();
    Value::Object(states).to_string()
}

/// A per-test scratch directory, removed on drop.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn new(label: &str) -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-migration-ci-graph-{label}-{}-{sequence}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    pub(crate) fn mutated(
        &self,
        source: &Path,
        edits: &[(&str, &str)],
    ) -> Result<PathBuf, Box<dyn Error>> {
        mutate(source, &self.0, edits)
    }
}

/// Writes `source` into `directory` with every `(from, to)` replacement
/// applied. Each `from` must be present, so a stale anchor fails the test
/// instead of silently exercising the unmodified workflow.
fn mutate(
    source: &Path,
    directory: &Path,
    edits: &[(&str, &str)],
) -> Result<PathBuf, Box<dyn Error>> {
    let mut workflow = fs::read_to_string(source)?;
    for (from, to) in edits {
        if !workflow.contains(from) {
            return Err(format!("mutation anchor missing: {from}").into());
        }
        workflow = workflow.replace(from, to);
    }
    let name = source.file_name().ok_or("workflow path has no name")?;
    let target = directory.join(name);
    fs::write(&target, workflow)?;
    Ok(target)
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _cleanup = fs::remove_dir_all(&self.0);
    }
}

/// One `ci validate-lane` call: the legacy argv plus Rust-only options.
#[derive(Default)]
pub(crate) struct Call {
    pub(crate) legacy: Vec<String>,
    pub(crate) extra: Vec<String>,
}

impl Call {
    pub(crate) fn lane(lane_plan: &str, needs: &str) -> Self {
        Self {
            legacy: vec![
                "--lane-plan".into(),
                lane_plan.into(),
                "--needs".into(),
                needs.into(),
            ],
            extra: Vec::new(),
        }
    }

    pub(crate) fn raw(args: &[&str]) -> Self {
        Self {
            legacy: args.iter().map(|arg| (*arg).to_owned()).collect(),
            extra: Vec::new(),
        }
    }

    pub(crate) fn with(mut self, flag: &str, value: &Path) -> Self {
        self.extra.push(flag.into());
        self.extra.push(value.to_string_lossy().into_owned());
        self
    }

    pub(crate) fn with_text(mut self, flag: &str, value: &str) -> Self {
        self.extra.push(flag.into());
        self.extra.push(value.into());
        self
    }

    fn rust(&self) -> Result<Output, Box<dyn Error>> {
        Ok(Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(repository_root())
            .args(["ci", "validate-lane"])
            .args(&self.legacy)
            .args(&self.extra)
            .env_remove(LEGACY_ENV)
            .output()?)
    }

    /// Runs the port; when `MIGRATION_CI_GRAPH_LEGACY_PYTHON` is set, also
    /// runs the legacy script on the legacy argv and requires identical
    /// streams and status.
    pub(crate) fn run(&self) -> Result<Output, Box<dyn Error>> {
        let ported = self.rust()?;
        if let Some(python) = std::env::var_os(LEGACY_ENV) {
            let legacy = Command::new(python)
                .current_dir(repository_root())
                .arg(LEGACY_SCRIPT)
                .args(&self.legacy)
                .output()?;
            assert_eq!(text(&legacy.stdout), text(&ported.stdout), "stdout parity");
            assert_eq!(text(&legacy.stderr), text(&ported.stderr), "stderr parity");
            assert_eq!(legacy.status.code(), ported.status.code(), "status parity");
        }
        Ok(ported)
    }

    /// Runs only the port: for Rust-only contracts the legacy script lacks.
    pub(crate) fn run_rust_only(&self) -> Result<Output, Box<dyn Error>> {
        self.rust()
    }
}

pub(crate) fn validate_entrypoints(workflows: &Path) -> Result<Output, Box<dyn Error>> {
    Ok(Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(repository_root())
        .args(["ci", "validate-graph", "--workflows"])
        .arg(workflows)
        .output()?)
}

pub(crate) fn entrypoint_files() -> Vec<String> {
    let mut files = vec!["ci-control.yml".to_owned()];
    for kind in ["pr", "main"] {
        files.extend(LANES.iter().map(|lane| format!("{kind}_{lane}.yml")));
    }
    files.extend(LANES.iter().map(|lane| format!("ci-{lane}-lane.yml")));
    files
}

impl Scratch {
    pub(crate) fn entrypoints(
        &self,
        file: &str,
        edits: &[(&str, &str)],
    ) -> Result<PathBuf, Box<dyn Error>> {
        let workflows = repository_root().join(".github/workflows");
        let target = self.0.join("workflows");
        fs::create_dir_all(&target)?;
        for name in entrypoint_files() {
            fs::copy(workflows.join(&name), target.join(&name))?;
        }
        mutate(&workflows.join(file), &target, edits)?;
        Ok(target)
    }
}

pub(crate) fn assert_output(output: &Output, code: i32, stderr: &str) {
    assert_eq!(text(&output.stdout), "", "stdout");
    assert_eq!(text(&output.stderr), stderr, "stderr");
    assert_eq!(output.status.code(), Some(code), "status");
}
