use serde_json::Value;
use sha2::Digest;
use std::error::Error;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(crate) type TestResult = Result<(), Box<dyn Error>>;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A per-case scratch directory removed on drop.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-native-{label}-{}-{sequence}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self(path.canonicalize()?))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _cleanup = fs::remove_dir_all(&self.0);
    }
}

/// The legacy script a `native` command replaces.
#[derive(Clone, Copy)]
pub(crate) enum Tool {
    SelectRuntime,
    HostDependencies,
    LinuxDeps,
    WindowsDeps,
    ReleaseMatrix,
    RuntimePackage,
}

impl Tool {
    fn command(self) -> &'static str {
        match self {
            Self::SelectRuntime => "select-runtime",
            Self::HostDependencies => "verify-host-dependencies",
            Self::LinuxDeps => "linux-runtime-deps",
            Self::WindowsDeps => "windows-runtime-deps",
            Self::ReleaseMatrix => "release-matrix",
            Self::RuntimePackage => "verify-runtime-package",
        }
    }

    fn fixture(self) -> &'static str {
        match self {
            Self::SelectRuntime => "select_runtime.json",
            Self::HostDependencies => "host_dependencies.json",
            Self::LinuxDeps => "linux_deps.json",
            Self::WindowsDeps => "windows_deps.json",
            Self::ReleaseMatrix => "release_matrix.json",
            Self::RuntimePackage => "runtime_package.json",
        }
    }
}

/// Every case of a fixture document whose `group` is `group`.
pub(crate) fn cases(tool: Tool, group: &str) -> Result<Vec<Value>, Box<dyn Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/native_policy")
        .join(tool.fixture());
    let document: Value = serde_json::from_slice(&fs::read(path)?)?;
    let cases: Vec<Value> = document["cases"]
        .as_array()
        .ok_or("fixture lacks cases")?
        .iter()
        .filter(|case| case["group"] == group)
        .cloned()
        .collect();
    assert!(!cases.is_empty(), "no {group} cases");
    Ok(cases)
}

fn text(value: &Value) -> &str {
    value.as_str().unwrap_or_default()
}

/// Lays out one case: `files` (path to text, or `{"hex": ..}` for raw
/// bytes) and `tools` (stub inspection tools placed alone on `PATH`, each
/// printing its canned `output` and exiting with `code`).
fn build(root: &Path, case: &Value) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(package) = case.get("package") {
        let id = text(&package["id"]);
        let artifact = root.join(id);
        fs::create_dir_all(artifact.join("lib"))?;
        let library = artifact.join("lib/llama.bin");
        let content = package["hex"]
            .as_str()
            .map(hex::decode)
            .transpose()?
            .unwrap_or_else(|| text(&package["content"]).as_bytes().to_vec());
        fs::write(&library, content)?;
        if case.get("pes").is_some() {
            crate::pe_stub::build(root, case)?;
            if let Some(name) = package.get("library_name").and_then(Value::as_str) {
                fs::copy(artifact.join("lib").join(name), &library)?;
            }
        }
        let checksum = hex::encode(sha2::Sha256::digest(fs::read(&library)?));
        let target = text(&package["target"]);
        let os = text(&package["os"]);
        let arch = text(&package["arch"]);
        let mut manifest = serde_json::json!({
            "runtime": {"id": id, "mesh_version": "0.75.0", "skippy_abi": "0.1.32", "platform": {"os":os,"arch":arch,"target":target}, "backend":{"kind":"cpu"}, "libraries":["lib/llama.bin"], "files":{"lib/llama.bin":checksum}},
            "build":{"primary_library":"lib/llama.bin","library_sha256":checksum}
        });
        if let Some(floor) = package.get("min_glibc") {
            manifest["runtime"]["platform"]["min_glibc"] = floor.clone();
        }
        if let Some(backend) = package.get("backend") {
            manifest["runtime"]["backend"] = backend.clone();
        }
        if let Some(name) = package.get("library_name").and_then(Value::as_str) {
            if case.get("pes").is_some() {
                fs::remove_file(&library)?;
            } else {
                fs::rename(&library, artifact.join("lib").join(name))?;
            }
            let relative = format!("lib/{name}");
            manifest["runtime"]["libraries"] = serde_json::json!([relative]);
            manifest["runtime"]["files"] = serde_json::json!({relative.clone():checksum});
            manifest["build"]["primary_library"] = serde_json::json!(relative);
        }
        if let Some(tool) = package.get("tool").and_then(Value::as_str) {
            let path = artifact.join("tools").join(tool);
            fs::create_dir_all(path.parent().ok_or("tool without parent")?)?;
            let content = package
                .get("tool_hex")
                .and_then(Value::as_str)
                .map(hex::decode)
                .transpose()?
                .unwrap_or_else(|| b"probe".to_vec());
            fs::write(&path, content)?;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
            manifest["runtime"]["tools"] = serde_json::json!({format!("tools/{tool}"): hex::encode(sha2::Sha256::digest(fs::read(&path)?))});
        }
        if let Some(mutation) = package.get("mutate").and_then(Value::as_str) {
            match mutation {
                "checksum" => {
                    let primary = manifest["build"]["primary_library"]
                        .as_str()
                        .ok_or("primary library missing")?
                        .to_owned();
                    manifest["runtime"]["files"][&primary] = serde_json::json!("0".repeat(64));
                }
                "traversal" => {
                    manifest["runtime"]["libraries"] = serde_json::json!(["../outside.bin"])
                }
                _ => return Err(format!("unknown package mutation: {mutation}").into()),
            }
        }
        if package["relocatable"].as_bool() == Some(false) {
            manifest["build"]["relocatable_libraries"] = serde_json::json!(["lib/not-primary.so"]);
        }
        fs::write(
            artifact.join("manifest.json"),
            serde_json::to_vec(&manifest)?,
        )?;
    }
    for (relative, content) in case["files"].as_object().into_iter().flatten() {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().ok_or("file without parent")?)?;
        match content.get("hex") {
            Some(hex) => fs::write(&path, hex::decode(text(hex))?)?,
            None => fs::write(&path, text(content))?,
        }
    }
    for (relative, destination) in case["directory_links"].as_object().into_iter().flatten() {
        let link = root.join(relative);
        fs::create_dir_all(link.parent().ok_or("link without parent")?)?;
        std::os::unix::fs::symlink(text(destination), link)?;
    }
    let bin = root.join(".stub-bin");
    fs::create_dir_all(&bin)?;
    for (name, stub) in case["tools"].as_object().into_iter().flatten() {
        let output = bin.join(format!("{name}.out"));
        fs::write(&output, text(&stub["output"]))?;
        let script = format!(
            "#!/bin/sh\nprintf '%s %s\\n' \"${{0##*/}}\" \"$*\" >> '{}'\n/bin/cat '{}'\nexit {}\n",
            bin.join("calls.log").display(),
            output.display(),
            stub["code"].as_i64().unwrap_or(0)
        );
        let path = bin.join(name);
        fs::write(&path, script)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
    }
    crate::elf_stub::build(root, &bin, case)?;
    if case.get("package").is_none() {
        crate::pe_stub::build(root, case)?;
    }
    Ok(bin)
}

/// The observable result of one run.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Outcome {
    pub(crate) code: Option<i32>,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    /// The `--report` file, when the case names one.
    pub(crate) report: Option<String>,
    /// Stub invocations, one argv line each.
    pub(crate) calls: String,
}

fn execute(tool: Tool, case: &Value) -> Result<Outcome, Box<dyn Error>> {
    let scratch = Scratch::new(tool.command())?;
    let root = scratch.path();
    let bin = build(root, case)?;
    let args: Vec<&str> = case["args"]
        .as_array()
        .ok_or("case lacks args")?
        .iter()
        .map(text)
        .collect();
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command.args(["native", tool.command()]);
    let mut child = command
        .args(&args)
        .current_dir(root)
        .env(
            "PATH",
            if matches!(tool, Tool::RuntimePackage) {
                std::env::join_paths([
                    bin.as_path(),
                    Path::new("/opt/homebrew/opt/python@3.13/libexec/bin"),
                    Path::new("/usr/bin"),
                    Path::new("/bin"),
                ])?
            } else {
                bin.as_os_str().to_owned()
            },
        )
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    if case["name"] == "macos-directory-cycle" {
        let deadline = Instant::now() + Duration::from_secs(3);
        while child.try_wait()?.is_none() {
            if Instant::now() >= deadline {
                child.kill()?;
                child.wait()?;
                return Err("macOS directory-cycle verifier exceeded three seconds".into());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    let output = child.wait_with_output()?;
    let report = case["report"]
        .as_str()
        .map(|relative| fs::read_to_string(root.join(relative)).unwrap_or_default());
    Ok(Outcome {
        code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: last_line_if_traceback(&String::from_utf8_lossy(&output.stderr)),
        report,
        calls: fs::read_to_string(bin.join("calls.log")).unwrap_or_default(),
    })
}

/// An uncaught exception is compared by its traceback's final line; the
/// frames name interpreter paths the port does not reproduce.
fn last_line_if_traceback(stderr: &str) -> String {
    if !stderr.starts_with("Traceback (most recent call last):\n") {
        return stderr.to_owned();
    }
    let last = stderr.trim_end_matches('\n').rsplit('\n').next();
    format!("{}\n", last.unwrap_or_default())
}

fn expected(case: &Value) -> Outcome {
    Outcome {
        code: case["code"]
            .as_i64()
            .and_then(|code| i32::try_from(code).ok()),
        stdout: text(&case["stdout"]).to_owned(),
        stderr: text(&case["stderr"]).to_owned(),
        report: case["report"]
            .as_str()
            .map(|_| text(&case["report_text"]).to_owned()),
        calls: text(&case["calls"]).to_owned(),
    }
}

/// Runs every case through the port and requires the checked-in golden.
pub(crate) fn check(tool: Tool, group: &str) -> TestResult {
    for case in cases(tool, group)? {
        let name = text(&case["name"]);
        let golden = expected(&case);
        let ported = execute(tool, &case)?;
        if golden.code != Some(0) {
            assert_eq!(ported.code, golden.code, "{name}");
            assert_eq!(ported.stdout, golden.stdout, "{name}");
            assert_eq!(ported.report, golden.report, "{name}");
            assert_eq!(ported.calls, golden.calls, "{name}");
            assert!(!ported.stderr.is_empty());
        } else {
            assert_eq!(ported, golden, "output contract for {name}");
        }
    }
    Ok(())
}
