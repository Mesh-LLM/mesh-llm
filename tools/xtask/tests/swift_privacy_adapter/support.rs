use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

const TEMPLATE: &[u8] =
    include_bytes!("../../src/release/swift_privacy/fixtures/PrivacyInfo.xcprivacy");

pub struct Fixture {
    pub root: tempfile::TempDir,
    pub template: PathBuf,
    pub framework: PathBuf,
    behavior: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Invocation {
    arguments: Vec<String>,
    cwd: PathBuf,
    stdin_eof: bool,
    ambient_secret_present: bool,
}

impl Fixture {
    pub fn new(behavior: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        let template = root.path().join("template.xcprivacy");
        let framework = root.path().join("Framework.xcframework");
        fs::write(&template, TEMPLATE).unwrap();
        fs::create_dir(&framework).unwrap();
        fs::write(
            root.path().join("lint-plan.json"),
            serde_json::to_vec(&serde_json::json!({
                "behavior": behavior, "fail_path": null
            }))
            .unwrap(),
        )
        .unwrap();
        Self {
            root,
            template,
            framework,
            behavior: behavior.to_owned(),
        }
    }

    pub fn tool(&self) -> PathBuf {
        std::env::var_os("SPV_ADAPTER_FIXTURE")
            .map(PathBuf::from)
            .expect("build the inert Rust example and set SPV_ADAPTER_FIXTURE to its absolute path")
    }

    pub fn command(&self) -> Command {
        self.command_with_tool(&self.tool())
    }

    pub fn command_with_tool(&self, tool: &Path) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        command
            .current_dir(self.root.path())
            .env_clear()
            .env("SPV_AMBIENT_SECRET", "synthetic-parent-only")
            .args(["release", "swift-privacy", "--template"])
            .arg(&self.template)
            .arg("--xcframework")
            .arg(&self.framework)
            .arg("--plutil")
            .arg(tool)
            .args([
                "--timeout-ms",
                "2000",
                "--grace-ms",
                "200",
                "--cleanup-ms",
                "2000",
                "--max-output-bytes",
                "4096",
            ])
            .stdin(Stdio::null());
        command
    }

    pub fn embed(&self, name: &str, equal: bool) -> PathBuf {
        let directory = self.framework.join(name);
        fs::create_dir(&directory).unwrap();
        let path = directory.join("PrivacyInfo.xcprivacy");
        fs::write(&path, if equal { TEMPLATE } else { b"different" }).unwrap();
        path
    }

    pub fn fail_at(&self, path: &Path) {
        fs::write(
            self.root.path().join("lint-plan.json"),
            serde_json::to_vec(&serde_json::json!({
                "behavior": self.behavior, "fail_path": path
            }))
            .unwrap(),
        )
        .unwrap();
    }

    pub fn discovery_order(&self) -> Vec<PathBuf> {
        fs::read_dir(&self.framework)
            .unwrap()
            .map(|entry| entry.unwrap().path().join("PrivacyInfo.xcprivacy"))
            .collect()
    }

    pub fn invocations(&self) -> Vec<PathBuf> {
        let trace = self.root.path().join("lint-trace.jsonl");
        if !trace.exists() {
            return Vec::new();
        }
        fs::read_to_string(trace)
            .unwrap()
            .lines()
            .map(|line| {
                let invocation: Invocation = serde_json::from_str(line).unwrap();
                assert_eq!(invocation.cwd, self.root.path().canonicalize().unwrap());
                assert!(invocation.stdin_eof);
                assert!(!invocation.ambient_secret_present);
                assert_eq!(invocation.arguments.len(), 2);
                assert_eq!(invocation.arguments[0], "-lint");
                PathBuf::from(&invocation.arguments[1])
            })
            .collect()
    }

    pub fn template_stdout(&self) -> Vec<u8> {
        format!(
            "verified Swift privacy manifest: {}\n",
            self.template.display()
        )
        .into_bytes()
    }

    pub fn success_stdout(&self, count: usize) -> Vec<u8> {
        let mut bytes = self.template_stdout();
        bytes.extend(
            format!(
                "verified {count} embedded privacy manifest file(s) in {}\n",
                self.framework.display()
            )
            .bytes(),
        );
        bytes
    }

    pub fn await_file(&self, name: &str, child: &mut Child) {
        let deadline = Instant::now() + Duration::from_secs(4);
        while !self.root.path().join(name).exists() {
            assert!(
                child.try_wait().unwrap().is_none(),
                "child exited before {name}"
            );
            assert!(Instant::now() < deadline, "fixture did not publish {name}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

pub struct OwnedChild(pub Child);

impl Drop for OwnedChild {
    fn drop(&mut self) {
        if self.0.try_wait().unwrap().is_none() {
            self.0.kill().unwrap();
            self.0.wait().unwrap();
        }
    }
}
