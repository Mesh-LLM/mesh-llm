//! Actual finite local Git graph and selected-ref CLI; no external fetch/model.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::Value as Json;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
struct Fixture {
    _temp: tempfile::TempDir,
    origin: PathBuf,
    checkout: PathBuf,
    home: PathBuf,
    git: PathBuf,
    url: String,
    initial: String,
    output: PathBuf,
    summary: PathBuf,
}
fn tool(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|path| path.join(name))
        .find(|path| path.is_file())
        .unwrap_or_else(|| panic!("required {name} tool missing"))
}
fn invoke(executable: &Path, cwd: &Path, args: &[String], home: &Path) -> process::ProcessReport {
    let spec = ProcessSpec {
        executable: executable.into(),
        cwd: cwd.into(),
        arguments: args.iter().map(|arg| Value::Public(arg.into())).collect(),
        environment: BTreeMap::from([
            (
                "PATH".into(),
                Value::Public(std::env::var_os("PATH").unwrap()),
            ),
            ("HOME".into(), Value::Public(home.as_os_str().into())),
            ("GIT_MASTER".into(), Value::Public("1".into())),
            ("GIT_TERMINAL_PROMPT".into(), Value::Public("0".into())),
        ]),
    };
    let limits = Limits {
        execution: Duration::from_secs(15),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.cleanup.complete, "{report:?}");
    assert!(!report.stdout.truncated, "{report:?}");
    report
}
impl Fixture {
    fn new() -> Self {
        Self::with_origin_name("origin")
    }
    fn with_origin_name(origin_name: &str) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let origin = temp.path().join(origin_name);
        let checkout = temp.path().join("checkout");
        let home = temp.path().join("home");
        fs::create_dir(&origin).unwrap();
        fs::create_dir(&home).unwrap();
        let git = tool("git");
        let url = format!("file://{}", origin.display());
        let mut fixture = Self {
            _temp: temp,
            origin,
            checkout,
            home,
            git,
            url,
            initial: String::new(),
            output: PathBuf::new(),
            summary: PathBuf::new(),
        };
        fixture.git_at(&fixture.origin, &["init", "-b", "main"]);
        fixture.git_at(&fixture.origin, &["config", "user.name", "Fixture"]);
        fixture.git_at(
            &fixture.origin,
            &["config", "user.email", "fixture@example.invalid"],
        );
        let pin = fixture.origin.join("third_party/llama.cpp/upstream.txt");
        fs::create_dir_all(pin.parent().unwrap()).unwrap();
        fs::write(pin, format!("{}\n", "a".repeat(40))).unwrap();
        fixture.commit("initial");
        fixture.initial = fixture.git_at(&fixture.origin, &["rev-parse", "HEAD"]);
        fixture.git_at(&fixture.origin, &["branch", "candidate"]);
        fixture.git_at(
            &fixture.origin,
            &[
                "clone",
                "--depth=1",
                &fixture.url,
                fixture.checkout.to_str().unwrap(),
            ],
        );
        fixture.output = fixture.home.join("github-output");
        fixture.summary = fixture.home.join("summary");
        fs::write(&fixture.output, "prior-output\n").unwrap();
        fs::write(&fixture.summary, "prior-summary\n").unwrap();
        fixture
    }
    fn git_at(&self, root: &Path, args: &[&str]) -> String {
        let mut argv = vec!["-C".into(), root.to_str().unwrap().into()];
        argv.extend(args.iter().map(|arg| (*arg).into()));
        let report = invoke(&self.git, root, &argv, &self.home);
        assert!(
            report.status.is_some_and(|status| status.success()),
            "{report:?}"
        );
        String::from_utf8(report.stdout.bytes_retained)
            .unwrap()
            .trim()
            .into()
    }
    fn commit(&self, message: &str) {
        self.git_at(&self.origin, &["add", "-A"]);
        self.git_at(&self.origin, &["commit", "-m", message]);
    }
    fn select(
        &self,
        reference: &str,
        event: &str,
        upstream: &str,
        origin: &str,
    ) -> process::ProcessReport {
        let args = [
            "repository",
            "selected-ref",
            "--repository",
            self.checkout.to_str().unwrap(),
            "--ref",
            reference,
            "--event",
            event,
            "--upstream",
            upstream,
            "--expected-origin",
            origin,
            "--github-output",
            self.output.to_str().unwrap(),
            "--summary",
            self.summary.to_str().unwrap(),
            "--timeout-secs",
            "10",
        ]
        .map(str::to_owned);
        invoke(
            Path::new(env!("CARGO_BIN_EXE_xtask")),
            &self.checkout,
            &args,
            &self.home,
        )
    }
    fn pass(&self, reference: &str) -> Json {
        let report = self.select(reference, "workflow_dispatch", "", &self.url);
        assert!(
            report.status.is_some_and(|status| status.success()),
            "{report:?}"
        );
        serde_json::from_slice(&report.stdout.bytes_retained).unwrap()
    }
    fn reject(&self, reference: &str, event: &str, upstream: &str, origin: &str) {
        let before = fs::read(&self.output).unwrap();
        let summary = fs::read(&self.summary).unwrap();
        let report = self.select(reference, event, upstream, origin);
        assert!(
            !report.status.is_some_and(|status| status.success()),
            "{report:?}"
        );
        assert!(report.stdout.bytes_retained.is_empty(), "{report:?}");
        assert_eq!(fs::read(&self.output).unwrap(), before);
        assert_eq!(fs::read(&self.summary).unwrap(), summary);
    }
}
#[test]
fn branch_and_commit_freeze_same_origin_source_pin_without_repair_after_branch_moves() {
    let fixture = Fixture::new();
    let frozen = fixture.pass("candidate");
    assert_eq!(frozen["source"], fixture.initial);
    assert_eq!(frozen["mesh_source"], fixture.initial);
    assert_eq!(frozen["upstream"], "a".repeat(40));
    assert_eq!(frozen["changed"], "false");
    assert_eq!(frozen["mode"], "pinned-build");
    assert_eq!(frozen["certify"], "true");
    fs::write(
        fixture.origin.join("third_party/llama.cpp/upstream.txt"),
        format!("{}\n", "b".repeat(40)),
    )
    .unwrap();
    fixture.commit("advance");
    fixture.git_at(&fixture.origin, &["branch", "-f", "candidate", "HEAD"]);
    let newer = fixture.pass("refs/heads/candidate");
    assert_eq!(newer["source"], fixture.initial);
    assert_ne!(newer["mesh_source"], frozen["mesh_source"]);
    assert_eq!(newer["upstream"], "b".repeat(40));
    assert_eq!(frozen["mesh_source"], fixture.initial);
    assert_eq!(
        fixture.pass(&fixture.initial)["mesh_source"],
        fixture.initial
    );
}
#[test]
fn admission_rejects_invalid_refs_manual_scope_override_and_foreign_origin_without_outputs() {
    let fixture = Fixture::new();
    for reference in [
        "",
        " candidate",
        "refs/pull/1977/head",
        "refs/tags/v1",
        "main~1",
        "--upload-pack=evil",
        &"c".repeat(40),
    ] {
        fixture.reject(reference, "workflow_dispatch", "", &fixture.url);
    }
    fixture.reject("candidate", "schedule", "", &fixture.url);
    fixture.reject("candidate", "workflow_dispatch", "latest", &fixture.url);
    fixture.reject(
        "candidate",
        "workflow_dispatch",
        "",
        "file:///different-origin",
    );
}
#[test]
fn locally_present_commit_outside_fetched_branch_ancestry_is_rejected() {
    let fixture = Fixture::new();
    fixture.git_at(&fixture.checkout, &["config", "user.name", "Fixture"]);
    fixture.git_at(
        &fixture.checkout,
        &["config", "user.email", "fixture@example.invalid"],
    );
    fixture.git_at(
        &fixture.checkout,
        &["commit", "--allow-empty", "-m", "local only"],
    );
    let orphan = fixture.git_at(&fixture.checkout, &["rev-parse", "HEAD"]);
    fixture.reject(&orphan, "workflow_dispatch", "", &fixture.url);
}
#[test]
fn relocated_pin_is_admitted_but_bad_ambiguous_and_missing_pin_fail_before_publication() {
    let fixture = Fixture::new();
    let primary = fixture.origin.join("third_party/llama.cpp/upstream.txt");
    let relocated = fixture
        .origin
        .join("skippy/third_party/llama.cpp/upstream.txt");
    fs::create_dir_all(relocated.parent().unwrap()).unwrap();
    fs::rename(&primary, &relocated).unwrap();
    fixture.commit("relocate");
    assert_eq!(fixture.pass("main")["upstream"], "a".repeat(40));
    fs::write(&relocated, "latest\n").unwrap();
    fixture.commit("bad pin");
    fixture.reject("main", "workflow_dispatch", "", &fixture.url);
    fs::write(&relocated, format!("{}\n", "a".repeat(40))).unwrap();
    fs::write(&primary, format!("{}\n", "b".repeat(40))).unwrap();
    fixture.commit("ambiguous");
    fixture.reject("main", "workflow_dispatch", "", &fixture.url);
    fs::remove_file(primary).unwrap();
    fs::remove_file(relocated).unwrap();
    fixture.commit("missing");
    fixture.reject("main", "workflow_dispatch", "", &fixture.url);
}

#[path = "selected_ref/adversarial.rs"]
mod adversarial;

#[test]
fn valid_local_origin_with_secret_substrings_uses_exact_git_protocol_bytes() {
    let fixture = Fixture::with_origin_name("origin-token-secret");
    let selected = fixture.pass("candidate");
    assert_eq!(selected["source"], fixture.initial);
    assert_eq!(selected["mesh_source"], fixture.initial);
    assert_eq!(selected["upstream"], "a".repeat(40));
}

#[test]
fn pin_symlink_whose_target_text_is_a_valid_sha_is_rejected_before_publication() {
    let fixture = Fixture::new();
    let pin = fixture.origin.join("third_party/llama.cpp/upstream.txt");
    fs::remove_file(&pin).unwrap();
    std::os::unix::fs::symlink("a".repeat(40), &pin).unwrap();
    fixture.commit("symlink pin");
    fixture.reject("main", "workflow_dispatch", "", &fixture.url);
}
