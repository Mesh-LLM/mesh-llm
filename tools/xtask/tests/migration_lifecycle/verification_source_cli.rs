//! Actual independent candidate admission CLI over finite real local Git worktrees.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
fn run(executable: &Path, arguments: Vec<String>, cwd: &Path) -> process::RawProcessReport {
    let environment = [
        ("PATH", "/usr/bin:/bin"),
        ("LC_ALL", "C"),
        ("GIT_MASTER", "1"),
        ("GIT_CONFIG_NOSYSTEM", "1"),
        ("GIT_CONFIG_GLOBAL", "/dev/null"),
        ("GIT_TERMINAL_PROMPT", "0"),
        ("GIT_ALLOW_PROTOCOL", "file"),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), Value::Public(v.into())))
    .collect::<BTreeMap<_, _>>();
    process::supervise_raw(
        &ProcessSpec {
            executable: executable.into(),
            cwd: cwd.into(),
            environment,
            arguments: arguments
                .into_iter()
                .map(|a| Value::Public(a.into()))
                .collect(),
        },
        &Limits {
            execution: Duration::from_secs(30),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap()
}
fn git(root: &Path, args: &[&str]) -> String {
    let mut arguments = vec![
        "--no-optional-locks".into(),
        "-c".into(),
        "core.hooksPath=/dev/null".into(),
    ];
    arguments.extend(args.iter().map(|a| (*a).to_owned()));
    let report = run(Path::new("/usr/bin/git"), arguments, root);
    assert!(report.process.success(), "{:?}", report.process);
    String::from_utf8(report.stdout.unwrap().as_bytes().to_vec())
        .unwrap()
        .trim()
        .to_owned()
}
fn commit(root: &Path) -> String {
    git(root, &["add", "-A"]);
    git(
        root,
        &[
            "-c",
            "user.name=Verification Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
    );
    git(root, &["rev-parse", "HEAD"])
}
struct Fixture {
    _temp: tempfile::TempDir,
    work: PathBuf,
    input: PathBuf,
    document: Json,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let work = temp.path().canonicalize().unwrap();
        let controller = work.join("controller");
        let candidate = work.join("candidate");
        fs::create_dir_all(controller.join("src")).unwrap();
        fs::write(controller.join(".gitignore"), "target/\n.deps/\n").unwrap();
        fs::write(controller.join("src/lib.rs"), "base\n").unwrap();
        git(&controller, &["init", "--quiet"]);
        let base = commit(&controller);
        fs::write(controller.join("src/lib.rs"), "candidate\n").unwrap();
        let head = commit(&controller);
        let tree = git(&controller, &["rev-parse", "HEAD^{tree}"]);
        git(
            &controller,
            &[
                "worktree",
                "add",
                "--quiet",
                "--detach",
                candidate.to_str().unwrap(),
                &head,
            ],
        );
        git(&controller, &["checkout", "--quiet", "--detach", &base]);
        fs::write(controller.join("trusted-note.txt"), "controller advanced\n").unwrap();
        let revision = commit(&controller);
        let digest = hex::encode(Sha256::digest(
            fs::read(env!("CARGO_BIN_EXE_xtask")).unwrap(),
        ));
        let document = json!({"authority":{"controller":{"root":controller,"revision":revision,"executable_sha256":digest},
            "root":candidate,"base":base,"candidate":head,"tree":tree}});
        let input = work.join("input.json");
        Self {
            _temp: temp,
            work,
            input,
            document,
        }
    }
    fn cli(&self, document: &Json) -> process::RawProcessReport {
        fs::write(&self.input, serde_json::to_vec(document).unwrap()).unwrap();
        run(
            Path::new(env!("CARGO_BIN_EXE_xtask")),
            vec![
                "automation".into(),
                "canary-receipts".into(),
                "verification-source-admit".into(),
                "--input".into(),
                self.input.to_str().unwrap().into(),
            ],
            &self.work,
        )
    }
}
#[test]
fn actual_cli_admits_only_exact_independent_candidate_with_advanced_controller() {
    let f = Fixture::new();
    let report = f.cli(&f.document);
    assert!(report.process.success(), "{:?}", report.process);
    let value: Json = serde_json::from_slice(report.stdout.unwrap().as_bytes()).unwrap();
    assert_eq!(value["status"], "verification_source_admitted");
    assert_eq!(value["candidate"], f.document["authority"]["candidate"]);
    assert_eq!(value["tree"], f.document["authority"]["tree"]);
    assert!(value.get("run_id").is_none());
}
#[test]
fn actual_cli_rejects_wrong_tree_base_executable_and_workflow_authority_without_success_output() {
    let f = Fixture::new();
    for field in ["tree", "base", "executable", "workflow"] {
        let mut document = f.document.clone();
        match field {
            "tree" => document["authority"]["tree"] = json!("0".repeat(40)),
            "base" => {
                document["authority"]["base"] =
                    document["authority"]["controller"]["revision"].clone()
            }
            "executable" => {
                document["authority"]["controller"]["executable_sha256"] = json!("f".repeat(64))
            }
            "workflow" => document["authority"]["run_id"] = json!("123"),
            _ => unreachable!(),
        }
        let report = f.cli(&document);
        assert_eq!(report.process.outcome, process::Outcome::Exited);
        assert!(!report.process.success(), "admitted {field}");
        assert!(
            report.stdout.unwrap().as_bytes().is_empty(),
            "{field}: emitted success evidence"
        );
        assert!(
            String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
                .contains("verification source rejected")
        );
    }
}
#[test]
fn actual_cli_owns_bounded_input_and_domain_help_without_legacy_transaction_routing() {
    use std::os::unix::ffi::OsStrExt;
    let temp = tempfile::tempdir().unwrap();
    let pipe = temp.path().join("pipe");
    let encoded = std::ffi::CString::new(pipe.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(encoded.as_ptr(), 0o600) }, 0);
    let report = run(
        Path::new(env!("CARGO_BIN_EXE_xtask")),
        vec![
            "automation".into(),
            "canary-receipts".into(),
            "verification-source-admit".into(),
            "--input".into(),
            pipe.to_str().unwrap().into(),
        ],
        temp.path(),
    );
    assert_eq!(report.process.outcome, process::Outcome::Exited);
    assert!(!report.process.success());
    assert!(report.stdout.unwrap().as_bytes().is_empty());
    assert!(
        String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).contains("regular JSON file")
    );
    let help = run(
        Path::new(env!("CARGO_BIN_EXE_xtask")),
        vec![
            "automation".into(),
            "canary-receipts".into(),
            "verification-source-admit".into(),
            "--help".into(),
        ],
        temp.path(),
    );
    assert!(help.process.success());
    assert!(
        String::from_utf8_lossy(help.stdout.unwrap().as_bytes())
            .contains("verification-source-admit --input PATH")
    );
}

#[test]
fn actual_verification_inspection_routes_own_help_and_reject_special_input_before_content() {
    use std::os::unix::ffi::OsStrExt;
    let temp = tempfile::tempdir().unwrap();
    let pipe = temp.path().join("pipe");
    let encoded = std::ffi::CString::new(pipe.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(encoded.as_ptr(), 0o600) }, 0);
    for verb in [
        "verification-manifest-policy",
        "verification-parity-inventory",
        "verification-split-roster-check",
    ] {
        let report = run(
            Path::new(env!("CARGO_BIN_EXE_xtask")),
            vec![
                "automation".into(),
                "canary-receipts".into(),
                verb.into(),
                "--input".into(),
                pipe.to_str().unwrap().into(),
            ],
            temp.path(),
        );
        assert_eq!(report.process.outcome, process::Outcome::Exited);
        assert!(!report.process.success());
        assert!(report.stdout.unwrap().as_bytes().is_empty());
        assert!(
            String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
                .contains("regular JSON file")
        );
        let help = run(
            Path::new(env!("CARGO_BIN_EXE_xtask")),
            vec![
                "automation".into(),
                "canary-receipts".into(),
                verb.into(),
                "--help".into(),
            ],
            temp.path(),
        );
        assert!(help.process.success());
        assert!(
            String::from_utf8_lossy(help.stdout.unwrap().as_bytes())
                .contains(&format!("{verb} --input PATH"))
        );
    }
}

fn caller_function(source: &str, name: &str) -> String {
    let marker = format!("{name}() {{\n");
    assert_eq!(source.matches(&marker).count(), 1, "duplicate {name}");
    let body = source.split_once(&marker).unwrap().1;
    let end = if name == "repair_workload_controller_unchanged" {
        "\n  }\n"
    } else {
        "\n}\n"
    };
    format!("{marker}{}{end}", body.split_once(end).unwrap().0)
}
fn copied_verification_helpers() -> String {
    let source = include_str!("../../../../scripts/llama-canary-agent-repair.sh");
    let frozen = source
        .find("# Legacy workload automation selection begins.")
        .unwrap();
    let load = source.find("\nload_candidate_bundle\n").unwrap();
    assert!(
        frozen < load,
        "controller must freeze before candidate import"
    );
    [
        "repair_workload_controller_unchanged",
        "verification_source_inspection",
        "verification_candidate_unchanged",
        "repair_family_plan_step",
        "run_for",
    ]
    .map(|name| caller_function(source, name))
    .join("\n")
}
impl Fixture {
    fn wrapper(&self, document: &Json, action: &str) -> process::RawProcessReport {
        // Only the logging adapter is a fixture stub; copied helper launches the actual xtask API.
        let script = format!(
            "set -eu\nHARNESS_MODE=verify\nrepair_workload_controller=$1\nrepair_workload_automation=(\"$1\")\nTRUSTED_ROOT=$2\nBASE_HEAD=$3\nrepair_workload_controller_sha=$4\nROOT=$5\nCANDIDATE_BASE_HEAD=$6\nCERTIFIED_SHA=$7\nVERIFICATION_TREE=$8\nRUNNER_TEMP=$9\nPATH=\"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin\"\nexport PATH\nrun_verification_logged() {{ shift 2; \"$@\"; }}\n{}\n{}\n",
            copied_verification_helpers(),
            action
        );
        let a = &document["authority"];
        let c = &a["controller"];
        let arguments = vec![
            "-c".into(),
            script,
            "copied-verification-wrapper".into(),
            env!("CARGO_BIN_EXE_xtask").into(),
            c["root"].as_str().unwrap().into(),
            c["revision"].as_str().unwrap().into(),
            c["executable_sha256"].as_str().unwrap().into(),
            a["root"].as_str().unwrap().into(),
            a["base"].as_str().unwrap().into(),
            a["candidate"].as_str().unwrap().into(),
            a["tree"].as_str().unwrap().into(),
            self.work.to_str().unwrap().into(),
        ];
        run(Path::new("/bin/bash"), arguments, &self.work)
    }
    fn no_transport_temps(&self) {
        for entry in fs::read_dir(&self.work).unwrap() {
            let name = entry.unwrap().file_name();
            let name = name.to_string_lossy();
            assert!(!name.starts_with("independent-verification."));
            assert!(!name.starts_with("canary-timeout."));
        }
    }
}
#[test]
fn copied_verify_wrapper_keeps_controller_and_candidate_authorities_independent() {
    let f = Fixture::new();
    let report = f.wrapper(
        &f.document,
        "verification_source_inspection verification-source-admit",
    );
    assert!(report.process.success(), "{:?}", report.process);
    let value: Json = serde_json::from_slice(report.stdout.unwrap().as_bytes()).unwrap();
    assert_eq!(value["candidate"], f.document["authority"]["candidate"]);
    f.no_transport_temps();
    for field in ["tree", "base", "digest", "root"] {
        let mut document = f.document.clone();
        match field {
            "tree" => document["authority"]["tree"] = json!("0".repeat(40)),
            "base" => {
                document["authority"]["base"] =
                    document["authority"]["controller"]["revision"].clone()
            }
            "digest" => {
                document["authority"]["controller"]["executable_sha256"] = json!("f".repeat(64))
            }
            "root" => {
                document["authority"]["root"] = document["authority"]["controller"]["root"].clone()
            }
            _ => unreachable!(),
        }
        let report = f.wrapper(
            &document,
            "verification_source_inspection verification-source-admit",
        );
        assert_eq!(report.process.outcome, process::Outcome::Exited);
        assert!(!report.process.success(), "admitted {field}");
        assert!(report.stdout.unwrap().as_bytes().is_empty());
        f.no_transport_temps();
    }
    let report = f.wrapper(
        &f.document,
        "verification_source_inspection local-manifest-policy",
    );
    assert!(
        !report.process.success(),
        "normal verify cannot select local repair owner"
    );
    f.no_transport_temps();
}
#[test]
fn copied_plan_step_revalidates_after_failed_child_and_preserves_clean_child_status() {
    let f = Fixture::new();
    let clean = f.wrapper(
        &f.document,
        "repair_family_plan_step '' /bin/bash -c 'exit 23'",
    );
    assert_eq!(clean.process.outcome, process::Outcome::Exited);
    assert_eq!(
        clean.process.status.and_then(|status| status.code()),
        Some(23)
    );
    let dirty = f.wrapper(&f.document,
        "repair_family_plan_step '' /bin/bash -c 'printf changed >> \"$1/src/lib.rs\"; exit 23' child \"$ROOT\"");
    assert_eq!(dirty.process.outcome, process::Outcome::Exited);
    assert_eq!(
        dirty.process.status.and_then(|status| status.code()),
        Some(1)
    );
    assert!(
        String::from_utf8_lossy(dirty.stderr.unwrap().as_bytes())
            .contains("verification source rejected")
    );
    f.no_transport_temps();
}
#[test]
fn copied_timeout_uses_frozen_typed_owner_and_preserves_child_argv_status() {
    let f = Fixture::new();
    let report = f.wrapper(&f.document,
        "run_for 'finite verify fixture' 5 /bin/bash -c 'test \"$1\" = \"literal space\" && exit 23' child 'literal space'");
    assert_eq!(report.process.outcome, process::Outcome::Exited);
    assert_eq!(
        report.process.status.and_then(|status| status.code()),
        Some(23)
    );
    f.no_transport_temps();
    let mut document = f.document.clone();
    document["authority"]["controller"]["executable_sha256"] = json!("f".repeat(64));
    let rejected = f.wrapper(&document, "run_for 'finite verify fixture' 5 /usr/bin/true");
    assert_eq!(
        rejected.process.status.and_then(|status| status.code()),
        Some(125)
    );
    f.no_transport_temps();
}

#[test]
fn copied_verify_content_callers_refuse_wrong_tree_before_content_work() {
    let f = Fixture::new();
    let mut document = f.document.clone();
    document["authority"]["tree"] = json!("0".repeat(40));
    for verb in [
        "verification-manifest-policy",
        "verification-parity-inventory",
        "verification-split-roster-check",
    ] {
        let report = f.wrapper(&document, &format!("verification_source_inspection {verb}"));
        assert_eq!(report.process.outcome, process::Outcome::Exited);
        assert!(!report.process.success());
        assert!(report.stdout.unwrap().as_bytes().is_empty());
        assert!(
            String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
                .contains("verification source rejected")
        );
        f.no_transport_temps();
    }
}

#[path = "verification_caller_wrapper.rs"]
mod wrapper_qualification;
