use crate::{
    github::{self, Gh},
    process::Cancellation,
    provenance,
    real_git::Repository,
    transport::Transport,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    os::unix::fs::PermissionsExt,
    thread,
    time::{Duration, Instant},
};
struct Fixture {
    root: tempfile::TempDir,
    environment: BTreeMap<OsString, OsString>,
    cancellation: Cancellation,
}
impl Fixture {
    fn new(body: &str) -> Self {
        let root = tempfile::Builder::new()
            .prefix("release gh evidence ")
            .tempdir()
            .unwrap();
        fs::create_dir(root.path().join("bin")).unwrap();
        let executable = root.path().join("bin/gh");
        fs::write(&executable,format!("#!/bin/sh\nprintf '%s\\0' \"$@\" > \"$FIXTURE_ARGUMENTS\"\nprintf '%s' \"$OPERATOR_INPUT\" > \"$FIXTURE_ENV\"\nif read -r ignored; then exit 29; fi\n{body}\n")).unwrap();
        fs::set_permissions(executable, fs::Permissions::from_mode(0o700)).unwrap();
        let mut environment: BTreeMap<_, _> = std::env::vars_os().collect();
        environment.insert(
            "PATH".into(),
            std::env::join_paths(
                std::iter::once(root.path().join("bin"))
                    .chain(std::env::split_paths(&std::env::var_os("PATH").unwrap())),
            )
            .unwrap(),
        );
        for (key, value) in [
            ("FIXTURE_ARGUMENTS", root.path().join("arguments")),
            ("FIXTURE_RESPONSE", root.path().join("response.json")),
            ("FIXTURE_ENV", root.path().join("environment")),
            ("FIXTURE_CHILD", root.path().join("child")),
        ] {
            environment.insert(key.into(), value.into());
        }
        environment.insert("OPERATOR_INPUT".into(), "operator value with spaces".into());
        environment.insert("GH_TOKEN".into(), "fixture-private-auth".into());
        Self {
            root,
            environment,
            cancellation: Cancellation::default(),
        }
    }
    fn response(&self, value: &Value) {
        fs::write(
            self.root.path().join("response.json"),
            serde_json::to_vec(value).unwrap(),
        )
        .unwrap();
    }
    fn owned(&self, budget: Duration) -> Transport {
        Transport::new(
            self.root.path(),
            self.environment.clone(),
            self.cancellation.clone(),
            budget,
        )
        .unwrap()
    }
    fn arguments(&self) -> Vec<String> {
        fs::read(self.root.path().join("arguments"))
            .unwrap()
            .split(|b| *b == 0)
            .filter(|v| !v.is_empty())
            .map(|v| String::from_utf8(v.to_vec()).unwrap())
            .collect()
    }
    fn child_gone(&self) {
        let pid: i32 = fs::read_to_string(self.root.path().join("child"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // SAFETY: signal zero only probes the PID recorded by this finite owned fixture.
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }
}
fn release_json() -> Value {
    json!({"tagName":"v1.0.0","name":"Original title","publishedAt":"2026-10-01T12:34:56Z","url":"https://example.invalid/release","isPrerelease":true,"isDraft":false,"body":"raw release body\nsecond line","targetCommitish":"main","assets":[{"name":"host archive","size":8192,"url":"https://example.invalid/asset"}]})
}
#[test]
fn release_inventory_evidence_inert_gh_release_default_override_metadata_argv_env() {
    let fixture = Fixture::new("cat \"$FIXTURE_RESPONSE\"");
    fixture.response(&release_json());
    let mut owned = fixture.owned(Duration::from_secs(10));
    let release = github::release(&mut owned, "Mesh-LLM/mesh-llm", None).unwrap();
    assert_eq!(release.tag, "v1.0.0");
    assert_eq!(release.raw, release_json().as_object().unwrap().clone());
    assert_eq!(
        fixture.arguments(),
        vec![
            "release",
            "view",
            "--repo",
            "Mesh-LLM/mesh-llm",
            "--json",
            "tagName,name,publishedAt,url,isPrerelease,isDraft,body,targetCommitish,assets"
        ]
    );
    github::release(&mut owned, "Mesh-LLM/mesh-llm", Some("v1.0.0")).unwrap();
    assert_eq!(&fixture.arguments()[..3], ["release", "view", "v1.0.0"]);
    assert_eq!(
        fs::read_to_string(fixture.root.path().join("environment")).unwrap(),
        "operator value with spaces"
    );
    assert!(
        !fixture
            .arguments()
            .join(" ")
            .contains("fixture-private-auth")
    );
}
#[test]
fn release_inventory_evidence_inert_gh_pr_scope_limit_and_raw_fields() {
    let fixture = Fixture::new("cat \"$FIXTURE_RESPONSE\"");
    fixture.response(&release_json());
    let mut owned = fixture.owned(Duration::from_secs(20));
    let release = github::release(&mut owned, "Mesh-LLM/mesh-llm", None).unwrap();
    let mut repo = Repository::new();
    let source = provenance::freeze(&mut repo, "v1.0.0", "HEAD").unwrap();
    let raw = json!({"number":42,"title":"raw title","url":"https://example.invalid/42","body":"full body\nnext","mergedAt":"2026-10-01T12:34:56Z","mergeCommit":null,"labels":[{"name":"fix"}],"author":{"login":"original"},"baseRefName":"main","headRefName":"feature"});
    fixture.response(&Value::Array(vec![raw.clone(); 1000]));
    let prs = github::pull_requests(&mut owned, "Mesh-LLM/mesh-llm", &release, &source).unwrap();
    assert_eq!(prs.rows.len(), 1000);
    assert!(prs.may_be_truncated);
    assert_eq!(prs.query_limit, 1000);
    assert!(prs.query_scope.contains("not a complete enumeration"));
    for (key, value) in raw.as_object().unwrap() {
        assert_eq!(&prs.rows[0][key], value);
    }
    assert!(prs.rows[0]["merge_commit_in_range"].is_null());
    assert_eq!(
        fixture.arguments(),
        vec![
            "pr",
            "list",
            "--repo",
            "Mesh-LLM/mesh-llm",
            "--state",
            "merged",
            "--search",
            "merged:>=2026-10-01T12:34:56Z",
            "--limit",
            "1000",
            "--json",
            "number,title,url,body,mergedAt,mergeCommit,labels,author,baseRefName,headRefName"
        ]
    );
    for invalid in [json!({"not":"a PR array"}), json!([7])] {
        fixture.response(&invalid);
        assert!(github::pull_requests(&mut owned, "Mesh-LLM/mesh-llm", &release, &source).is_err());
    }
    fixture.response(&Value::Array(vec![raw; 1001]));
    assert!(github::pull_requests(&mut owned, "Mesh-LLM/mesh-llm", &release, &source).is_err());
    let before = fixture.arguments();
    let mut mismatched = release;
    mismatched.tag = "different-release".into();
    assert!(github::pull_requests(&mut owned, "Mesh-LLM/mesh-llm", &mismatched, &source).is_err());
    assert_eq!(fixture.arguments(), before);
}
#[test]
fn release_inventory_evidence_inert_gh_invalid_json_shape_status_and_bounds() {
    let fixture = Fixture::new("cat \"$FIXTURE_RESPONSE\"");
    let mut owned = fixture.owned(Duration::from_secs(10));
    fs::write(fixture.root.path().join("response.json"), b"{invalid JSON").unwrap();
    assert!(github::release(&mut owned, "Mesh-LLM/mesh-llm", None).is_err());
    for raw in [json!([]), json!({}), {
        let mut v = release_json();
        v["publishedAt"] = json!("2026-10-01 OR arbitrary query");
        v
    }] {
        fixture.response(&raw);
        assert!(github::release(&mut owned, "Mesh-LLM/mesh-llm", None).is_err());
    }
    let failed = Fixture::new("printf '%s' \"$GH_TOKEN\" >&2; exit 7");
    let err = github::release(
        &mut failed.owned(Duration::from_secs(10)),
        "Mesh-LLM/mesh-llm",
        None,
    )
    .err()
    .unwrap();
    assert!(!err.0.contains("fixture-private-auth"));
    let oversized = Fixture::new("while :; do printf 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'; done");
    let mut bounded = oversized.owned(Duration::from_secs(5));
    bounded.capture_limit(256);
    assert!(github::release(&mut bounded, "Mesh-LLM/mesh-llm", None).is_err());
}
const OWNED_WAIT: &str = "sleep 30 &\nchild=$!\ntrap 'kill \"$child\" 2>/dev/null || :; wait \"$child\" 2>/dev/null || :; exit 143' TERM INT\nprintf '%s' \"$child\" > \"$FIXTURE_CHILD\"\nwait \"$child\"";
#[test]
fn release_inventory_evidence_inert_gh_deadline_reaps_owned_descendant() {
    let fixture = Fixture::new(OWNED_WAIT);
    let start = Instant::now();
    let mut owned = fixture.owned(Duration::from_secs(1));
    assert!(
        owned
            .request(&["release", "view"])
            .err()
            .unwrap()
            .0
            .contains("deadline")
    );
    assert!(start.elapsed() < Duration::from_secs(10));
    fixture.child_gone();
}
#[test]
fn release_inventory_evidence_inert_gh_cancellation_reaps_owned_descendant() {
    let fixture = Fixture::new(OWNED_WAIT);
    let marker = fixture.root.path().join("child");
    let cancellation = fixture.cancellation.clone();
    let canceller = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !marker.is_file() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        cancellation.cancel();
    });
    let error = fixture
        .owned(Duration::from_secs(20))
        .request(&["release", "view"])
        .err()
        .unwrap();
    canceller.join().unwrap();
    assert_eq!(error.0, "release inventory cancelled");
    fixture.child_gone();
}
#[test]
fn release_inventory_evidence_inert_gh_admission_no_launch_on_cancel_or_deadline() {
    let fixture = Fixture::new("exit 0");
    fixture.cancellation.cancel();
    assert!(
        fixture
            .owned(Duration::from_secs(10))
            .request(&["release", "view"])
            .is_err()
    );
    assert!(!fixture.root.path().join("arguments").exists());
    let fresh = Fixture::new("exit 0");
    assert!(
        fresh
            .owned(Duration::ZERO)
            .request(&["release", "view"])
            .is_err()
    );
    assert!(!fresh.root.path().join("arguments").exists());
    assert!(
        github::release(
            &mut fresh.owned(Duration::from_secs(10)),
            "Mesh-LLM/mesh-llm",
            Some("--danger")
        )
        .is_err()
    );
    assert!(!fresh.root.path().join("arguments").exists());
}
