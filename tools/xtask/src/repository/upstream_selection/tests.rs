//! Hermetic walk tests over a fake GitHub transport: no process, no network.

use super::*;
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::fs;

/// A deterministic lowercase 40-character SHA for a seed: two FNV-1a folds so
/// distinct seeds never collide, and every seed is a legal lowercase SHA.
fn sha_for(seed: &str) -> String {
    let mut forward: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in seed.bytes() {
        forward ^= u64::from(byte);
        forward = forward.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut backward: u64 = 0x8422_2325_cbf2_9ce4;
    for byte in seed.bytes().rev() {
        backward ^= u64::from(byte);
        backward = backward.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{forward:016x}{backward:016x}{:08x}", (forward ^ backward) as u32)
}

fn base_request(pin: &str) -> Request {
    Request {
        upstream_repository: DEFAULT_UPSTREAM_REPOSITORY.to_owned(),
        branch: DEFAULT_BRANCH.to_owned(),
        pin: pin.to_owned(),
        requested: None,
        max_history: DEFAULT_MAX_HISTORY,
        max_candidates: DEFAULT_MAX_CANDIDATES,
        retry_delay: Duration::ZERO,
    }
}

fn run(name: &str, status: &str, conclusion: Option<&str>) -> Value {
    run_at(name, status, conclusion, "2026-10-15T00:00:00Z")
}

fn run_at(name: &str, status: &str, conclusion: Option<&str>, started_at: &str) -> Value {
    json!({
        "id": 1,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started_at": started_at,
        "app": {"id": 1, "slug": "github-actions"},
    })
}

/// Serves a branch history and per-commit check runs, with optional request
/// failures and a per-call log.
struct Fake {
    commits: Vec<Value>,
    checks: BTreeMap<String, Vec<Value>>,
    calls: RefCell<Vec<String>>,
    failures_remaining: Cell<u32>,
}

impl Fake {
    /// history: (seed, subject) newest first; every commit gets one passing run
    /// unless a later builder call replaces it.
    fn new(history: &[(&str, &str)]) -> Self {
        let commits = history
            .iter()
            .enumerate()
            .map(|(index, (seed, subject))| {
                json!({
                    "sha": sha_for(seed),
                    "commit": {
                        "message": format!("{subject}\n\nbody"),
                        "committer": {"date": format!("2026-10-15T0{}:00:00Z", index % 10)},
                    },
                })
            })
            .collect::<Vec<_>>();
        Self {
            commits,
            checks: BTreeMap::new(),
            calls: RefCell::new(Vec::new()),
            failures_remaining: Cell::new(0),
        }
    }

    fn with_checks(mut self, seed: &str, runs: Vec<Value>) -> Self {
        self.checks.insert(sha_for(seed), runs);
        self
    }

    fn fail_times(self, times: u32) -> Self {
        self.failures_remaining.set(times);
        self
    }

    fn calls(&self) -> Vec<String> {
        self.calls.borrow().clone()
    }

    fn checks_paths(&self) -> Vec<String> {
        self.calls()
            .into_iter()
            .filter(|call| call.contains("/check-runs"))
            .collect()
    }
}

impl Gh for Fake {
    fn api(&self, path: &str, query: &[(&str, String)]) -> Result<String, String> {
        let described = format!(
            "{path}?{}",
            query
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join("&")
        );
        self.calls.borrow_mut().push(described.clone());
        if self.failures_remaining.get() > 0 {
            self.failures_remaining.set(self.failures_remaining.get() - 1);
            return Err(format!("gh api {path} failed: 502"));
        }
        let page = query
            .iter()
            .find(|(key, _)| *key == "page")
            .and_then(|(_, value)| value.parse::<usize>().ok())
            .unwrap_or(1);
        let rows = if path.ends_with("/check-runs") {
            let sha = path
                .split('/')
                .rev()
                .nth(1)
                .ok_or("check-runs path without a SHA")?;
            self.checks.get(sha).cloned().unwrap_or_default()
        } else {
            self.commits.clone()
        };
        let start = (page - 1) * PAGE_SIZE;
        let slice = if start >= rows.len() {
            Vec::new()
        } else {
            rows[start..rows.len().min(start + PAGE_SIZE)].to_vec()
        };
        if path.ends_with("/check-runs") {
            return Ok(json!({
                "total_count": rows.len(),
                "check_runs": slice,
            })
            .to_string());
        }
        Ok(Value::Array(slice).to_string())
    }
}

fn select_with(fake: &Fake, request: &Request) -> Result<Selection, String> {
    select(fake, request)
}

#[test]
fn newest_settled_passing_commit_is_selected() {
    let fake = Fake::new(&[("tip", "tip"), ("second", "second"), ("third", "third")])
        .with_checks(
            "tip",
            vec![run("build", "in_progress", None), run("lint", "completed", Some("success"))],
        )
        .with_checks("second", vec![run("build", "completed", Some("success"))]);
    let selection = select_with(&fake, &base_request(&sha_for("pin"))).unwrap();
    assert_eq!(selection.sha, sha_for("second"));
    assert_eq!(selection.outcome, OUTCOME_WALK);
    assert_eq!(selection.examined, 2);
    let paths = fake.checks_paths();
    assert!(paths.iter().any(|path| path.contains(&sha_for("tip"))));
    assert!(paths.iter().any(|path| path.contains(&sha_for("second"))));
    assert!(!paths.iter().any(|path| path.contains(&sha_for("third"))));
}

#[test]
fn failing_running_and_unchecked_commits_are_skipped() {
    let fake = Fake::new(&[
        ("fail", "fail"),
        ("running", "running"),
        ("unchecked", "unchecked"),
        ("green", "green"),
    ])
    .with_checks("fail", vec![run("build", "completed", Some("failure"))])
    .with_checks("running", vec![run("build", "queued", None)])
    .with_checks("unchecked", Vec::new())
    .with_checks("green", vec![run("build", "completed", Some("success"))]);
    let selection = select_with(&fake, &base_request(&sha_for("pin"))).unwrap();
    assert_eq!(selection.sha, sha_for("green"));
    assert_eq!(selection.examined, 4);
    assert_eq!(selection.candidates[2].checks, 0);
    assert_eq!(selection.candidates[1].pending, 1);
    assert_eq!(selection.candidates[0].failing.len(), 1);
}

#[test]
fn rerun_success_replaces_an_earlier_cancellation() {
    let fake = Fake::new(&[("tip", "tip")]).with_checks(
        "tip",
        vec![
            run_at("cuda", "completed", Some("cancelled"), "2026-10-15T00:00:00Z"),
            run_at("cuda", "completed", Some("success"), "2026-10-15T01:00:00Z"),
        ],
    );
    let selection = select_with(&fake, &base_request(&sha_for("pin"))).unwrap();
    assert_eq!(selection.sha, sha_for("tip"));
    assert_eq!(selection.candidates[0].checks, 1);
}

#[test]
fn later_running_rerun_keeps_the_commit_unsettled() {
    let fake = Fake::new(&[("tip", "tip")]).with_checks(
        "tip",
        vec![
            run_at("cuda", "completed", Some("success"), "2026-10-15T00:00:00Z"),
            run_at("cuda", "queued", None, "2026-10-15T01:00:00Z"),
        ],
    );
    let selection = select_with(&fake, &base_request(&sha_for("pin"))).unwrap();
    assert_eq!(selection.outcome, OUTCOME_PIN);
    assert_eq!(selection.examined, 1);
    assert_eq!(selection.candidates[0].pending, 1);
}

#[test]
fn every_non_passing_conclusion_rejects_a_commit() {
    let request = base_request(&sha_for("pin"));
    let commit = CommitRow {
        sha: sha_for("tip"),
        date: "2026-10-15T00:00:00Z".into(),
        subject: "tip".into(),
    };
    for conclusion in [
        "failure",
        "timed_out",
        "cancelled",
        "action_required",
        "startup_failure",
        "stale",
        "unexpected",
    ] {
        let fake = Fake::new(&[("tip", "tip")])
            .with_checks("tip", vec![run("build", "completed", Some(conclusion))]);
        let candidate = assess(&fake, &request, &commit).unwrap();
        assert!(!candidate.passing(), "{conclusion}");
        assert_eq!(candidate.failing, [format!("build ({conclusion})")]);
    }
    let fake = Fake::new(&[("tip", "tip")])
        .with_checks("tip", vec![run("build", "completed", None)]);
    let candidate = assess(&fake, &request, &commit).unwrap();
    assert!(!candidate.passing());
    assert_eq!(candidate.failing, ["build (no conclusion)"]);
    for status in ["queued", "in_progress", "waiting", "requested", "pending"] {
        let fake = Fake::new(&[("tip", "tip")])
            .with_checks("tip", vec![run("build", status, None)]);
        let candidate = assess(&fake, &request, &commit).unwrap();
        assert!(!candidate.passing(), "{status}");
        assert_eq!(candidate.pending, 1);
    }
    for conclusion in ["success", "skipped", "neutral"] {
        let fake = Fake::new(&[("tip", "tip")])
            .with_checks("tip", vec![run("build", "completed", Some(conclusion))]);
        let candidate = assess(&fake, &request, &commit).unwrap();
        assert!(candidate.passing(), "{conclusion}");
    }
}

#[test]
fn a_commit_without_checks_never_passes() {
    let request = base_request(&sha_for("pin"));
    let commit = CommitRow {
        sha: sha_for("tip"),
        date: String::new(),
        subject: "tip".into(),
    };
    let fake = Fake::new(&[("tip", "tip")]).with_checks("tip", Vec::new());
    assert!(!assess(&fake, &request, &commit).unwrap().passing());
}

#[test]
fn the_walk_stops_at_the_current_pin() {
    let fake = Fake::new(&[("tip", "tip"), ("pin", "pin"), ("older", "older")])
        .with_checks("tip", vec![run("build", "completed", Some("failure"))])
        .with_checks("pin", vec![run("build", "completed", Some("failure"))])
        .with_checks("older", vec![run("build", "completed", Some("success"))]);
    let selection = select_with(&fake, &base_request(&sha_for("pin"))).unwrap();
    assert_eq!(selection.sha, sha_for("pin"));
    assert_eq!(selection.outcome, OUTCOME_PIN);
    assert_eq!(selection.examined, 1);
    assert!(!fake
        .checks_paths()
        .iter()
        .any(|path| path.contains(&sha_for("older"))));
}

#[test]
fn a_pin_at_the_branch_tip_examines_nothing() {
    let fake = Fake::new(&[("tip", "tip"), ("older", "older")]);
    let selection = select_with(&fake, &base_request(&sha_for("tip"))).unwrap();
    assert_eq!(selection.sha, sha_for("tip"));
    assert_eq!(selection.outcome, OUTCOME_PIN);
    assert_eq!(selection.examined, 0);
    assert!(fake.checks_paths().is_empty());
}

#[test]
fn the_candidate_window_is_bounded_and_keeps_the_pin() {
    let fake = Fake::new(&[("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")])
        .with_checks("a", vec![run("build", "completed", Some("failure"))])
        .with_checks("b", vec![run("build", "completed", Some("failure"))])
        .with_checks("c", vec![run("build", "completed", Some("success"))]);
    let mut request = base_request(&sha_for("pin"));
    request.max_candidates = 2;
    let selection = select_with(&fake, &request).unwrap();
    assert_eq!(selection.outcome, OUTCOME_PIN);
    assert_eq!(selection.examined, 2);
    assert!(!fake
        .checks_paths()
        .iter()
        .any(|path| path.contains(&sha_for("c"))));
}

#[test]
fn a_requested_commit_that_passes_is_selected() {
    let fake = Fake::new(&[("tip", "tip"), ("requested", "requested"), ("older", "older")])
        .with_checks("tip", vec![run("build", "completed", Some("failure"))])
        .with_checks("requested", vec![run("build", "completed", Some("success"))]);
    let mut request = base_request(&sha_for("pin"));
    request.requested = Some(sha_for("requested"));
    let selection = select_with(&fake, &request).unwrap();
    assert_eq!(selection.sha, sha_for("requested"));
    assert_eq!(selection.outcome, OUTCOME_REQUESTED);
    assert_eq!(selection.examined, 1);
    assert!(!fake
        .checks_paths()
        .iter()
        .any(|path| path.contains(&sha_for("tip"))));
}

#[test]
fn a_requested_commit_that_fails_walks_down_and_is_reported() {
    let fake = Fake::new(&[("requested", "requested"), ("older", "older")])
        .with_checks("requested", vec![run("build", "completed", Some("failure"))])
        .with_checks("older", vec![run("build", "completed", Some("success"))]);
    let mut request = base_request(&sha_for("pin"));
    request.requested = Some(sha_for("requested"));
    let selection = select_with(&fake, &request).unwrap();
    assert_eq!(selection.sha, sha_for("older"));
    assert_eq!(selection.outcome, OUTCOME_WALK);
    let summary = summary(&selection, &request);
    assert!(summary.contains(&sha_for("requested")));
    assert!(summary.contains("did not pass its own CI"));
    assert!(summary.contains("build (failure)"));
}

#[test]
fn a_requested_commit_off_the_branch_fails_closed() {
    let fake = Fake::new(&[("tip", "tip")]);
    let mut request = base_request(&sha_for("pin"));
    request.requested = Some(sha_for("elsewhere"));
    let error = select_with(&fake, &request).unwrap_err();
    assert!(error.contains("is not within the newest"), "{error}");
}

#[test]
fn a_requested_commit_at_or_below_the_pin_fails_closed() {
    let fake = Fake::new(&[("tip", "tip"), ("pin", "pin"), ("older", "older")]);
    let mut request = base_request(&sha_for("pin"));
    request.requested = Some(sha_for("older"));
    let error = select_with(&fake, &request).unwrap_err();
    assert!(error.contains("at or below the current pin"), "{error}");
    let mut request = base_request(&sha_for("pin"));
    request.requested = Some(sha_for("pin"));
    assert!(select_with(&fake, &request).is_err());
}

#[test]
fn api_failures_are_retried_and_then_reported() {
    let fake = Fake::new(&[("tip", "tip")]).fail_times(3);
    let error = select_with(&fake, &base_request(&sha_for("pin"))).unwrap_err();
    assert!(error.contains("after 3 attempts"), "{error}");
    assert_eq!(fake.calls().len(), 3);
}

#[test]
fn a_transient_failure_recovers_within_the_attempt_budget() {
    let fake = Fake::new(&[("tip", "tip")])
        .with_checks("tip", vec![run("build", "completed", Some("success"))])
        .fail_times(2);
    let selection = select_with(&fake, &base_request(&sha_for("pin"))).unwrap();
    assert_eq!(selection.sha, sha_for("tip"));
    // One branch listing plus three attempts for the commit's check runs.
    assert_eq!(fake.calls().len(), 4);
}

#[test]
fn every_commit_page_and_check_run_page_is_read() {
    let history: Vec<(String, String)> = (0..150)
        .map(|index| (format!("c{index}"), format!("commit {index}")))
        .collect();
    let borrowed = history
        .iter()
        .map(|(seed, subject)| (seed.as_str(), subject.as_str()))
        .collect::<Vec<_>>();
    let mut fake = Fake::new(&borrowed);
    fake = fake.with_checks(
        "c149",
        (0..150)
            .map(|index| {
                run_at(
                    &format!("lane-{index}"),
                    "completed",
                    Some("success"),
                    "2026-10-15T00:00:00Z",
                )
            })
            .collect(),
    );
    let mut request = base_request(&sha_for("pin"));
    request.max_history = 1000;
    // The only passing commit is the oldest one, so the walk must reach it.
    request.max_candidates = 200;
    let selection = select_with(&fake, &request).unwrap();
    assert_eq!(selection.sha, sha_for("c149"));
    assert_eq!(selection.examined, 150);
    assert_eq!(selection.candidates.last().unwrap().checks, 150);
    assert_eq!(selection.candidates[0].checks, 0);
    let calls = fake.calls();
    assert!(calls.iter().any(|call| call.contains("commits?sha=master") && call.contains("page=2")));
    assert!(calls
        .iter()
        .any(|call| call.contains("/check-runs") && call.contains("page=2")));
}

#[test]
fn history_is_truncated_to_the_requested_bound() {
    let history: Vec<(String, String)> = (0..120)
        .map(|index| (format!("c{index}"), format!("commit {index}")))
        .collect();
    let borrowed = history
        .iter()
        .map(|(seed, subject)| (seed.as_str(), subject.as_str()))
        .collect::<Vec<_>>();
    let fake = Fake::new(&borrowed);
    let mut request = base_request(&sha_for("pin"));
    request.max_history = 10;
    let selection = select_with(&fake, &request).unwrap();
    assert_eq!(selection.outcome, OUTCOME_PIN);
    assert_eq!(selection.examined, 10);
    let mut request = base_request(&sha_for("pin"));
    request.max_history = 10;
    request.requested = Some(sha_for("c50"));
    let error = select_with(&fake, &request).unwrap_err();
    assert!(error.contains("not within the newest 10 commits"), "{error}");
}

#[test]
fn the_summary_reports_the_policy_and_the_walk() {
    let fake = Fake::new(&[("tip", "kv-cache tip"), ("green", "server green")])
        .with_checks("tip", vec![run("build", "completed", Some("failure"))])
        .with_checks("green", vec![run("build", "completed", Some("success"))]);
    let selection = select_with(&fake, &base_request(&sha_for("pin"))).unwrap();
    let summary = summary(&selection, &base_request(&sha_for("pin")));
    assert!(summary.starts_with("### llama.cpp upstream selection"));
    assert!(summary.contains("ggml-org/llama.cpp/master"));
    assert!(summary.contains(&sha_for("green")));
    assert!(summary.contains("kv-cache tip"));
    assert!(summary.contains("build (failure)"));
    assert!(summary.contains("1 with failing checks"));
    assert!(summary.contains("concluded success, skipped or neutral"));
}

#[test]
fn the_summary_reports_a_pin_that_was_kept() {
    let fake = Fake::new(&[("tip", "tip")])
        .with_checks("tip", vec![run("build", "completed", Some("failure"))]);
    let request = base_request(&sha_for("pin"));
    let selection = select_with(&fake, &request).unwrap();
    let summary = summary(&selection, &request);
    assert!(summary.contains("unchanged: no newer commit passes its own CI"));
    assert!(summary.contains(&sha_for("pin")));
}

#[test]
fn the_summary_reports_an_empty_walk() {
    let fake = Fake::new(&[("pin", "pin")]);
    let request = base_request(&sha_for("pin"));
    let selection = select_with(&fake, &request).unwrap();
    let summary = summary(&selection, &request);
    assert!(summary.contains("No candidate was examined"));
    assert!(!summary.contains("| commit |"));
}

#[test]
fn github_output_and_summary_are_written_together() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("output");
    let summary_path = directory.path().join("summary.md");
    let fake = Fake::new(&[("tip", "tip")])
        .with_checks("tip", vec![run("build", "completed", Some("success"))]);
    let request = base_request(&sha_for("pin"));
    let selection = select_with(&fake, &request).unwrap();
    publish(
        &selection,
        &request,
        Some(output.to_str().unwrap()),
        Some(summary_path.to_str().unwrap()),
    )
    .unwrap();
    let written = fs::read_to_string(&output).unwrap();
    assert_eq!(written, format!("upstream_selection=walk\nupstream_examined={}\n", selection.examined));
    assert!(fs::read_to_string(&summary_path)
        .unwrap()
        .contains("### llama.cpp upstream selection"));
    // A second run appends instead of replacing the step summary.
    publish(
        &selection,
        &request,
        Some(output.to_str().unwrap()),
        Some(summary_path.to_str().unwrap()),
    )
    .unwrap();
    assert_eq!(fs::read_to_string(&output).unwrap().matches("upstream_selection=").count(), 2);
    assert!(publish(&selection, &request, Some(output.to_str().unwrap()), None).is_err());
    assert!(publish(&selection, &request, None, None).is_ok());
}

#[test]
fn requests_reject_every_malformed_input() {
    let pin = sha_for("pin");
    let parse = |args: &[&str]| -> Result<Request, String> {
        let owned = args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>();
        let parsed = GRAMMAR
            .parse(&owned)
            .map_err(|report| report.stderr.clone())?;
        super::request(&parsed)
    };
    assert!(parse(&["--pin", &pin]).is_ok());
    assert!(parse(&["--pin", &pin, "--requested", "latest"]).unwrap().requested.is_none());
    assert!(parse(&["--pin", &pin, "--requested", ""]).unwrap().requested.is_none());
    assert_eq!(
        parse(&["--pin", &pin, "--requested", &sha_for("x")])
            .unwrap()
            .requested
            .as_deref(),
        Some(sha_for("x").as_str())
    );
    for args in [
        vec!["--requested", &pin],
        vec!["--pin", "short"],
        vec!["--pin", "8B8B8B8B8B8B8B8B8B8B8B8B8B8B8B8B8B8B8B8B"],
        vec!["--pin", &pin, "--requested", "nope"],
        vec!["--pin", &pin, "--upstream-repository", "llama.cpp"],
        vec!["--pin", &pin, "--upstream-repository", "ggml-org/../x"],
        vec!["--pin", &pin, "--branch", ""],
        vec!["--pin", &pin, "--branch", "-x"],
        vec!["--pin", &pin, "--max-history", "0"],
        vec!["--pin", &pin, "--max-candidates", "201"],
        vec!["--pin", &pin, "--retry-delay-ms", "-1"],
    ] {
        assert!(parse(&args).is_err(), "{args:?}");
    }
    assert!(seconds(
        &GRAMMAR
            .parse(&["--pin".to_owned(), pin.clone(), "--timeout-secs".to_owned(), "1".to_owned()])
            .unwrap()
    )
    .is_err());
    assert_eq!(
        seconds(&GRAMMAR.parse(&["--pin".to_owned(), pin]).unwrap()).unwrap(),
        DEFAULT_TIMEOUT_SECONDS
    );
}
