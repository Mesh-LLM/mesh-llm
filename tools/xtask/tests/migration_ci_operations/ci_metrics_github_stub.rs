//! `ci-ops collect-metrics --workflow/--run-id` against the GitHub collection
//! path of `scripts/collect-ci-metrics.py` (`gh_json`, `fetch_jobs`,
//! `fetch_exact_run`, `fetch_runs`). No real `gh` runs: `PATH` is the case's
//! stub directory followed by `/usr/bin:/bin`. The stub `gh` appends its argv
//! to `gh-argv.log` and replays the canned reply the case wrote to
//! `gh/<key>.{out,err,code}`, where `<key>` is `run_list`, `run_view_<id>` or
//! `jobs_<id>_<page>`. Goldens `fixtures/ci_operations/ci_metrics/github_*.json`
//! record the usual streams and files plus the recorded `gh_argv`, captured
//! from the legacy script running against the same stub.

use crate::ci_metrics::{MARKDOWN_OUTPUT, OUTPUT, RAW_OUTPUT, mask};
use crate::support::{CAPTURE_ENV, LEGACY_ENV, Stage, TestResult, fixture_dir, repo_root};
use serde_json::{Value, json};
use std::error::Error;
use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const SCRIPT: &str = "scripts/collect-ci-metrics.py";
const LOG: &str = "gh-argv.log";

/// A canned `gh` reply: stdout, stderr and exit status.
#[derive(Clone, Default)]
pub(crate) struct Reply {
    out: Option<String>,
    err: Option<String>,
    code: Option<i32>,
}

pub(crate) fn json_reply(value: &Value) -> Reply {
    Reply {
        out: Some(serde_json::to_string_pretty(value).unwrap_or_default() + "\n"),
        ..Reply::default()
    }
}

pub(crate) fn failing(code: i32, out: &str, err: &str) -> Reply {
    Reply {
        out: Some(out.to_owned()).filter(|text| !text.is_empty()),
        err: Some(err.to_owned()).filter(|text| !text.is_empty()),
        code: Some(code),
    }
}

pub(crate) fn text_reply(out: &str) -> Reply {
    Reply {
        out: Some(out.to_owned()),
        ..Reply::default()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Observed {
    pub(crate) code: i32,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) output: Option<String>,
    pub(crate) raw_output: Option<String>,
    pub(crate) markdown_output: Option<String>,
    pub(crate) gh_argv: Vec<Vec<String>>,
}

fn stub(stage: &Path) -> String {
    let dir = stage.to_string_lossy();
    format!(
        "#!/bin/sh\ndir='{dir}'\n\
{{ for arg in \"$@\"; do printf '%s\\n' \"$arg\"; done; printf '%s\\n' '<end>'; }} >> \"$dir/{LOG}\"\n\
case \"$1 $2\" in\n\
  'run list') key=run_list ;;\n\
  'run view') key=\"run_view_$3\" ;;\n\
  'api --method') run=${{4%/jobs}}; key=\"jobs_${{run##*/}}_${{10#page=}}\" ;;\n\
  *) key=unknown ;;\n\
esac\n\
reply=\"$dir/gh/$key\"\n\
if [ -f \"$reply.err\" ]; then cat \"$reply.err\" >&2; fi\n\
if [ -f \"$reply.out\" ]; then cat \"$reply.out\"; fi\n\
if [ -f \"$reply.code\" ]; then exit \"$(cat \"$reply.code\")\"; fi\n\
if [ ! -f \"$reply.out\" ]; then echo \"stub gh: no reply for $key\" >&2; exit 99; fi\n"
    )
}

fn prepare(
    name: &str,
    replies: &[(String, Reply)],
    with_gh: bool,
) -> Result<Stage, Box<dyn Error>> {
    let stage = Stage::empty(&format!("ci-metrics-{name}"))?;
    fs::create_dir_all(stage.path().join("bin"))?;
    if with_gh {
        stage.write("bin/gh", stub(stage.path()).as_bytes())?;
        fs::set_permissions(
            stage.path().join("bin/gh"),
            fs::Permissions::from_mode(0o755),
        )?;
    }
    for (key, reply) in replies {
        let parts = [("out", reply.out.clone()), ("err", reply.err.clone())];
        for (suffix, text) in parts {
            if let Some(text) = text {
                stage.write(&format!("gh/{key}.{suffix}"), text.as_bytes())?;
            }
        }
        if let Some(code) = reply.code {
            stage.write(&format!("gh/{key}.code"), code.to_string().as_bytes())?;
        }
    }
    Ok(stage)
}

fn reset(stage: &Stage) {
    for name in ["out", "raw", "md", LOG] {
        let path = stage.path().join(name);
        let _missing = fs::remove_dir_all(&path).or_else(|_| fs::remove_file(&path));
    }
}

fn read(stage: &Stage, relative: &str) -> Option<String> {
    let root = stage.root_arg();
    fs::read_to_string(stage.path().join(relative))
        .ok()
        .map(|text| mask(&text.replace(&root, "{root}")))
}

fn recorded_argv(stage: &Stage) -> Vec<Vec<String>> {
    let log = fs::read_to_string(stage.path().join(LOG)).unwrap_or_default();
    let mut calls = Vec::new();
    let mut current = Vec::new();
    for line in log.lines() {
        if line == "<end>" {
            calls.push(std::mem::take(&mut current));
        } else {
            current.push(line.to_owned());
        }
    }
    calls
}

fn run(
    stage: &Stage,
    program: &Path,
    prefix: &[PathBuf],
    args: &[&str],
) -> Result<Observed, Box<dyn Error>> {
    reset(stage);
    let root = stage.root_arg();
    let output = Command::new(program)
        .current_dir(stage.path())
        .args(prefix)
        .args(args)
        .env("PATH", format!("{root}/bin:/usr/bin:/bin"))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env_remove("COLUMNS")
        .stdin(Stdio::null())
        .output()?;
    let clean = |bytes: &[u8]| mask(&String::from_utf8_lossy(bytes).replace(&root, "{root}"));
    Ok(Observed {
        code: output.status.code().unwrap_or(-1),
        stdout: clean(&output.stdout),
        stderr: clean(&output.stderr),
        output: read(stage, OUTPUT),
        raw_output: read(stage, RAW_OUTPUT),
        markdown_output: read(stage, MARKDOWN_OUTPUT),
        gh_argv: recorded_argv(stage),
    })
}

fn golden_path(name: &str) -> PathBuf {
    fixture_dir()
        .join("ci_metrics")
        .join(format!("{name}.json"))
}

fn capture(name: &str, args: &[&str], legacy: &Observed) -> TestResult {
    let mut golden = json!({
        "args": args,
        "code": legacy.code,
        "stdout": legacy.stdout,
        "stderr": legacy.stderr,
        "output": legacy.output,
        "raw_output": legacy.raw_output,
        "gh_argv": legacy.gh_argv,
    });
    if let Some(summary) = &legacy.markdown_output {
        golden["markdown_output"] = json!(summary);
    }
    fs::write(
        golden_path(name),
        serde_json::to_string_pretty(&golden)? + "\n",
    )?;
    Ok(())
}

fn expected(name: &str) -> Result<Observed, Box<dyn Error>> {
    let golden: Value = serde_json::from_slice(&fs::read(golden_path(name))?)?;
    let text = |key: &str| golden[key].as_str().map(str::to_owned);
    Ok(Observed {
        code: i32::try_from(golden["code"].as_i64().ok_or("golden code")?)?,
        stdout: text("stdout").ok_or("golden stdout")?,
        stderr: text("stderr").ok_or("golden stderr")?,
        output: text("output"),
        raw_output: text("raw_output"),
        markdown_output: text("markdown_output"),
        gh_argv: serde_json::from_value(golden["gh_argv"].clone())?,
    })
}

/// Runs the port (and, if configured, the legacy script) against the stub.
pub(crate) fn gh_case(
    name: &str,
    args: &[&str],
    replies: &[(String, Reply)],
    with_gh: bool,
) -> Result<Observed, Box<dyn Error>> {
    let stage = prepare(name, replies, with_gh)?;
    let prefix = ["ci-ops", "collect-metrics"].map(PathBuf::from);
    let actual = run(
        &stage,
        Path::new(env!("CARGO_BIN_EXE_xtask")),
        &prefix,
        args,
    )?;
    if let Some(python) = std::env::var_os(LEGACY_ENV).map(PathBuf::from) {
        let legacy = run(&stage, &python, &[repo_root().join(SCRIPT)], args)?;
        if std::env::var_os(CAPTURE_ENV).is_some() {
            capture(name, args, &legacy)?;
        }
        assert_eq!(
            actual,
            legacy,
            "{name}: Rust port differs from legacy {}",
            python.display()
        );
    }
    assert_eq!(
        actual,
        expected(name)?,
        "{name}: Rust port differs from captured golden"
    );
    Ok(actual)
}

/// A `gh run list/view --json RUN_FIELDS` run object.
pub(crate) fn run_object(id: i64, conclusion: &str) -> Value {
    json!({
        "databaseId": id,
        "attempt": 1,
        "workflowName": "PR Builds",
        "displayTitle": format!("run {id}"),
        "event": "pull_request",
        "status": "completed",
        "conclusion": conclusion,
        "createdAt": "2026-07-01T00:00:00Z",
        "startedAt": "2026-07-01T00:00:05Z",
        "updatedAt": "2026-07-01T00:10:00Z",
        "url": format!("https://example.test/runs/{id}"),
        "headSha": format!("sha-{id}"),
        "headBranch": "feature",
    })
}

pub(crate) fn job(id: i64, name: &str, started_second: i64) -> Value {
    json!({
        "id": id,
        "name": name,
        "status": "completed",
        "conclusion": "success",
        "created_at": "2026-07-01T00:00:10Z",
        "started_at": format!("2026-07-01T00:00:{:02}Z", started_second % 60),
        "completed_at": "2026-07-01T00:09:00Z",
        "html_url": format!("https://example.test/jobs/{id}"),
        "labels": ["ubuntu-24.04"],
        "steps": [],
    })
}
