use crate::process;
use crate::workflow_yaml::{self, Node};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt as _,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::Duration,
};

pub(super) fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
pub(super) fn action(name: &str) -> Node {
    workflow_yaml::parse(
        &fs::read_to_string(root().join(format!(".github/actions/{name}/action.yml"))).unwrap(),
    )
    .unwrap()
}
pub(super) fn step(action: &Node, key: &str) -> String {
    let Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
        panic!("action steps")
    };
    steps
        .iter()
        .find_map(|step| {
            step.get(key).and_then(Node::text).or_else(|| {
                step.get("with")
                    .and_then(|n| n.get(key))
                    .and_then(Node::text)
            })
        })
        .unwrap()
        .to_owned()
}
pub(super) struct Fixture(pub(super) tempfile::TempDir);
impl Fixture {
    pub(super) fn new() -> Self {
        let f = Self(tempfile::tempdir().unwrap());
        fs::create_dir(f.path().join("bin")).unwrap();
        f
    }
    pub(super) fn path(&self) -> &Path {
        self.0.path()
    }
    pub(super) fn executable(&self, name: &str, body: &str) {
        let p = self.path().join("bin").join(name);
        fs::write(&p, format!("#!/bin/bash\nset -euo pipefail\n{body}\n")).unwrap();
        fs::set_permissions(p, fs::Permissions::from_mode(0o700)).unwrap();
    }
    pub(super) fn run(&self, command: Command) -> Output {
        let mut environment = ["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"]
            .into_iter()
            .filter_map(|key| {
                std::env::var_os(key).map(|value| (key.into(), process::Value::Public(value)))
            })
            .collect::<BTreeMap<_, _>>();
        for (key, value) in command.get_envs() {
            if let Some(value) = value {
                environment.insert(key.to_owned(), process::Value::Public(value.to_owned()));
            } else {
                environment.remove(key);
            }
        }
        let program = PathBuf::from(command.get_program());
        let executable = if program.is_absolute() {
            program
        } else {
            let path = command
                .get_envs()
                .find(|(key, _)| *key == "PATH")
                .and_then(|(_, value)| value)
                .map(|value| value.to_owned());
            let path = path.unwrap_or_else(|| std::env::var_os("PATH").unwrap());
            std::env::split_paths(&path)
                .map(|directory| directory.join(&program))
                .find(|candidate| {
                    candidate.is_file()
                        && fs::metadata(candidate).unwrap().permissions().mode() & 0o111 != 0
                })
                .expect("fixture tool executable")
        };
        let spec = process::ProcessSpec {
            executable,
            cwd: command
                .get_current_dir()
                .unwrap_or_else(|| self.path())
                .to_owned(),
            arguments: command
                .get_args()
                .map(|arg| process::Value::Public(arg.to_owned()))
                .collect(),
            environment,
        };
        let limits = process::Limits {
            execution: Duration::from_secs(8),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1024 * 1024,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        };
        let raw = process::supervise_raw(
            &spec,
            &limits,
            &process::Cancellation::default(),
            process::RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(1024 * 1024),
                stderr: std::num::NonZeroUsize::new(1024 * 1024),
            },
        )
        .unwrap();
        let report = raw.process;
        assert!(
            report.cleanup.complete
                && !report.cleanup.forced
                && !report.cleanup.graceful_signal_failed
                && report.cleanup.failure.is_none()
                && report.failure.is_none(),
            "{report:?}"
        );
        assert_eq!(report.outcome, process::Outcome::Exited, "{report:?}");
        let stdout = raw.stdout.unwrap();
        let stderr = raw.stderr.unwrap();
        assert_eq!(stdout.as_bytes().len() as u64, report.stdout.bytes_seen);
        assert_eq!(stderr.as_bytes().len() as u64, report.stderr.bytes_seen);
        // Privacy diagnostics may suppress cache credentials intentionally.
        // Complete raw fixture bytes are independent of sanitized persistence.
        Output {
            status: report.status.unwrap(),
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }
    pub(super) fn selector(&self, env: &[(&str, &str)]) -> Output {
        self.executable(
            "date",
            "[[ $* == '-u +%F' ]] || exit 41; printf '%s\\n' \"$FIXTURE_DATE\"",
        );
        let mut c = Command::new("bash");
        c.env_clear()
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.path().join("bin").display()),
            )
            .env("GITHUB_OUTPUT", self.path().join("outputs"));
        for (k, v) in [
            ("INPUT_EVENT_NAME", "pull_request"),
            ("INPUT_ORIGINAL_EVENT_NAME", ""),
            ("DISPATCH_ORIGINAL_EVENT_NAME", ""),
            ("INPUT_REPOSITORY", "Mesh-LLM/mesh-llm"),
            ("INPUT_HEAD_REPOSITORY", "Mesh-LLM/mesh-llm"),
            ("INPUT_HEAD_SHA", "0123456789abcdef0123456789abcdef01234567"),
            ("INPUT_REF", "refs/pull/12/merge"),
            ("INPUT_DEPOT_MAIN_ENABLED", "false"),
            ("INPUT_DEPOT_PR_ENABLED", "false"),
            ("INPUT_PR_CANARY_REF", ""),
            ("INPUT_FORCE_HOSTED", "false"),
            ("INPUT_MANUAL_USE_DEPOT", "false"),
            ("FIXTURE_DATE", "2026-09-03"),
        ] {
            c.env(k, v);
        }
        for (k, v) in env {
            c.env(k, v);
        }
        c.args(["-c", &step(&action("select-ci-runners"), "run")]);
        self.run(c)
    }
    pub(super) fn outputs(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for line in fs::read_to_string(self.path().join("outputs"))
            .unwrap()
            .lines()
        {
            let (k, v) = line.split_once('=').unwrap();
            assert!(
                map.insert(k.to_owned(), serde_json::Value::String(v.to_owned()))
                    .is_none()
            );
        }
        serde_json::Value::Object(map)
    }
    pub(super) fn cache(&self, env: &[(&str, &str)]) -> serde_json::Value {
        let document = action("configure-sccache-gha");
        for key in ["allow_depot_remote_cache", "allow_native_github_cache"] {
            assert_eq!(
                document
                    .get("inputs")
                    .unwrap()
                    .get(key)
                    .unwrap()
                    .get("default")
                    .and_then(Node::text),
                Some("false")
            );
        }
        let script = self.path().join("action.js");
        fs::write(&script, step(&action("configure-sccache-gha"), "script")).unwrap();
        let harness = self.path().join("harness.js");
        fs::write(&harness,r#"
const fs=require('fs');
const variables={},calls=[],secrets=[],failures=[],directories=[];
let starts=JSON.parse(process.env.START_CODES||'[0]');
const core={exportVariable:(k,v)=>{variables[k]=String(v);process.env[k]=String(v)},setSecret:v=>secrets.push(v),setFailed:v=>failures.push(v),info:()=>{},warning:()=>{}};
const io={mkdirP:async p=>directories.push(p)};
const exec={exec:async(command,args,options={})=>{if(command!=='sccache')throw Error('unexpected command');calls.push({args,env:{...(options.env||process.env)}});return args[0]==='--start-server'?(starts.shift()??0):args[0]==='--zero-stats'?Number(process.env.RESET_CODE||0):0;}};
(async()=>{const fn=new Function('core','io','exec',`return (async()=>{${fs.readFileSync(process.argv[2],'utf8')}\n})()`);await fn(core,io,exec);console.log(JSON.stringify({exports:variables,calls,secrets,failures,directories,job:process.env}));})().catch(e=>{console.error(e);process.exitCode=1});
"#).unwrap();
        let mut c = Command::new("node");
        c.env_clear()
            .env("PATH", std::env::var("PATH").unwrap())
            .arg(harness)
            .arg(script);
        for (k, v) in [
            ("INPUT_ALLOW_DEPOT_REMOTE_CACHE", "false"),
            ("INPUT_ALLOW_NATIVE_GITHUB_CACHE", "true"),
            ("GITHUB_EVENT_NAME", "push"),
            ("RUNNER_TEMP", "/fixture/runner temp"),
            ("ACTIONS_RUNTIME_TOKEN", "fixture-runtime-token"),
            ("ACTIONS_CACHE_URL", "http://cache.fixture"),
            ("ACTIONS_RESULTS_URL", "http://results.fixture"),
            ("SCCACHE_WEBDAV_ENDPOINT", ""),
            ("DEPOT_CACHE_TOKEN", ""),
        ] {
            c.env(k, v);
        }
        for (k, v) in env {
            c.env(k, v);
        }
        let output = self.run(c);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).unwrap()
    }
}
