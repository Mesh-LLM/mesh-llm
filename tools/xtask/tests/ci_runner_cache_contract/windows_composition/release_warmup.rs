//! Release-equivalent CUDA warm rows and finite execution of the actual pruner.
use super::super::{
    support,
    workflow_yaml::{self, Node},
};
use serde_json::{Value, json};
use std::{fs, process::Command};

fn workflow(name: &str) -> Node {
    let path = support::root().join(format!(".github/workflows/{name}.yml"));
    workflow_yaml::parse(&fs::read_to_string(path).unwrap()).unwrap()
}

fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}

fn job<'a>(document: &'a Node, name: &str) -> &'a Node {
    document.get("jobs").unwrap().get(name).unwrap()
}

fn rows(job: &Node) -> &[Node] {
    let Some(Node::Seq(rows)) = job
        .get("strategy")
        .and_then(|strategy| strategy.get("matrix"))
        .and_then(|matrix| matrix.get("include"))
    else {
        panic!("matrix include rows required");
    };
    rows
}

fn step<'a>(job: &'a Node, key: &str, value: &str) -> &'a Node {
    let Some(Node::Seq(steps)) = job.get("steps") else {
        panic!("job steps required");
    };
    let matching = steps
        .iter()
        .filter(|step| text(step, key) == value)
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1, "unique step {value}");
    matching[0]
}

#[test]
fn windows_warmer_matches_every_release_cuda_configuration() {
    let release = workflow("release");
    let warmer = workflow("windows-warm-caches");
    let producer = job(&release, "build_native_runtime_windows_gpu");
    let warm = job(&warmer, "warm_windows_gpu");
    let warm_rows = rows(warm);
    assert_eq!(
        warm_rows
            .iter()
            .filter(|row| text(row, "backend") == "cuda")
            .count(),
        3
    );
    assert!(warm_rows.iter().any(|row| text(row, "build_args") == "75"));
    let install = step(warm, "name", "Install CUDA toolkit")
        .get("with")
        .unwrap();
    let default_packages = text(install, "sub-packages")
        .strip_prefix("${{ matrix.cuda_sub_packages || '")
        .unwrap()
        .strip_suffix("' }}")
        .unwrap();
    for release_row in rows(producer)
        .iter()
        .filter(|row| text(row, "backend") == "cuda")
    {
        let architectures = text(release_row, "cuda_architectures");
        let matching = warm_rows
            .iter()
            .filter(|row| {
                text(row, "backend") == "cuda" && text(row, "build_args") == architectures
            })
            .collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "exact release row for {architectures}");
        let row = matching[0];
        let version = release_row
            .get("cuda_version")
            .and_then(Node::text)
            .map(str::to_owned)
            .unwrap_or_else(|| {
                text(producer.get("env").unwrap(), "WINDOWS_CUDA_VERSION")
                    .replace("matrix.cuda_version || ", "")
            });
        assert_eq!(text(row, "cuda_version"), version);
        assert_eq!(
            row.get("cuda_major").and_then(Node::text).unwrap_or("12"),
            text(release_row, "cuda_major")
        );
        let packages = row
            .get("cuda_sub_packages")
            .and_then(Node::text)
            .unwrap_or(default_packages);
        assert_eq!(
            serde_json::from_str::<Value>(packages).unwrap(),
            serde_json::from_str::<Value>(text(release_row, "cuda_sub_packages")).unwrap()
        );
    }
    for variable in ["WINDOWS_CUDA_VERSION", "MESH_CUDA_VERSION"] {
        assert_eq!(
            text(warm.get("env").unwrap(), variable),
            "${{ matrix.cuda_version || vars.CUDA_VERSION || '12.6.3' }}"
        );
    }
    let restore = step(warm, "uses", "./.github/actions/restore-windows-abi-cache")
        .get("with")
        .unwrap();
    assert_eq!(
        text(restore, "architecture_set"),
        "${{ matrix.build_args }}"
    );
    assert_eq!(
        text(restore, "cuda_toolchain_version"),
        "${{ env.WINDOWS_CUDA_VERSION }}"
    );
    assert_eq!(text(install, "cuda"), "${{ env.WINDOWS_CUDA_VERSION }}");
    let prepare = step(
        warm,
        "uses",
        "./.github/actions/prepare-native-runtime-input",
    )
    .get("env")
    .unwrap();
    assert_eq!(
        text(prepare, "MESH_LLM_CUDA_TOOLKIT_MAJOR"),
        "${{ matrix.cuda_major || '12' }}"
    );
    assert_eq!(
        text(prepare, "LLAMA_STAGE_CUDA_ARCHITECTURES"),
        "${{ matrix.backend == 'cuda' && matrix.build_args || '' }}"
    );
    let paths = warmer
        .get("on")
        .unwrap()
        .get("push")
        .unwrap()
        .get("paths")
        .unwrap()
        .list();
    assert!(paths.contains(&".github/workflows/release.yml"));
}

fn prune(caches: Value, retention: &str) -> Value {
    let fixture = support::Fixture::new();
    let warmer = workflow("windows-warm-caches");
    let pruner = job(&warmer, "prune_windows_abi_caches");
    let script = text(
        step(pruner, "name", "Prune old Windows ABI caches")
            .get("with")
            .unwrap(),
        "script",
    );
    let actual = fixture.path().join("prune.js");
    let harness = fixture.path().join("harness.js");
    fs::write(&actual, script).unwrap();
    fs::write(&harness, PRUNE_HARNESS).unwrap();
    let mut command = Command::new("node");
    command
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap())
        .env("HOME", fixture.path())
        .env("CACHE_NAMESPACE", "mesh-llm")
        .env("TARGET_REF", "refs/heads/main")
        .env("RETENTION", retention)
        .env("FIXTURE_CACHES", caches.to_string())
        .arg(harness)
        .arg(actual);
    let result = fixture.run(command);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&result.stdout).unwrap()
}

#[test]
fn windows_pruning_retains_each_cuda_architecture_set_independently() {
    let warmer = workflow("windows-warm-caches");
    let cuda = rows(job(&warmer, "warm_windows_gpu"))
        .iter()
        .filter(|row| text(row, "backend") == "cuda")
        .collect::<Vec<_>>();
    let mut caches = Vec::new();
    for generation in 0..3 {
        for (variant, row) in cuda.iter().enumerate() {
            let architecture = text(row, "build_args").replace(';', "_");
            caches.push(json!({
                "id": (3 - generation) * 10 + variant,
                "key": format!("mesh-llm-windows-2022-skippy-abi-cuda-{architecture}-toolkit-image-{generation}-hash"),
            }));
        }
    }
    caches.push(json!({"id": 999, "key": "unrelated-cache"}));
    for (retention, expected) in [("2", vec![10, 11, 12]), ("1", vec![10, 11, 12, 20, 21, 22])] {
        let output = prune(json!(caches), retention);
        assert_eq!(output["failures"], json!([]));
        let mut deleted = output["deleted"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap())
            .collect::<Vec<_>>();
        deleted.sort_unstable();
        assert_eq!(deleted, expected, "{output}");
    }
}

#[test]
fn windows_pruning_refuses_invalid_retention_without_deleting() {
    for retention in ["0", "-1", "invalid"] {
        let output = prune(json!([]), retention);
        assert_eq!(output["deleted"], json!([]));
        assert_eq!(output["failures"].as_array().unwrap().len(), 1);
    }
}

const PRUNE_HARNESS: &str = r#"'use strict';
const fs = require('fs');
const caches = JSON.parse(process.env.FIXTURE_CACHES);
const deleted = [], failures = [];
const context = {repo: {owner: 'fixture-owner', repo: 'fixture-repo'}};
const core = {notice: () => {}, setFailed: message => failures.push(message)};
global.fetch = () => { throw Error('network forbidden'); };
let requests = 0;
const github = {request: async (route, args) => {
  if (++requests > 30 || args.owner !== context.repo.owner || args.repo !== context.repo.repo)
    throw Error('unexpected repository or unbounded requests');
  if (route === 'GET /repos/{owner}/{repo}/actions/caches') {
    if (args.ref !== process.env.TARGET_REF || args.per_page !== 100 ||
        args.sort !== 'created_at' || args.direction !== 'desc')
      throw Error('cache scope or ordering changed');
    return {data: {actions_caches: caches.slice((args.page - 1) * 100, args.page * 100)}};
  }
  if (route !== 'DELETE /repos/{owner}/{repo}/actions/caches/{cache_id}' ||
      !caches.some(cache => cache.id === args.cache_id))
    throw Error('unexpected deletion');
  deleted.push(args.cache_id);
  return {};
}};
(async () => {
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  await new AsyncFunction('github', 'core', 'context', fs.readFileSync(process.argv[2], 'utf8'))(github, core, context);
  console.log(JSON.stringify({deleted, failures}));
})().catch(error => { console.error(error); process.exitCode = 1; });
"#;
