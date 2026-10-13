//! Actual canary steps with finite Docker/date boundaries and no daemon access.
use crate::{
    process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    },
    workflow_yaml,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

fn step(job: &str, name: &str) -> String {
    let tree = workflow_yaml::parse(include_str!(
        "../../../../.github/workflows/depot-registry-canary.yml"
    ))
    .unwrap();
    let workflow_yaml::Node::Seq(steps) = tree
        .get("jobs")
        .unwrap()
        .get(job)
        .unwrap()
        .get("steps")
        .unwrap()
    else {
        panic!("steps")
    };
    steps
        .iter()
        .find(|step| step.get("name").and_then(workflow_yaml::Node::text) == Some(name))
        .unwrap()
        .get("run")
        .unwrap()
        .text()
        .unwrap()
        .to_owned()
}
fn tool(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|dir| dir.join(name))
        .find(|path| {
            path.is_file() && fs::metadata(path).unwrap().permissions().mode() & 0o111 != 0
        })
        .unwrap_or_else(|| panic!("registry fixture requires {name}"))
        .canonicalize()
        .unwrap()
}
fn executable(path: &Path, source: &str) {
    fs::write(path, format!("#!/bin/sh\nset -eu\n{source}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
struct RegistryFixture {
    temp: tempfile::TempDir,
    digest: String,
    upstream: String,
    depot: String,
}
impl RegistryFixture {
    fn new() -> Self {
        let digest = format!("sha256:{}", "a".repeat(64));
        let fixture = Self {
            temp: tempfile::tempdir().unwrap(),
            upstream: format!("ubuntu@{digest}"),
            depot: format!("mesh.registry.depot.dev/library/ubuntu@{digest}"),
            digest,
        };
        let bin = fixture.root().join("bin");
        fs::create_dir(&bin).unwrap();
        for name in ["sed", "head", "jq"] {
            std::os::unix::fs::symlink(tool(name), bin.join(name)).unwrap();
        }
        executable(
            &bin.join("docker"),
            r#"printf '%s\n' "$*" >> "$FIXTURE/docker.calls"
case "$1" in
  pull) [ "$#" = 2 ] && [ "$2" = "$EXPECTED_IMAGE" ] || exit 91
    [ "$FAIL_PULL" != true ] || exit 92 ;;
  image) [ "$2" = inspect ] && [ "$3" = "$EXPECTED_IMAGE" ] || exit 93
    printf 'fixture/image@%s\n' "$RESOLVED_DIGEST" ;;
  *) exit 94 ;;
esac"#,
        );
        executable(
            &bin.join("date"),
            r#"[ "$*" = '+%s%N' ] || exit 95
if [ -f "$FIXTURE/date.called" ]; then printf '31000000000\n'; else printf '1000000000\n'; fi
printf called > "$FIXTURE/date.called""#,
        );
        fixture
    }
    fn root(&self) -> &Path {
        self.temp.path()
    }
    fn run(&self, job: &str, name: &str, overrides: &[(&str, &str)]) -> process::RawProcessReport {
        let mut environment: BTreeMap<_, _> = [
            ("PATH", self.root().join("bin").into_os_string()),
            ("FIXTURE", self.root().into()),
            ("GITHUB_OUTPUT", self.root().join("output").into_os_string()),
            ("GITHUB_REPOSITORY", "Mesh-LLM/mesh-llm".into()),
            ("GITHUB_REF", "refs/heads/main".into()),
            ("GITHUB_EVENT_NAME", "workflow_dispatch".into()),
            ("DEPOT_REGISTRY_HOST", "mesh.registry.depot.dev".into()),
            ("DEPOT_REPOSITORY", "library/ubuntu".into()),
            ("UPSTREAM_IMAGE", self.upstream.clone().into()),
            ("DEPOT_IMAGE", self.depot.clone().into()),
            ("EXPECTED_DIGEST", self.digest.clone().into()),
            ("RESOLVED_DIGEST", self.digest.clone().into()),
            ("EXPECTED_IMAGE", self.upstream.clone().into()),
            ("SOURCE", "upstream".into()),
            ("SAMPLE", "1".into()),
            ("DEPOT_ORG_ID", "1ntz5vlngn".into()),
            ("FAIL_PULL", "false".into()),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value)))
        .collect();
        for (key, value) in overrides {
            environment.insert((*key).into(), Value::Public((*value).into()));
        }
        let result = process::supervise_raw(
            &ProcessSpec {
                executable: tool("bash"),
                cwd: self.root().into(),
                environment,
                arguments: vec![
                    Value::Public("-c".into()),
                    Value::Public(step(job, name).into()),
                ],
            },
            &Limits {
                execution: Duration::from_secs(5),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(
            result.process.cleanup.complete && result.process.failure.is_none(),
            "{result:?}"
        );
        result
    }
}

#[test]
fn actual_registry_canary_policy_binds_digest_and_mirror_and_refuses_untrusted_or_invalid_inputs() {
    let fixture = RegistryFixture::new();
    let result = fixture.run(
        "policy",
        "Validate trusted invocation and image references",
        &[],
    );
    assert!(result.process.success(), "{result:?}");
    assert_eq!(
        fs::read_to_string(fixture.root().join("output")).unwrap(),
        format!("digest={}\ndepot_image={}\n", fixture.digest, fixture.depot)
    );
    for (key, value) in [
        ("GITHUB_REPOSITORY", "foreign/repository"),
        ("GITHUB_REF", "refs/heads/feature"),
        ("GITHUB_EVENT_NAME", "pull_request"),
        ("DEPOT_REGISTRY_HOST", "foreign.example"),
        ("DEPOT_REPOSITORY", "../ubuntu"),
        ("DEPOT_REPOSITORY", "library/ubuntu;touch sentinel"),
        ("UPSTREAM_IMAGE", "ubuntu:latest"),
        ("UPSTREAM_IMAGE", "ubuntu@sha256:abc"),
    ] {
        let fixture = RegistryFixture::new();
        let result = fixture.run(
            "policy",
            "Validate trusted invocation and image references",
            &[(key, value)],
        );
        assert!(!result.process.success(), "{key}: {result:?}");
        assert!(!fixture.root().join("output").exists());
        assert!(!fixture.root().join("docker.calls").exists());
        assert!(!fixture.root().join("sentinel").exists());
    }
}

#[test]
fn actual_registry_canary_pull_observations_bind_source_sample_time_and_digest() {
    for source in ["upstream", "depot"] {
        for sample in 1..=5 {
            let fixture = RegistryFixture::new();
            let image = if source == "depot" {
                &fixture.depot
            } else {
                &fixture.upstream
            };
            let sample_text = sample.to_string();
            let result = fixture.run(
                "pull",
                "Pull exact image on a fresh runner",
                &[
                    ("SOURCE", source),
                    ("SAMPLE", &sample_text),
                    ("EXPECTED_IMAGE", image),
                ],
            );
            assert!(result.process.success(), "{source}/{sample}: {result:?}");
            let observation: serde_json::Value =
                serde_json::from_slice(&fs::read(fixture.root().join("observation.json")).unwrap())
                    .unwrap();
            assert_eq!(
                observation,
                serde_json::json!({"source":source,"sample":sample,"elapsed_ms":30000,"digest":fixture.digest})
            );
            let calls = fs::read_to_string(fixture.root().join("docker.calls")).unwrap();
            assert_eq!(calls.lines().count(), 2);
            assert!(calls.lines().next().unwrap().ends_with(image));
        }
    }
}

#[test]
fn actual_registry_canary_refuses_wrong_runner_failed_pull_or_digest_without_observation() {
    let different_digest = format!("sha256:{}", "b".repeat(64));
    for (key, value) in [
        ("DEPOT_ORG_ID", "foreign"),
        ("DEPOT_ORG_ID", ""),
        ("FAIL_PULL", "true"),
        ("RESOLVED_DIGEST", "sha256:wrong"),
        ("RESOLVED_DIGEST", different_digest.as_str()),
    ] {
        let fixture = RegistryFixture::new();
        let result = fixture.run(
            "pull",
            "Pull exact image on a fresh runner",
            &[(key, value)],
        );
        assert!(!result.process.success(), "{key}: {result:?}");
        assert!(!fixture.root().join("observation.json").exists());
        if key == "DEPOT_ORG_ID" {
            assert!(!fixture.root().join("docker.calls").exists());
            assert!(!fixture.root().join("date.called").exists());
        }
    }
}
