//! Execute recovery source admission and runtime restore with owned inputs.
use crate::{
    process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    },
    workflow_yaml,
};
use flate2::{Compression, write::GzEncoder};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
const SHA: &str = "ff18c0b5a74c0317d943bb6611a1ce4836b35d61";
const ARCHIVE: &str = "mesh-llm-v0.76.1-x86_64-unknown-linux-gnu.tar.gz";

fn step(name: &str) -> String {
    let tree = workflow_yaml::parse(include_str!(
        "../../../../.github/workflows/resume-crates-release.yml"
    ))
    .unwrap();
    let workflow_yaml::Node::Seq(steps) = tree
        .get("jobs")
        .unwrap()
        .get("publish")
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
        .unwrap_or_else(|| panic!("recovery fixture requires {name}"))
        .canonicalize()
        .unwrap()
}
fn executable(path: &Path, source: &str) {
    fs::write(path, format!("#!/bin/sh\nset -eu\n{source}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
struct Fixture {
    temp: tempfile::TempDir,
}
impl Fixture {
    fn new() -> Self {
        let fixture = Self {
            temp: tempfile::tempdir().unwrap(),
        };
        for name in ["bin", "source", "scratch"] {
            fs::create_dir(fixture.root().join(name)).unwrap();
        }
        for name in ["awk", "mkdir", "cp", "sha256sum", "tar", "gzip", "find"] {
            std::os::unix::fs::symlink(tool(name), fixture.root().join("bin").join(name)).unwrap();
        }
        executable(
            &fixture.root().join("bin/git"),
            r#"printf '%s\n' "$*" >> "$FIXTURE/git.calls"
case "$*" in
  '-C controller ls-remote origin refs/tags/v0.76.1 refs/tags/v0.76.1^{}')
    printf '%s\trefs/tags/v0.76.1\n%s\trefs/tags/v0.76.1^{}\n' 1111111111111111111111111111111111111111 "$REMOTE_SHA" ;;
  '-C release-source rev-parse HEAD') printf '%s\n' "$CHECKOUT_SHA" ;;
  *) exit 93 ;;
esac"#,
        );
        executable(
            &fixture.root().join("bin/curl"),
            r#"output=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    --retry) shift 2 ;;
    --fail|--location) shift ;;
    https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.76.1/*) url="$1"; shift ;;
    *) exit 94 ;;
  esac
done
[ -n "$output" ]
printf '%s\n' "$url" >> "$FIXTURE/curl.calls"
cp "$FIXTURE/source/${url##*/}" "$output""#,
        );
        fixture
    }
    fn root(&self) -> &Path {
        self.temp.path()
    }
    fn artifact(&self, omitted: Option<&str>, corrupt: bool) {
        let mut archive = tar::Builder::new(GzEncoder::new(Vec::new(), Compression::default()));
        for name in ["libmtmd.so", "libllama-common.so", "libllama.so"] {
            if omitted == Some(name) {
                continue;
            }
            let body = b"inert owned fixture library\n";
            let mut header = tar::Header::new_gnu();
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive
                .append_data(
                    &mut header,
                    format!("mesh-bundle/native-runtimes/cpu/lib/{name}"),
                    &body[..],
                )
                .unwrap();
        }
        let bytes = archive.into_inner().unwrap().finish().unwrap();
        let digest = hex::encode(Sha256::digest(&bytes));
        fs::write(
            self.root().join("source").join(ARCHIVE),
            if corrupt { &b"corrupt"[..] } else { &bytes },
        )
        .unwrap();
        fs::write(
            self.root().join("source").join(format!("{ARCHIVE}.sha256")),
            format!("{digest}  {ARCHIVE}\n"),
        )
        .unwrap();
    }
    fn run(&self, name: &str, overrides: &[(&str, &str)]) -> process::RawProcessReport {
        let mut environment: BTreeMap<_, _> = [
            ("PATH", self.root().join("bin").into_os_string()),
            ("FIXTURE", self.root().into()),
            ("RUNNER_TEMP", self.root().join("scratch").into_os_string()),
            ("GITHUB_OUTPUT", self.root().join("output").into_os_string()),
            ("GITHUB_REPOSITORY", "Mesh-LLM/mesh-llm".into()),
            ("RELEASE_TAG", "v0.76.1".into()),
            ("EXPECTED_SOURCE_SHA", SHA.into()),
            ("REMOTE_SHA", SHA.into()),
            ("CHECKOUT_SHA", SHA.into()),
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
                arguments: vec![Value::Public("-c".into()), Value::Public(step(name).into())],
            },
            &Limits {
                execution: Duration::from_secs(8),
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
fn actual_crates_recovery_source_admission_requires_stable_tag_and_both_exact_identities() {
    let fixture = Fixture::new();
    let admitted = fixture.run("Verify exact stable release source", &[]);
    assert!(admitted.process.success(), "{admitted:?}");
    assert_eq!(
        fs::read_to_string(fixture.root().join("git.calls"))
            .unwrap()
            .lines()
            .count(),
        2
    );
    for (key, value) in [
        ("RELEASE_TAG", "v0.76.1-rc.1"),
        ("EXPECTED_SOURCE_SHA", "ff18c0b"),
        ("REMOTE_SHA", "1111111111111111111111111111111111111111"),
        ("CHECKOUT_SHA", "1111111111111111111111111111111111111111"),
    ] {
        let fixture = Fixture::new();
        let result = fixture.run("Verify exact stable release source", &[(key, value)]);
        assert!(!result.process.success(), "{key}: {result:?}");
        if key == "RELEASE_TAG" || key == "EXPECTED_SOURCE_SHA" {
            assert!(!fixture.root().join("git.calls").exists());
        }
    }
}

#[test]
fn actual_crates_recovery_runtime_restore_requires_checksum_and_every_library_before_output() {
    for omitted in [
        None,
        Some("libmtmd.so"),
        Some("libllama-common.so"),
        Some("libllama.so"),
        Some("corrupt"),
    ] {
        let fixture = Fixture::new();
        fixture.artifact(omitted, omitted == Some("corrupt"));
        let result = fixture.run("Restore verified release runtime libraries", &[]);
        assert_eq!(
            result.process.success(),
            omitted.is_none(),
            "{omitted:?}: {result:?}"
        );
        let output = fs::read_to_string(fixture.root().join("output")).unwrap_or_default();
        if omitted.is_none() {
            let directory = PathBuf::from(output.trim().strip_prefix("lib_dir=").unwrap());
            assert!(directory.starts_with(fixture.root().join("scratch")));
            for name in ["libmtmd.so", "libllama-common.so", "libllama.so"] {
                assert!(directory.join(name).is_file());
            }
        } else {
            assert!(output.is_empty());
        }
        assert_eq!(
            fs::read_to_string(fixture.root().join("curl.calls"))
                .unwrap()
                .lines()
                .count(),
            2
        );
    }
}
