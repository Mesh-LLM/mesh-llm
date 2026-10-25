//! Actual prepare producer against owned local repositories; no native build.
use super::process;
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, time::Duration};

struct LocalQueue {
    directory: tempfile::TempDir,
}
impl LocalQueue {
    fn new() -> Self {
        let fixture = Self {
            directory: tempfile::tempdir().unwrap(),
        };
        fs::create_dir(fixture.path().join("upstream")).unwrap();
        fs::create_dir(fixture.path().join("patches")).unwrap();
        fixture.git("upstream", &["init", "-q", "--initial-branch=master"]);
        fixture.git("upstream", &["config", "uploadpack.allowFilter", "true"]);
        fixture.write("upstream/sample.txt", "one\ntwo\nthree\n");
        fixture.git("upstream", &["add", "sample.txt"]);
        fixture.git("upstream", &["commit", "-qm", "base"]);
        fixture.write(
            "pin",
            &format!(
                "{}\n",
                fixture.git("upstream", &["rev-parse", "HEAD"]).trim()
            ),
        );
        fixture.git("", &["clone", "-q", "upstream", "author"]);
        fixture
    }
    fn path(&self) -> &Path {
        self.directory.path()
    }
    fn write(&self, relative: &str, text: &str) {
        let path = self.path().join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    fn environment(&self) -> BTreeMap<std::ffi::OsString, process::Value> {
        [
            ("PATH", "/usr/bin:/bin".to_owned()),
            ("HOME", self.path().to_string_lossy().into_owned()),
            ("GIT_CONFIG_NOSYSTEM", "1".to_owned()),
            ("GIT_CONFIG_GLOBAL", "/dev/null".to_owned()),
            ("GIT_ALLOW_PROTOCOL", "file".to_owned()),
            ("GIT_TERMINAL_PROMPT", "0".to_owned()),
            ("GIT_AUTHOR_NAME", "Queue Author".to_owned()),
            ("GIT_AUTHOR_EMAIL", "author@example.invalid".to_owned()),
            ("GIT_COMMITTER_NAME", "Queue Committer".to_owned()),
            (
                "GIT_COMMITTER_EMAIL",
                "committer@example.invalid".to_owned(),
            ),
            ("GIT_AUTHOR_DATE", "2000-01-01T00:00:00Z".to_owned()),
            ("GIT_COMMITTER_DATE", "2000-01-01T00:00:00Z".to_owned()),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), process::Value::Public(v.into())))
        .collect()
    }
    fn command(
        &self,
        executable: &str,
        cwd: &Path,
        arguments: &[&str],
        extra: &[(&str, String)],
    ) -> (bool, String, String) {
        let mut environment = self.environment();
        for (key, value) in extra {
            environment.insert((*key).into(), process::Value::Public(value.as_str().into()));
        }
        let raw = process::supervise_raw(
            &process::ProcessSpec {
                executable: executable.into(),
                cwd: cwd.into(),
                environment,
                arguments: arguments
                    .iter()
                    .map(|v| process::Value::Public((*v).into()))
                    .collect(),
            },
            &process::Limits {
                execution: Duration::from_secs(20),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
            &process::Cancellation::default(),
            process::RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        let report = raw.process;
        assert_eq!(report.outcome, process::Outcome::Exited);
        assert!(
            report.failure.is_none()
                && report.cleanup.complete
                && !report.cleanup.forced
                && !report.cleanup.graceful_signal_failed
                && report.cleanup.failure.is_none(),
            "{report:?}"
        );
        let stdout = raw.stdout.unwrap();
        let stderr = raw.stderr.unwrap();
        assert_eq!(stdout.as_bytes().len() as u64, report.stdout.bytes_seen);
        assert_eq!(stderr.as_bytes().len() as u64, report.stderr.bytes_seen);
        (
            report.status.unwrap().success(),
            String::from_utf8(stdout.as_bytes().to_vec()).unwrap(),
            String::from_utf8(stderr.as_bytes().to_vec()).unwrap(),
        )
    }
    fn git(&self, relative: &str, arguments: &[&str]) -> String {
        let mut args = vec![
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
        ];
        args.extend_from_slice(arguments);
        let (ok, out, error) =
            self.command("/usr/bin/git", &self.path().join(relative), &args, &[]);
        assert!(ok, "{arguments:?}: {error}");
        out
    }
    fn patch(&self, relative: &str, contents: &str, subject: &str) {
        self.write("author/sample.txt", contents);
        self.git("author", &["commit", "-qam", subject]);
        let patch = self.git("author", &["format-patch", "-1", "--stdout"]);
        self.write(&format!("patches/{relative}"), &patch);
    }
    fn prepare(&self, work: &str, extra: &[(&str, String)]) -> (bool, String, String) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        let script = root.join("skippy/scripts/prepare-llama.sh");
        let mut settings = vec![
            (
                "LLAMA_UPSTREAM_URL",
                format!("file://{}", self.path().join("upstream").display()),
            ),
            (
                "LLAMA_WORKDIR",
                self.path().join(work).display().to_string(),
            ),
            (
                "LLAMA_PIN_FILE",
                self.path().join("pin").display().to_string(),
            ),
            (
                "LLAMA_PATCH_DIR",
                self.path().join("patches").display().to_string(),
            ),
            ("LLAMA_GIT_MAX_ATTEMPTS", "1".into()),
            (
                "MESH_LLM_AUTOMATION_BIN",
                env!("CARGO_BIN_EXE_xtask").into(),
            ),
        ];
        settings.extend_from_slice(extra);
        self.command(
            "/bin/bash",
            &root,
            &[script.to_str().unwrap(), "pinned"],
            &settings,
        )
    }
    fn prepared(&self, work: &str) {
        let (ok, _, error) = self.prepare(work, &[]);
        assert!(ok, "{error}");
    }
}

#[test]
fn prepare_producer_replaces_blobless_checkout_and_merges_disjoint_pinned_changes() {
    let fixture = LocalQueue::new();
    fixture.patch("0001-local.patch", "patched\ntwo\nthree\n", "local patch");
    fixture.write("upstream/sample.txt", "one\ntwo\nupstream\n");
    fixture.git("upstream", &["commit", "-qam", "upstream change"]);
    fixture.write(
        "pin",
        &format!(
            "{}\n",
            fixture.git("upstream", &["rev-parse", "HEAD"]).trim()
        ),
    );
    let url = format!("file://{}", fixture.path().join("upstream").display());
    fixture.git("", &["clone", "-q", "--filter=blob:none", &url, "work"]);
    assert_eq!(
        fixture
            .git(
                "work",
                &["config", "--bool", "--get", "remote.origin.promisor"]
            )
            .trim(),
        "true"
    );
    fixture.prepared("work");
    assert_eq!(
        fs::read_to_string(fixture.path().join("work/sample.txt")).unwrap(),
        "patched\ntwo\nupstream\n"
    );
    for key in ["remote.origin.promisor", "extensions.partialClone"] {
        let (ok, _, _) = fixture.command(
            "/usr/bin/git",
            &fixture.path().join("work"),
            &["config", "--get", key],
            &[],
        );
        assert!(!ok, "partial marker {key} remains");
    }
    fixture.directory.close().unwrap();
}
#[test]
fn prepare_producer_rejects_duplicate_queue_before_reset_or_application() {
    let fixture = LocalQueue::new();
    fixture.git("", &["clone", "-q", "upstream", "work"]);
    let before = fixture.git("work", &["rev-parse", "HEAD"]);
    fixture.write("work/untracked", "preserved\n");
    fixture.write("patches/0001-first.patch", "invalid first\n");
    fixture.write("patches/0001-second.patch", "invalid duplicate\n");
    let (ok, out, error) = fixture.prepare("work", &[]);
    assert!(
        !ok && error.contains("invalid llama patch sequence: expected 0002, found 0001"),
        "{out} {error}"
    );
    assert_eq!(fixture.git("work", &["rev-parse", "HEAD"]), before);
    assert_eq!(
        fs::read_to_string(fixture.path().join("work/untracked")).unwrap(),
        "preserved\n"
    );
    assert!(!fixture.path().join("work/.mesh-llm-patched-sha").exists());
    fixture.directory.close().unwrap();
}
#[test]
fn prepare_producer_applies_core_support_generated_in_order_and_accepts_crlf_series() {
    let fixture = LocalQueue::new();
    fixture.patch("0001-core.patch", "core\n", "core change");
    fixture.patch(
        "model_support/0001-models-test.patch",
        "support\n",
        "model support change",
    );
    fixture.patch(
        "generated/0001-family-test.patch",
        "generated\n",
        "generated change",
    );
    fixture.write("patches/model_support/series", "0001-models-test.patch\r\n");
    fixture.write("patches/generated/series", "0001-family-test.patch\r\n");
    fixture.prepared("work");
    assert_eq!(
        fs::read_to_string(fixture.path().join("work/sample.txt")).unwrap(),
        "generated\n"
    );
    assert_eq!(
        fixture
            .git(
                "work",
                &["log", "--reverse", "--format=%s", "--max-count=3"]
            )
            .lines()
            .collect::<Vec<_>>(),
        ["core change", "model support change", "generated change"]
    );
    fixture.directory.close().unwrap();
}
#[test]
fn prepare_producer_commit_identity_ignores_ambient_dates_identity_signing_and_hooks() {
    use std::os::unix::fs::PermissionsExt as _;
    let fixture = LocalQueue::new();
    fixture.patch("0001-local.patch", "patched\n", "local patch");
    for name in ["applypatch-msg", "pre-applypatch"] {
        fixture.write(
            &format!("hooks/{name}"),
            "#!/bin/sh\nprintf unexpected > \"$HOOK_MARKER\"\nexit 71\n",
        );
        fs::set_permissions(
            fixture.path().join("hooks").join(name),
            fs::Permissions::from_mode(0o700),
        )
        .unwrap();
    }
    let mut shas = Vec::new();
    for (work, date, zone) in [
        ("first", "2001-01-01T00:00:00Z", "UTC"),
        ("second", "2031-01-01T00:00:00Z", "America/Toronto"),
    ] {
        let (ok, _, error) = fixture.prepare(
            work,
            &[
                ("GIT_COMMITTER_DATE", date.into()),
                ("GIT_AUTHOR_NAME", work.into()),
                ("GIT_COMMITTER_EMAIL", format!("{work}@example.invalid")),
                ("TZ", zone.into()),
                ("GIT_CONFIG_COUNT", "2".into()),
                ("GIT_CONFIG_KEY_0", "commit.gpgsign".into()),
                ("GIT_CONFIG_VALUE_0", "true".into()),
                ("GIT_CONFIG_KEY_1", "core.hooksPath".into()),
                (
                    "GIT_CONFIG_VALUE_1",
                    fixture.path().join("hooks").display().to_string(),
                ),
                (
                    "HOOK_MARKER",
                    fixture.path().join("hook-ran").display().to_string(),
                ),
            ],
        );
        assert!(ok, "{error}");
        shas.push(
            fs::read_to_string(fixture.path().join(work).join(".mesh-llm-patched-sha")).unwrap(),
        );
        assert_eq!(
            fs::read_to_string(fixture.path().join(work).join(".mesh-llm-prepare-schema")).unwrap(),
            "5\n"
        );
        assert!(!fixture.path().join("hook-ran").exists());
    }
    assert_eq!(shas[0], shas[1]);
    fixture.directory.close().unwrap();
}
