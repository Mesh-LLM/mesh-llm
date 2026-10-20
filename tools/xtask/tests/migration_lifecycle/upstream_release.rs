//! Exercise the real selector CLI with bounded canned GitHub responses.
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    process::{Command, Output},
};

const SHA: &str = "1111111111111111111111111111111111111111";
const GH: &str = r#"#!/bin/sh
set -eu
printf '%s\n' "$@" >> requests
test "$1" = api
test "$2" = --hostname
test "$3" = github.com
test "$4" = --method
test "$5" = GET
test "$GH_TOKEN" = fixture-private-value
for endpoint do :; done
case "$endpoint" in
  repos/ggml-org/llama.cpp/releases/latest)
    cat release.json ;;
  repos/ggml-org/llama.cpp/git/ref/tags/v0.7.0)
    cat ref.json ;;
  *) exit 19 ;;
esac
"#;

struct Fixture(tempfile::TempDir);
impl Fixture {
    fn new(script: &str) -> Self {
        let fixture = Self(tempfile::tempdir().unwrap());
        let root = fixture.0.path();
        let gh = root.join("gh");
        fs::write(&gh, script).unwrap();
        fs::set_permissions(gh, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(root.join("release.json"), r#"{"id":123,"tag_name":"v0.7.0","draft":false,"prerelease":false,"published_at":"2026-10-15","target_commitish":"master"}"#).unwrap();
        fs::write(
            root.join("ref.json"),
            format!(r#"{{"ref":"refs/tags/v0.7.0","object":{{"type":"commit","sha":"{SHA}"}}}}"#),
        )
        .unwrap();
        fs::write(root.join("output"), "prior-output\n").unwrap();
        fs::write(root.join("summary"), "prior-summary\n").unwrap();
        fixture
    }
    fn run(&self, upstream: &str) -> Output {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(self.0.path())
            .env_clear()
            .env("PATH", format!("{}:/usr/bin:/bin", self.0.path().display()))
            .env("HOME", self.0.path())
            .env("GH_TOKEN", "fixture-private-value")
            .env("GH_HOST", "must-not-override.invalid")
            .args([
                "repository",
                "llama-upstream-release",
                "--upstream",
                upstream,
                "--github-output",
                "output",
                "--summary",
                "summary",
            ])
            .output()
            .unwrap()
    }
    fn read(&self, name: &str) -> String {
        fs::read_to_string(self.0.path().join(name)).unwrap()
    }
}

#[test]
fn stable_release_cli_freezes_tag_and_appends_github_evidence() {
    let fixture = Fixture::new(GH);
    let result = fixture.run("");
    assert!(result.status.success(), "{result:?}");
    assert_eq!(fixture.read("output"), format!("prior-output\nsha={SHA}\n"));
    let summary = fixture.read("summary");
    assert!(summary.starts_with("prior-summary\nLatest stable"));
    assert!(summary.contains("releases/tag/v0.7.0"));
    assert!(summary.contains(SHA));
    assert_eq!(
        fixture.read("requests").matches("releases/latest").count(),
        1
    );
    assert_eq!(
        fixture
            .read("requests")
            .matches("git/ref/tags/v0.7.0")
            .count(),
        2
    );
}

#[test]
fn stable_release_cli_override_does_not_contact_github() {
    let fixture = Fixture::new("#!/bin/sh\nexit 99\n");
    let result = fixture.run(SHA);
    assert!(result.status.success(), "{result:?}");
    assert!(!fixture.0.path().join("requests").exists());
    assert!(fixture.read("summary").contains("override"));
}

#[test]
fn stable_release_cli_accepts_complete_json_beyond_diagnostic_line_limit() {
    let fixture = Fixture::new(GH);
    let mut release: serde_json::Value =
        serde_json::from_str(&fixture.read("release.json")).unwrap();
    release["body"] = serde_json::json!("upstream release notes ".repeat(5000));
    fs::write(
        fixture.0.path().join("release.json"),
        serde_json::to_vec(&release).unwrap(),
    )
    .unwrap();
    let result = fixture.run("");
    assert!(result.status.success(), "{result:?}");
    assert!(fixture.read("output").contains(SHA));
}

#[test]
fn stable_release_cli_api_errors_and_malformed_json_publish_nothing() {
    for script in [
        "#!/bin/sh\nprintf '%s' \"$GH_TOKEN\" >&2\nexit 1\n",
        "#!/bin/sh\nprintf 'not-json'\n",
        "#!/bin/sh\nhead -c 1048577 /dev/zero\n",
    ] {
        let fixture = Fixture::new(script);
        let result = fixture.run("");
        assert!(!result.status.success(), "{result:?}");
        assert_eq!(fixture.read("output"), "prior-output\n");
        assert_eq!(fixture.read("summary"), "prior-summary\n");
        for stream in [&result.stdout, &result.stderr] {
            assert!(!String::from_utf8_lossy(stream).contains("fixture-private-value"));
        }
    }
}
