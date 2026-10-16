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
test "$GH_TOKEN" = fixture-private-value
case "$6" in
  graphql)
    test "$5" = POST
    test "$7" = -f
    case "$8" in 'query=query('*'releases(first:100,'*) ;; *) exit 20 ;; esac
    case "$*" in *cursor=next*) cat release-2.json ;; *) cat release.json ;; esac ;;
  repos/ggml-org/llama.cpp/git/ref/tags/v0.7.0|repos/ggml-org/llama.cpp/git/ref/tags/b11671)
    test "$5" = GET
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
        fs::write(
            root.join("release.json"),
            serde_json::to_vec(&serde_json::json!({
                "data":{"repository":{"releases":{
                    "nodes":[{"id":123,"tag_name":"v0.7.0","draft":false,"prerelease":false,
                        "published_at":"2026-10-15T17:13:19Z","target_commitish":"master"}],
                    "pageInfo":{"hasNextPage":false,"endCursor":null}
                }}}
            }))
            .unwrap(),
        )
        .unwrap();
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
fn release_cli_freezes_tag_and_appends_github_evidence() {
    let fixture = Fixture::new(GH);
    let result = fixture.run("");
    assert!(result.status.success(), "{result:?}");
    assert_eq!(fixture.read("output"), format!("prior-output\nsha={SHA}\n"));
    let summary = fixture.read("summary");
    assert!(summary.starts_with("prior-summary\nNewest published llama.cpp release"));
    assert!(summary.contains("releases/tag/v0.7.0"));
    assert!(summary.contains(SHA));
    assert_eq!(
        fixture
            .read("requests")
            .matches("releases(first:100,")
            .count(),
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
fn release_cli_override_does_not_contact_github() {
    let fixture = Fixture::new("#!/bin/sh\nexit 99\n");
    let result = fixture.run(SHA);
    assert!(result.status.success(), "{result:?}");
    assert!(!fixture.0.path().join("requests").exists());
    assert!(fixture.read("summary").contains("override"));
}

#[test]
fn release_cli_accepts_complete_json_beyond_diagnostic_line_limit() {
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
fn release_cli_api_errors_and_malformed_json_publish_nothing() {
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

#[test]
fn release_cli_selects_newer_prerelease_across_pages_and_records_its_kind() {
    let fixture = Fixture::new(GH);
    let mut first: serde_json::Value = serde_json::from_str(&fixture.read("release.json")).unwrap();
    first["data"]["repository"]["releases"]["pageInfo"] =
        serde_json::json!({"hasNextPage":true,"endCursor":"next"});
    fs::write(
        fixture.0.path().join("release.json"),
        serde_json::to_vec(&first).unwrap(),
    )
    .unwrap();
    let mut second = first;
    second["data"]["repository"]["releases"]["nodes"] = serde_json::json!([
        {"id":124,"tag_name":"b11671","draft":false,"prerelease":true,
         "published_at":"2026-10-16T00:18:22Z"}
    ]);
    second["data"]["repository"]["releases"]["pageInfo"] =
        serde_json::json!({"hasNextPage":false,"endCursor":null});
    fs::write(
        fixture.0.path().join("release-2.json"),
        serde_json::to_vec(&second).unwrap(),
    )
    .unwrap();
    fs::write(
        fixture.0.path().join("ref.json"),
        format!(r#"{{"ref":"refs/tags/b11671","object":{{"type":"commit","sha":"{SHA}"}}}}"#),
    )
    .unwrap();
    let result = fixture.run("");
    assert!(result.status.success(), "{result:?}");
    let output: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(output["release_tag"], "b11671");
    assert_eq!(output["prerelease"], true);
    assert!(
        fixture
            .read("summary")
            .contains("llama.cpp prerelease: [b11671]")
    );
    assert!(fixture.read("requests").contains("cursor=next"));
}
