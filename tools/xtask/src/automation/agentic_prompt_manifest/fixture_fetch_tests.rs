use serde_json::{Value, json};

use super::fixture_fetch;

fn dataset() -> Value {
    json!({"repo_id":"thoughtworks/agentic-coding-trajectories","repo_type":"dataset","revision":"a".repeat(40),"files":["README.md","sessions.parquet"],"parquet_file":"sessions.parquet"})
}

#[test]
fn fixture_fetch_arguments_preserve_pins_strict_verification_and_path_boundaries() {
    let cache = std::path::Path::new("fixture-cache");
    let verify = fixture_fetch::arguments(&dataset(), Some(cache), true).unwrap();
    let expected = [
        "cache",
        "verify",
        "thoughtworks/agentic-coding-trajectories",
        "--repo-type",
        "dataset",
        "--revision",
        &"a".repeat(40),
        "--fail-on-missing-files",
        "--cache-dir",
        "fixture-cache",
    ]
    .map(std::ffi::OsString::from);
    assert_eq!(verify, expected);
    let download = fixture_fetch::arguments(&dataset(), Some(cache), false).unwrap();
    let expected_download = [
        "download",
        "thoughtworks/agentic-coding-trajectories",
        "README.md",
        "sessions.parquet",
        "--quiet",
        "--repo-type",
        "dataset",
        "--revision",
        &"a".repeat(40),
        "--cache-dir",
        "fixture-cache",
    ]
    .map(std::ffi::OsString::from);
    assert_eq!(download, expected_download);
    for invalid in ["../escape", "/absolute", "--revision", "dir\\escape"] {
        let mut changed = dataset();
        changed["files"] = json!([invalid]);
        assert!(fixture_fetch::arguments(&changed, None, false).is_err());
    }
}

#[cfg(unix)]
mod owned {
    use super::*;
    use crate::process::{Cancellation, Value as ProcessValue};
    use std::{fs, os::unix::fs::PermissionsExt, time::Duration};

    fn fixture(directory: &std::path::Path, fail: bool) -> fixture_fetch::Adapter {
        let executable = directory.join("hf-fixture");
        fs::write(&executable,b"#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"$HF_TEST_LOG\"\nprintf 'end\\n' >> \"$HF_TEST_LOG\"\nif [ \"$1\" = download ]; then printf '%s\\n' \"$HF_TEST_SNAPSHOT\"; else if [ \"$HF_TEST_FAIL\" = 1 ]; then printf 'verification failed\\n' >&2; exit 23; fi; fi\n").unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
        fixture_fetch::Adapter {
            executable,
            cwd: directory.to_path_buf(),
            environment: [
                ("PATH".into(), ProcessValue::Public("/usr/bin:/bin".into())),
                (
                    "HF_TEST_LOG".into(),
                    ProcessValue::Public(directory.join("calls").into_os_string()),
                ),
                (
                    "HF_TEST_SNAPSHOT".into(),
                    ProcessValue::Public(directory.join("snapshot").into_os_string()),
                ),
                (
                    "HF_TEST_FAIL".into(),
                    ProcessValue::Public(if fail { "1" } else { "0" }.into()),
                ),
            ]
            .into_iter()
            .collect(),
            cache_dir: Some(directory.join("cache")),
            budget: Duration::from_secs(5),
        }
    }

    #[test]
    fn actual_fixture_hf_adapter_downloads_then_strictly_verifies_before_admitting_snapshot() {
        let directory = tempfile::tempdir().unwrap();
        let adapter = fixture(directory.path(), false);
        fs::create_dir(directory.path().join("snapshot")).unwrap();
        fs::write(
            directory.path().join("snapshot/sessions.parquet"),
            b"cached fixture",
        )
        .unwrap();
        let selected =
            fixture_fetch::fetch(&dataset(), &adapter, &Cancellation::default()).unwrap();
        assert_eq!(selected, directory.path().join("snapshot/sessions.parquet"));
        let trace = fs::read_to_string(directory.path().join("calls")).unwrap();
        let cache = directory.path().join("cache");
        let revision = "a".repeat(40);
        let expected = format!(
            "download\nthoughtworks/agentic-coding-trajectories\nREADME.md\nsessions.parquet\n--quiet\n--repo-type\ndataset\n--revision\n{revision}\n--cache-dir\n{}\nend\ncache\nverify\nthoughtworks/agentic-coding-trajectories\n--repo-type\ndataset\n--revision\n{revision}\n--fail-on-missing-files\n--cache-dir\n{}\nend\n",
            cache.display(),
            cache.display()
        );
        assert_eq!(trace, expected);
    }

    #[test]
    fn actual_fixture_hf_verification_failure_refuses_an_existing_parquet() {
        let directory = tempfile::tempdir().unwrap();
        let adapter = fixture(directory.path(), true);
        fs::create_dir(directory.path().join("snapshot")).unwrap();
        fs::write(
            directory.path().join("snapshot/sessions.parquet"),
            b"must not admit",
        )
        .unwrap();
        let error = fixture_fetch::fetch(&dataset(), &adapter, &Cancellation::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("verification failed"), "{error}");
        let trace = fs::read_to_string(directory.path().join("calls")).unwrap();
        assert!(trace.contains("cache\nverify\n"));
        assert!(trace.contains("--fail-on-missing-files\n"));
    }
}
