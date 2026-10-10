use super::{
    history_fetch::{self, Receipt},
    history_hub,
    history_upload::{self, Input, Kind},
};
use crate::{automation::private_state::PrivateState, process};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
fn state(parent: &Path) -> PrivateState {
    let state = PrivateState::create(parent, "history-fixture").unwrap();
    state.prepare().unwrap();
    state
}
#[cfg(unix)]
fn executable(parent: &Path, name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = parent.join(name);
    std::fs::write(&path, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    path
}
#[test]
fn anonymous_environment_has_no_token_or_shared_cache_and_publication_has_one_secret() {
    let parent = tempfile::tempdir().unwrap();
    let state = state(parent.path());
    let environment = history_hub::environment(&state, None);
    for key in [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HF_CACHE",
    ] {
        assert!(!environment.contains_key(std::ffi::OsStr::new(key)));
    }
    assert!(
        matches!(&environment[std::ffi::OsStr::new("HF_HUB_DISABLE_IMPLICIT_TOKEN")],process::Value::Public(value) if value=="1")
    );
    assert!(
        matches!(&environment[std::ffi::OsStr::new("HF_HOME")],process::Value::Public(value) if Path::new(value).starts_with(state.root()))
    );
    let environment = history_hub::environment(&state, Some("fixture-publication-token".into()));
    assert!(
        matches!(&environment[std::ffi::OsStr::new("HF_TOKEN")],process::Value::Secret(value) if value=="fixture-publication-token")
    );
    assert!(
        matches!(&environment[std::ffi::OsStr::new("HF_HUB_DISABLE_IMPLICIT_TOKEN")],process::Value::Public(value) if value=="0")
    );
    state.finish(Ok::<_, String>(())).unwrap();
}
#[test]
fn repository_identifiers_cannot_escape_the_fixed_hf_dataset_endpoint() {
    for repo in ["owner/name", "owner.with-dots/name_with-underscores"] {
        assert!(
            history_hub::lookup_url(repo)
                .unwrap()
                .starts_with("https://huggingface.co/api/datasets/")
        );
    }
    for repo in [
        "",
        "owner",
        "owner/name/extra",
        "../name",
        "owner/..",
        "owner/name?token=secret",
        "owner/name#fragment",
        "owner/name%2fescape",
        "owner/na me",
    ] {
        assert!(history_hub::repository(repo).is_err(), "{repo}");
    }
}
#[cfg(unix)]
#[test]
fn successful_anonymous_download_uses_staging_and_never_needs_lookup() {
    let parent = tempfile::tempdir().unwrap();
    let hf = executable(
        parent.path(),
        "hf",
        "test \"$1\" = download; test \"$3\" = --repo-type; test \"$4\" = dataset; test \"$5\" = --local-dir; test \"$7\" = --exclude; test \"$8\" = '*.md'; test -z \"${HF_TOKEN:-}\"; test \"$HF_HUB_DISABLE_IMPLICIT_TOKEN\" = 1; mkdir -p \"$6/data/runs\"; printf '{}' > \"$6/data/runs/fixture.jsonl\"",
    );
    let output = parent.path().join("history");
    assert_eq!(
        history_fetch::fetch(
            "owner/dataset",
            &output,
            &hf,
            Path::new("/bin/false"),
            Duration::from_secs(2),
            &process::Cancellation::default()
        )
        .unwrap(),
        Receipt::Downloaded
    );
    assert!(output.join("data/runs/fixture.jsonl").is_file());
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 2);
}
#[cfg(unix)]
#[test]
fn only_a_verified_lookup_404_bootstraps_and_partial_download_never_becomes_baseline() {
    let parent = tempfile::tempdir().unwrap();
    let hf = executable(
        parent.path(),
        "hf-failure",
        "mkdir -p \"$6/data/runs\"; printf partial > \"$6/data/runs/partial.jsonl\"; exit 1",
    );
    for (index, status) in ["200", "401", "403", "404", "500", "000", "garbage"]
        .into_iter()
        .enumerate()
    {
        let curl = executable(
            parent.path(),
            &format!("curl-{index}"),
            &format!("test \"$1\" = --disable; test -z \"${{HF_TOKEN:-}}\"; printf '{status}'"),
        );
        let output = parent.path().join(format!("history-{index}"));
        let result = history_fetch::fetch(
            "owner/dataset",
            &output,
            &hf,
            &curl,
            Duration::from_secs(2),
            &process::Cancellation::default(),
        );
        if status == "404" {
            assert_eq!(result.unwrap(), Receipt::Bootstrap);
        } else {
            assert!(result.is_err(), "{status}");
        }
        assert!(!output.exists());
    }
}
#[cfg(unix)]
#[test]
fn status_404_from_failed_transport_is_not_bootstrap() {
    let parent = tempfile::tempdir().unwrap();
    let hf = executable(parent.path(), "hf-failure", "exit 1");
    let curl = executable(parent.path(), "curl-failure", "printf 404; exit 28");
    assert!(
        history_fetch::fetch(
            "owner/dataset",
            &parent.path().join("history"),
            &hf,
            &curl,
            Duration::from_secs(2),
            &process::Cancellation::default()
        )
        .is_err()
    );
}
#[cfg(unix)]
#[test]
fn download_supervision_timeout_cannot_be_hidden_by_later_lookup() {
    let parent = tempfile::tempdir().unwrap();
    let hf = executable(parent.path(), "hf-stall", "sleep 2");
    let curl = executable(parent.path(), "curl-404", "printf 404");
    let result = history_fetch::fetch(
        "owner/dataset",
        &parent.path().join("history"),
        &hf,
        &curl,
        Duration::from_millis(30),
        &process::Cancellation::default(),
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("supervision failed")
    );
}
#[cfg(unix)]
#[test]
fn actual_local_http_lookup_preserves_404_auth_server_error_and_transport_failure() {
    use std::io::{Read, Write};
    let parent = tempfile::tempdir().unwrap();
    let state = state(parent.path());
    for status in [200, 401, 403, 404, 500] {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::scope(|scope| {
            scope.spawn(move || {
                let deadline=std::time::Instant::now()+Duration::from_secs(2);
                let (mut socket,_)=loop { match listener.accept() { Ok(connection)=>break connection,Err(error) if error.kind()==std::io::ErrorKind::WouldBlock=> { assert!(std::time::Instant::now()<deadline); std::thread::sleep(Duration::from_millis(5)); },Err(error)=>panic!("{error}") } };
                socket.set_nonblocking(false).unwrap(); socket.set_read_timeout(Some(Duration::from_secs(1))).unwrap(); let mut bytes=[0;1024]; assert!(socket.read(&mut bytes).unwrap()>0); socket.write_all(format!("HTTP/1.1 {status} Fixture\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}").as_bytes()).unwrap();
            });
            let actual = history_hub::lookup(
                Path::new("/usr/bin/curl"),
                &format!("http://{address}/api/datasets/owner/name"),
                &state,
                &process::Cancellation::default(),
            )
            .unwrap();
            assert_eq!(actual, status);
        });
    }
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    assert!(
        history_hub::lookup(
            Path::new("/usr/bin/curl"),
            &format!("http://{address}/"),
            &state,
            &process::Cancellation::default()
        )
        .is_err()
    );
    state.finish(Ok::<_, String>(())).unwrap();
}
#[cfg(unix)]
#[test]
fn upload_source_and_destination_are_fixed_and_token_is_environment_only() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("output");
    std::fs::create_dir_all(output.join("summary")).unwrap();
    let sha = "a".repeat(40);
    std::fs::write(
        output.join("summary/history.jsonl"),
        format!("{{\"schema_version\":3,\"complete\":true,\"source_sha\":\"{sha}\"}}\n"),
    )
    .unwrap();
    std::fs::write(output.join("card.md"), "# fixture card\n").unwrap();
    let hf = executable(
        parent.path(),
        "hf-upload",
        "test \"$1\" = upload; test \"$2\" = owner/dataset; test \"$5\" = --repo-type; test \"$6\" = dataset; test \"$7\" = --commit-message; test \"$HF_TOKEN\" = fixture-publication-token; test \"$HF_HUB_DISABLE_IMPLICIT_TOKEN\" = 0; case \"$4\" in data/runs/2026-10-02/123.jsonl) test \"$8\" = 'run 2026-10-02 123 @ aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';; card.md) test \"$8\" = 'card refresh 2026-10-02';; *) exit 9;; esac; case \"$*\" in *fixture-publication-token*) exit 8;; esac",
    );
    for kind in [Kind::Shard, Kind::Card] {
        let input = Input {
            repo: "owner/dataset",
            output: &output,
            kind,
            date: "2026-10-02",
            run_id: if matches!(kind, Kind::Shard) {
                Some("123")
            } else {
                None
            },
            source_sha: if matches!(kind, Kind::Shard) {
                Some(&sha)
            } else {
                None
            },
        };
        history_upload::upload(
            &input,
            &hf,
            "fixture-publication-token".into(),
            Duration::from_secs(2),
            &process::Cancellation::default(),
        )
        .unwrap();
    }
}
#[cfg(unix)]
#[test]
fn secret_process_output_is_redacted_even_when_upload_fails() {
    let parent = tempfile::tempdir().unwrap();
    let state = state(parent.path());
    let hf = executable(
        parent.path(),
        "hf-leak",
        "printf '%s' \"$HF_TOKEN\"; printf '%s' \"$HF_TOKEN\" >&2; exit 1",
    );
    let report = history_hub::execute(
        &hf,
        vec!["upload".into()],
        &state,
        Some("fixture-publication-token".into()),
        Duration::from_secs(2),
        &process::Cancellation::default(),
    )
    .unwrap();
    assert!(!report.success());
    for bytes in [&report.stdout.bytes_retained, &report.stderr.bytes_retained] {
        assert!(!String::from_utf8_lossy(bytes).contains("fixture-publication-token"));
    }
    state.finish(Ok::<_, String>(())).unwrap();
}
#[cfg(unix)]
#[test]
fn invalid_shard_source_identity_or_link_fails_before_upload() {
    use std::os::unix::fs::symlink;
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("output");
    std::fs::create_dir_all(output.join("summary")).unwrap();
    let sha = "a".repeat(40);
    let input = Input {
        repo: "owner/dataset",
        output: &output,
        kind: Kind::Shard,
        date: "2026-10-02",
        run_id: Some("123"),
        source_sha: Some(&sha),
    };
    std::fs::write(
        output.join("summary/history.jsonl"),
        "{\"schema_version\":3,\"complete\":false,\"source_sha\":\"other\"}\n",
    )
    .unwrap();
    assert!(
        history_upload::upload(
            &input,
            Path::new("/bin/true"),
            "fixture-publication-token".into(),
            Duration::from_secs(1),
            &process::Cancellation::default()
        )
        .is_err()
    );
    std::fs::remove_file(output.join("summary/history.jsonl")).unwrap();
    std::fs::write(parent.path().join("secret"), "secret").unwrap();
    symlink(
        parent.path().join("secret"),
        output.join("summary/history.jsonl"),
    )
    .unwrap();
    assert!(
        history_upload::upload(
            &input,
            Path::new("/bin/true"),
            "fixture-publication-token".into(),
            Duration::from_secs(1),
            &process::Cancellation::default()
        )
        .is_err()
    );
}

#[cfg(unix)]
#[test]
fn fetch_preserves_existing_destination_and_rejects_downloaded_symlink() {
    use std::os::unix::fs::symlink;
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("existing");
    std::fs::create_dir(&output).unwrap();
    std::fs::write(output.join("sentinel"), "keep").unwrap();
    assert!(
        history_fetch::fetch(
            "owner/name",
            &output,
            Path::new("/bin/false"),
            Path::new("/bin/false"),
            Duration::from_secs(1),
            &process::Cancellation::default()
        )
        .is_err()
    );
    assert_eq!(
        std::fs::read_to_string(output.join("sentinel")).unwrap(),
        "keep"
    );
    let dangling = parent.path().join("dangling");
    symlink(parent.path().join("not-present"), &dangling).unwrap();
    assert!(
        history_fetch::fetch(
            "owner/name",
            &dangling,
            Path::new("/bin/false"),
            Path::new("/bin/false"),
            Duration::from_secs(1),
            &process::Cancellation::default()
        )
        .is_err()
    );
    let hf = executable(
        parent.path(),
        "hf-link",
        "mkdir -p \"$6/data\"; ln -s /etc/passwd \"$6/data/link\"",
    );
    let output = parent.path().join("history");
    assert!(
        history_fetch::fetch(
            "owner/name",
            &output,
            &hf,
            Path::new("/bin/false"),
            Duration::from_secs(1),
            &process::Cancellation::default()
        )
        .is_err()
    );
    assert!(!output.exists());
}
#[cfg(unix)]
#[test]
fn empty_token_invalid_date_and_failed_upload_never_report_success() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("output");
    std::fs::create_dir(&output).unwrap();
    std::fs::write(output.join("card.md"), "card").unwrap();
    let input = Input {
        repo: "owner/name",
        output: &output,
        kind: Kind::Card,
        date: "2026-10-02",
        run_id: None,
        source_sha: None,
    };
    assert!(
        history_upload::upload(
            &input,
            Path::new("/bin/true"),
            "".into(),
            Duration::from_secs(1),
            &process::Cancellation::default()
        )
        .is_err()
    );
    assert!(
        history_upload::upload(
            &input,
            Path::new("/bin/false"),
            "fixture-publication-token".into(),
            Duration::from_secs(1),
            &process::Cancellation::default()
        )
        .is_err()
    );
    let input = Input {
        date: "2026-02-30",
        ..input
    };
    assert!(
        history_upload::upload(
            &input,
            Path::new("/bin/true"),
            "fixture-publication-token".into(),
            Duration::from_secs(1),
            &process::Cancellation::default()
        )
        .is_err()
    );
}
