//! Actual source guard mutations and bounded execution of retained adapters only.
use crate::{
    authority_sources, cache_marker, support,
    workflow_yaml::{self, Node},
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, process::Command};
fn workflows() -> BTreeMap<String, Node> {
    fs::read_dir(support::root().join(".github/workflows"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "yml" || e == "yaml"))
        .map(|p| {
            (
                p.file_name().unwrap().to_str().unwrap().to_owned(),
                workflow_yaml::parse(&fs::read_to_string(p).unwrap()).unwrap(),
            )
        })
        .collect()
}
fn replace(node: &mut Node, path: &[&str], value: Node) {
    let Node::Map(entries) = node else {
        panic!("mapping")
    };
    let (_, child) = entries.iter_mut().find(|(k, _)| k == path[0]).unwrap();
    if path.len() == 1 {
        *child = value;
    } else {
        replace(child, &path[1..], value)
    }
}
fn alter_step(document: &mut Node, job: &str, name: &str, key: &str, value: Node) {
    let Node::Map(jobs) = document else { panic!() };
    let (_, jobs) = jobs.iter_mut().find(|(k, _)| k == "jobs").unwrap();
    let Node::Map(jobs) = jobs else { panic!() };
    let (_, job) = jobs.iter_mut().find(|(k, _)| k == job).unwrap();
    let Node::Map(fields) = job else { panic!() };
    let (_, steps) = fields.iter_mut().find(|(k, _)| k == "steps").unwrap();
    let Node::Seq(steps) = steps else { panic!() };
    let step = steps
        .iter_mut()
        .find(|s| s.get("name").and_then(Node::text) == Some(name))
        .unwrap();
    let Node::Map(fields) = step else { panic!() };
    if let Some((_, old)) = fields.iter_mut().find(|(k, _)| k == key) {
        *old = value;
    } else {
        fields.push((key.into(), value));
    }
}
fn body(workflow: &str, job: &str, name: &str) -> String {
    let documents = workflows();
    let Node::Seq(steps) = documents[workflow]
        .get("jobs")
        .unwrap()
        .get(job)
        .unwrap()
        .get("steps")
        .unwrap()
    else {
        panic!()
    };
    steps
        .iter()
        .find(|s| s.get("name").and_then(Node::text) == Some(name))
        .unwrap()
        .get("run")
        .and_then(Node::text)
        .unwrap()
        .to_owned()
}
#[test]
fn authority_marker_parsed_protocol_refuses_reordered_keys_hit_policy_and_bypass() {
    let actual = workflows();
    cache_marker::check(&actual).unwrap();
    for (workflow, job, name, key, value) in [
        (
            "ci-quality-slice.yml",
            "authority_sentinel",
            "Restore saved PR poison marker",
            "with",
            Node::Map(vec![("key".into(), Node::Scalar("wrong".into()))]),
        ),
        (
            "depot-canary.yml",
            "seed_authority_marker",
            "Save trusted authority marker",
            "uses",
            Node::Scalar("actions/cache/save@main".into()),
        ),
        (
            "depot-canary.yml",
            "verify_pr_write",
            "Require PR poison marker miss",
            "run",
            Node::Scalar("set -euo pipefail\n# require miss\ntrue".into()),
        ),
        (
            "ci-quality-slice.yml",
            "authority_sentinel",
            "Save PR poison marker",
            "if",
            Node::Scalar("false".into()),
        ),
        (
            "depot-canary.yml",
            "seed_authority_marker",
            "Require trusted marker hit",
            "env",
            Node::Map(vec![("CACHE_HIT".into(), Node::Scalar("true".into()))]),
        ),
    ] {
        let mut changed = actual.clone();
        alter_step(changed.get_mut(workflow).unwrap(), job, name, key, value);
        assert!(
            cache_marker::check(&changed).is_err(),
            "{workflow}:{name}:{key}"
        );
    }
    let mut changed = actual.clone();
    let Node::Map(root) = changed.get_mut("ci-quality-slice.yml").unwrap() else {
        panic!()
    };
    let Node::Map(jobs) = &mut root.iter_mut().find(|(k, _)| k == "jobs").unwrap().1 else {
        panic!()
    };
    let Node::Map(job) = &mut jobs
        .iter_mut()
        .find(|(k, _)| k == "authority_sentinel")
        .unwrap()
        .1
    else {
        panic!()
    };
    let Node::Seq(steps) = &mut job.iter_mut().find(|(k, _)| k == "steps").unwrap().1 else {
        panic!()
    };
    let save = steps
        .iter()
        .position(|s| s.get("name").and_then(Node::text) == Some("Save PR poison marker"))
        .unwrap();
    let clear = steps
        .iter()
        .position(|s| {
            s.get("name").and_then(Node::text)
                == Some("Clear local poison marker before proof restore")
        })
        .unwrap();
    steps.swap(save, clear);
    assert!(cache_marker::check(&changed).is_err());
}
#[test]
fn authority_delivery_and_reusable_sources_refuse_raw_runner_credentials_sdk_and_verifier_bypass() {
    let actual = workflows();
    authority_sources::check(&actual).unwrap();
    for (job, key, value) in [
        ("authority_automation", "runs-on", "depot-ubuntu-24.04"),
        ("seed_authority_marker", "needs", "untrusted"),
        ("verify_pr_write", "if", "true"),
    ] {
        let mut changed = actual.clone();
        replace(
            changed.get_mut("depot-canary.yml").unwrap(),
            &["jobs", job, key],
            Node::Scalar(value.into()),
        );
        assert!(authority_sources::check(&changed).is_err());
    }
    for name in [
        "Admit protected automation producer identity",
        "Verify protected authority automation without executing it",
        "Attest provider-injected cache backend",
    ] {
        let mut changed = actual.clone();
        alter_step(
            changed.get_mut("depot-canary.yml").unwrap(),
            "seed_authority_marker",
            name,
            "run",
            Node::Scalar("set -euo pipefail\ntrue".into()),
        );
        assert!(authority_sources::check(&changed).is_err(), "{name}");
    }
    let mut changed = actual.clone();
    replace(
        changed.get_mut("ci-linux-sdk-slice.yml").unwrap(),
        &["jobs", "rust_smoke", "if"],
        Node::Scalar("${{ contains(inputs.sdk_matrix, 'rust') }}".into()),
    );
    assert!(authority_sources::check(&changed).is_err());
    let mut changed = actual.clone();
    replace(
        changed.get_mut("ci-linux-sdk-slice.yml").unwrap(),
        &["on", "workflow_call", "inputs"],
        Node::Map(vec![(
            "runs_on".into(),
            Node::Map(vec![("type".into(), Node::Scalar("string".into()))]),
        )]),
    );
    assert!(authority_sources::check(&changed).is_err());
}
fn command(root: &Path, script: &str) -> Command {
    let mut c = Command::new("/bin/bash");
    c.env_clear().current_dir(root).args(["-c", script]);
    c.env("PATH", "/usr/bin:/bin")
        .env("TMPDIR", root)
        .env("RUNNER_TEMP", root)
        .env("RUNNER_OS", "Linux")
        .env("RUNNER_ARCH", "X64");
    c
}
fn run(root: &Path, script: &str, fields: &[(&str, &str)]) -> std::process::Output {
    let mut c = command(root, script);
    for (k, v) in fields {
        c.env(k, v);
    }
    support::Fixture::new().run(c)
}
#[test]
fn authority_actual_marker_bodies_preserve_exact_bytes_fail_closed_and_clear_owned_local_copy() {
    let id = "0123456789abcdef0123456789abcdef";
    let trusted = format!("mesh-llm-depot-authority-marker-v1\nsentinel_id={id}\n");
    let poison = format!("mesh-llm-depot-authority-pr-marker-v1\nsentinel_id={id}\npr_number=42\n");
    for (workflow, job, writer, validator, expected, hit) in [
        (
            "depot-canary.yml",
            "seed_authority_marker",
            "Prepare deterministic non-secret marker",
            "Validate trusted marker content",
            trusted.as_str(),
            "true",
        ),
        (
            "ci-quality-slice.yml",
            "authority_sentinel",
            "Replace with deterministic PR poison marker",
            "Validate saved PR poison marker content",
            poison.as_str(),
            "true",
        ),
    ] {
        let root = tempfile::tempdir().unwrap();
        let fields = [("SENTINEL_ID", id), ("PR_NUMBER", "42"), ("CACHE_HIT", hit)];
        assert!(
            run(root.path(), &body(workflow, job, writer), &fields)
                .status
                .success()
        );
        let path = root.path().join(".depot-authority-sentinel/marker");
        assert_eq!(fs::read_to_string(&path).unwrap(), expected);
        let script = body(workflow, job, validator);
        assert!(run(root.path(), &script, &fields).status.success());
        fs::write(&path, b"wrong\n").unwrap();
        assert!(!run(root.path(), &script, &fields).status.success());
        fs::remove_file(&path).unwrap();
        assert!(!run(root.path(), &script, &fields).status.success());
        if job == "authority_sentinel" {
            let mut miss = fields;
            miss[2].1 = "false";
            assert!(!run(root.path(), &script, &miss).status.success());
        }
        root.close().unwrap();
    }
    let root = tempfile::tempdir().unwrap();
    let marker = root.path().join(".depot-authority-sentinel");
    fs::create_dir(&marker).unwrap();
    fs::write(marker.join("marker"), &poison).unwrap();
    let clear = body(
        "depot-canary.yml",
        "verify_pr_write",
        "Clear local poison marker before trusted restore",
    );
    assert!(run(root.path(), &clear, &[]).status.success());
    assert!(!marker.exists());
    for (workflow, job, name, accepted_hit) in [
        (
            "ci-quality-slice.yml",
            "authority_sentinel",
            "Require trusted seed isolation after poison publication",
            false,
        ),
        (
            "depot-canary.yml",
            "verify_pr_write",
            "Require PR poison marker miss",
            false,
        ),
        (
            "depot-canary.yml",
            "seed_authority_marker",
            "Require trusted marker hit",
            true,
        ),
    ] {
        let script = body(workflow, job, name);
        for hit in ["true", "false"] {
            assert_eq!(
                run(root.path(), &script, &[("CACHE_HIT", hit)])
                    .status
                    .success(),
                (hit == "true") == accepted_hit,
                "{name}/{hit}"
            );
        }
    }
    root.close().unwrap();
}
fn bundle(root: &Path, source: &str, bytes: &[u8]) -> String {
    let stage = root.join("immutable-automation-restored");
    fs::write(stage.join("xtask"), bytes).unwrap();
    let source = format!("{source}\n");
    fs::write(stage.join("source.txt"), source.as_bytes()).unwrap();
    let digest = hex::encode(Sha256::digest(bytes));
    let source_digest = hex::encode(Sha256::digest(source.as_bytes()));
    fs::write(
        stage.join("SHA256SUMS"),
        format!("{digest}  xtask\n{source_digest}  source.txt\n"),
    )
    .unwrap();
    digest
}
#[test]
fn authority_actual_manual_delivery_verifies_supplied_identity_before_native_endpoint_calls() {
    let source = "0123456789012345678901234567890123456789";
    for job in ["seed_authority_marker", "verify_pr_write"] {
        for mode in [
            "valid",
            "source-mismatch",
            "digest-mismatch",
            "symlink",
            "extra-member",
            "endpoint-loopback",
            "second-endpoint-loopback",
        ] {
            let root = tempfile::tempdir().unwrap();
            let binary = fs::read(env!("CARGO_BIN_EXE_xtask")).unwrap();
            let hash = hex::encode(Sha256::digest(&binary));
            let env_path = root.path().join("env");
            let env_text = env_path.to_str().unwrap();
            let fields = [
                ("EXPECTED_SOURCE_SHA", source),
                (
                    "AUTOMATION_RESULTS_ORIGIN",
                    "https://results-receiver.actions.githubusercontent.com",
                ),
                ("AUTOMATION_SOURCE_SHA", source),
                ("AUTOMATION_ARTIFACT_ID", "1234"),
                ("AUTOMATION_BINARY_SHA256", hash.as_str()),
                ("GITHUB_ENV", env_text),
            ];
            let admit = body(
                "depot-canary.yml",
                job,
                "Admit protected automation producer identity",
            );
            let mut wrong = fields;
            wrong[0].1 = "a123456789012345678901234567890123456789";
            assert!(!run(root.path(), &admit, &wrong).status.success());
            assert!(!root.path().join("immutable-automation-restored").exists());
            assert!(run(root.path(), &admit, &fields).status.success());
            let content = if mode == "source-mismatch" {
                "a123456789012345678901234567890123456789"
            } else {
                source
            };
            bundle(root.path(), content, &binary);
            let stage = root.path().join("immutable-automation-restored");
            if mode == "digest-mismatch" {
                fs::write(stage.join("xtask"), b"#!/bin/bash\n: > executed\n").unwrap();
            }
            if mode == "symlink" {
                fs::remove_file(stage.join("xtask")).unwrap();
                std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_xtask"), stage.join("xtask"))
                    .unwrap();
            }
            if mode == "extra-member" {
                fs::write(stage.join("unexpected"), b"extra").unwrap();
            }
            let verify = body(
                "depot-canary.yml",
                job,
                "Verify protected authority automation without executing it",
            );
            let endpoints = body(
                "depot-canary.yml",
                job,
                "Attest provider-injected cache backend",
            );
            let script =
                format!("{verify}\nexport MESH_LLM_AUTOMATION_BIN=\"$PWD/xtask\"\n{endpoints}");
            let cache = if mode == "endpoint-loopback" {
                "http://[::ffff:127.0.0.1]:8080/private-fixture?secret=redaction"
            } else {
                "http://cache.fixture:8080/path"
            };
            let results = if mode == "second-endpoint-loopback" {
                "http://[::1]:8080/private-fixture?secret=redaction"
            } else {
                "http://results.fixture:8080/path"
            };
            let mut c = command(root.path(), &script);
            for (k, v) in fields {
                c.env(k, v);
            }
            c.env("ACTIONS_CACHE_URL", cache)
                .env("ACTIONS_RESULTS_URL", results);
            let output = support::Fixture::new().run(c);
            assert_eq!(
                output.status.success(),
                mode == "valid",
                "{job}/{mode}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(!stage.join("executed").exists());
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !stderr.contains(cache)
                    && !stderr.contains(results)
                    && !stderr.contains("private-fixture")
                    && !stderr.contains("redaction")
            );
            if mode == "valid" {
                assert_eq!(
                    fs::read_to_string(&env_path).unwrap(),
                    format!(
                        "MESH_LLM_AUTOMATION_BIN={}\n",
                        stage.join("xtask").display()
                    )
                );
            }
            root.close().unwrap();
        }
    }
}
#[test]
fn authority_documented_allowlist_and_future_executor_keep_local_least_privilege() {
    let source = fs::read_to_string(support::root().join("ci/DEPOT_MIGRATION.md")).unwrap();
    authority_sources::documentation(&source).unwrap();
    for changed in [
        source.replace(
            "depot-canary.yml@refs/heads/main",
            "smoke.yml@refs/heads/main",
        ),
        source.replace("`permissions: contents: read`", "`permissions: write-all`"),
        source.replace(
            "`persist-credentials: false`",
            "`persist-credentials: true`",
        ),
    ] {
        assert!(authority_sources::documentation(&changed).is_err());
    }
    let mut credentials = workflows();
    let raw = fs::read_to_string(support::root().join(".github/workflows/sdk-smoke.yml")).unwrap();
    credentials.insert(
        "sdk-smoke.yml".into(),
        workflow_yaml::parse(
            &raw.replace("persist-credentials: false", "persist-credentials: true"),
        )
        .unwrap(),
    );
    assert!(authority_sources::check(&credentials).is_err());
    let mut changed = workflows();
    replace(
        changed.get_mut("sdk-smoke.yml").unwrap(),
        &["jobs", "sdk_smoke", "runs-on"],
        Node::Scalar("depot-ubuntu-24.04".into()),
    );
    assert!(authority_sources::check(&changed).is_err());
}

#[test]
fn authority_trusted_hosted_origin_emits_only_canonical_public_value_and_refuses_unbound_inputs() {
    const ORIGIN: &str = "https://results-receiver.actions.githubusercontent.com";
    let source = "0123456789012345678901234567890123456789";
    let script = body(
        "depot-canary.yml",
        "authority_automation",
        "Admit trusted hosted artifact service origin",
    );
    for (origin, expected) in [
        (ORIGIN, true),
        (
            "https://results-receiver.actions.githubusercontent.com/",
            true,
        ),
        ("", false),
        (
            "http://results-receiver.actions.githubusercontent.com",
            false,
        ),
        (
            "https://results-receiver.actions.githubusercontent.com:443",
            false,
        ),
        (
            "https://results-receiver.actions.githubusercontent.com/path",
            false,
        ),
        (
            "https://results-receiver.actions.githubusercontent.com?secret=fixture-token",
            false,
        ),
        (
            "https://results-receiver.actions.githubusercontent.com//",
            false,
        ),
        (
            "https://user:fixture-token@results-receiver.actions.githubusercontent.com",
            false,
        ),
        ("http://provider.fixture:8080/path", false),
        (
            "https://results-receiver.actions.githubusercontent.com.evil",
            false,
        ),
        (
            "https://RESULTS-RECEIVER.actions.githubusercontent.com",
            false,
        ),
        (
            "https://results-receiver.actions.githubusercontent.com\n",
            false,
        ),
    ] {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("output");
        let fields = [
            ("EXPECTED_SOURCE_SHA", source),
            ("AUTOMATION_SOURCE_SHA", source),
            ("ACTIONS_RESULTS_URL", origin),
            ("ACTIONS_RUNTIME_TOKEN", "fixture-runtime-token"),
            ("GITHUB_OUTPUT", path.to_str().unwrap()),
        ];
        let output = run(root.path(), &script, &fields);
        assert_eq!(output.status.success(), expected);
        assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture-token"));
        assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture-runtime-token"));
        if expected {
            assert_eq!(
                fs::read_to_string(path).unwrap(),
                format!("origin={ORIGIN}\n")
            );
        } else {
            assert!(!path.exists());
        }
        root.close().unwrap();
    }
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("output");
    let output = run(
        root.path(),
        &script,
        &[
            (
                "EXPECTED_SOURCE_SHA",
                "a123456789012345678901234567890123456789",
            ),
            ("AUTOMATION_SOURCE_SHA", source),
            ("ACTIONS_RESULTS_URL", ORIGIN),
            ("GITHUB_OUTPUT", path.to_str().unwrap()),
        ],
    );
    assert!(!output.status.success());
    assert!(!path.exists());
    root.close().unwrap();
    for job in ["seed_authority_marker", "verify_pr_write"] {
        let root = tempfile::tempdir().unwrap();
        let output = run(
            root.path(),
            &body(
                "depot-canary.yml",
                job,
                "Admit protected automation producer identity",
            ),
            &[
                ("EXPECTED_SOURCE_SHA", source),
                ("AUTOMATION_SOURCE_SHA", source),
                ("AUTOMATION_ARTIFACT_ID", "1234"),
                ("AUTOMATION_BINARY_SHA256", &"a".repeat(64)),
                (
                    "AUTOMATION_RESULTS_ORIGIN",
                    "http://provider.fixture:8080/path",
                ),
            ],
        );
        assert!(!output.status.success());
        assert!(!root.path().join("immutable-automation-restored").exists());
        root.close().unwrap();
    }
}
#[test]
fn authority_artifact_download_origin_override_is_step_local_and_producer_bound() {
    let documents = workflows();
    let original = documents["depot-canary.yml"].clone();
    authority_sources::manual(&original).unwrap();
    for job in ["seed_authority_marker", "verify_pr_write"] {
        let mut altered = original.clone();
        alter_step(
            &mut altered,
            job,
            "Download protected authority automation",
            "env",
            Node::Map(vec![(
                "ACTIONS_RESULTS_URL".into(),
                Node::Scalar("http://provider.fixture:8080/path".into()),
            )]),
        );
        assert!(authority_sources::manual(&altered).is_err());
        let mut altered = original.clone();
        alter_step(
            &mut altered,
            job,
            "Admit protected automation producer identity",
            "env",
            Node::Map(vec![]),
        );
        assert!(authority_sources::manual(&altered).is_err());
        let mut altered = original.clone();
        alter_step(
            &mut altered,
            job,
            "Attest provider-injected cache backend",
            "env",
            Node::Map(vec![(
                "ACTIONS_RESULTS_URL".into(),
                Node::Scalar("https://results-receiver.actions.githubusercontent.com".into()),
            )]),
        );
        assert!(authority_sources::manual(&altered).is_err());
    }
    let mut altered = original.clone();
    alter_step(
        &mut altered,
        "authority_automation",
        "Admit trusted hosted artifact service origin",
        "run",
        Node::Scalar("printf 'origin=%s\\n' \"$ACTIONS_RESULTS_URL\" >> \"$GITHUB_OUTPUT\"".into()),
    );
    assert!(authority_sources::manual(&altered).is_err());
}
#[test]
fn authority_quality_hosted_origin_binds_protected_checkout_receipt_and_refuses_provider_urls() {
    const ORIGIN: &str = "https://results-receiver.actions.githubusercontent.com";
    let sha = "0123456789012345678901234567890123456789";
    let script = body(
        "ci-quality-slice.yml",
        "runner_policy",
        "Admit protected hosted artifact service origin",
    );
    for (mode, url, source) in [
        ("canonical", ORIGIN, sha),
        (
            "slash",
            "https://results-receiver.actions.githubusercontent.com/",
            sha,
        ),
        (
            "provider",
            "http://provider.invalid:1234/private-fixture-token",
            sha,
        ),
        (
            "userinfo",
            "https://user:private-fixture-token@results-receiver.actions.githubusercontent.com",
            sha,
        ),
        (
            "query",
            "https://results-receiver.actions.githubusercontent.com?private-fixture-token",
            sha,
        ),
        (
            "mismatch",
            ORIGIN,
            "a123456789012345678901234567890123456789",
        ),
        ("missing", ORIGIN, ""),
    ] {
        let f = support::Fixture::new();
        f.executable("git",r#"[[ $* == 'rev-parse HEAD' && $GIT_MASTER == 1 && $GIT_OPTIONAL_LOCKS == 0 ]] || exit 41; printf '%s\n' "$FIXTURE_HEAD""#);
        let path = f.path().join("output");
        let mut c = Command::new("/bin/bash");
        c.env_clear()
            .current_dir(f.path())
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", f.path().join("bin").display()),
            )
            .env("GIT_MASTER", "1")
            .env("GIT_OPTIONAL_LOCKS", "0")
            .env("FIXTURE_HEAD", sha)
            .env("AUTOMATION_SOURCE_SHA", source)
            .env("ACTIONS_RESULTS_URL", url)
            .env("ACTIONS_RUNTIME_TOKEN", "private-fixture-token")
            .env("GITHUB_OUTPUT", &path)
            .args(["-c", &script]);
        let output = f.run(c);
        let valid = ["canonical", "slash"].contains(&mode);
        assert_eq!(output.status.success(), valid);
        assert!(output.stdout.is_empty());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("private-fixture-token"));
        if valid {
            assert_eq!(
                fs::read_to_string(path).unwrap(),
                format!("origin={ORIGIN}\n")
            )
        } else {
            assert!(!path.exists())
        }
        f.0.close().unwrap();
    }
    for origin in [ORIGIN, "http://provider.invalid:1234/path", ""] {
        let root = tempfile::tempdir().unwrap();
        let script = body(
            "ci-quality-slice.yml",
            "authority_sentinel",
            "Admit protected automation producer identity",
        );
        let output = run(
            root.path(),
            &script,
            &[
                ("AUTOMATION_SOURCE_SHA", sha),
                ("AUTOMATION_ARTIFACT_ID", "1234"),
                ("AUTOMATION_BINARY_SHA256", &"a".repeat(64)),
                ("AUTOMATION_RESULTS_ORIGIN", origin),
            ],
        );
        let admitted = origin == ORIGIN;
        assert_eq!(output.status.success(), admitted);
        assert_eq!(
            root.path().join("immutable-automation-restored").exists(),
            admitted
        );
        root.close().unwrap();
    }
}
#[test]
fn authority_quality_download_origin_scope_and_producer_custody_cannot_be_bypassed() {
    let documents = workflows();
    let original = documents["ci-quality-slice.yml"].clone();
    let get =
        |document: &Node, name: &str| document.get("jobs").unwrap().get(name).unwrap().clone();
    crate::cache_authority::producer(&get(&original, "runner_policy")).unwrap();
    crate::cache_authority::sentinel(&get(&original, "authority_sentinel")).unwrap();
    for (job, name, key, value) in [
        (
            "runner_policy",
            "Admit protected hosted artifact service origin",
            "run",
            Node::Scalar(
                "printf 'origin=%s\\n' \"$ACTIONS_RESULTS_URL\" >> \"$GITHUB_OUTPUT\"".into(),
            ),
        ),
        (
            "authority_sentinel",
            "Download protected authority automation",
            "env",
            Node::Map(vec![(
                "ACTIONS_RESULTS_URL".into(),
                Node::Scalar("http://provider.invalid:1234/path".into()),
            )]),
        ),
        (
            "authority_sentinel",
            "Attest provider-injected cache backend",
            "env",
            Node::Map(vec![(
                "ACTIONS_RESULTS_URL".into(),
                Node::Scalar("https://results-receiver.actions.githubusercontent.com".into()),
            )]),
        ),
    ] {
        let mut altered = original.clone();
        alter_step(&mut altered, job, name, key, value);
        let target = get(&altered, job);
        assert!(
            if job == "runner_policy" {
                crate::cache_authority::producer(&target)
            } else {
                crate::cache_authority::sentinel(&target)
            }
            .is_err()
        );
    }
}
