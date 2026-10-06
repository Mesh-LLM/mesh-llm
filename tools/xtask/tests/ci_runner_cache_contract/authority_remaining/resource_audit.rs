//! Actual audit run body with inert machine probes; no hosted measurements or credentials.
use super::*;
use std::process::Command;
const LABELS: &[&str] = &[
    "depot-ubuntu-24.04",
    "depot-ubuntu-24.04-4",
    "depot-ubuntu-24.04-8",
    "depot-ubuntu-24.04-16",
    "depot-ubuntu-24.04-arm",
    "depot-ubuntu-24.04-arm-8",
];
fn prepare() -> support::Fixture {
    let f = support::Fixture::new();
    fs::create_dir(f.path().join("docker")).unwrap();
    f.executable(
        "uname",
        r#"[[ $* == '-m' ]] || exit 41; printf '%s\n' "$FIXTURE_ARCH""#,
    );
    f.executable("nproc", r#"[[ $# == 0 ]] || exit 41; printf '8\n'"#);
    f.executable("df",r#"[[ $* == "-Pk $GITHUB_WORKSPACE" ]] || exit 41; printf 'header\nfilesystem 1 2 4096 5 workspace\n'"#);
    f.executable("awk",r#"case "$1" in
'/MemTotal/ { print $2 }') [[ $2 == /proc/meminfo ]] || exit 41; printf '1024\n';;
'NR == 2 { print $4 }') IFS= read -r header; IFS= read -r row; [[ $header == header && $row == 'filesystem 1 2 4096 5 workspace' ]] || exit 41; printf '4096\n';;
*) exit 41;; esac"#);
    f
}
fn command(f: &support::Fixture, label: &str, extra: &[(&str, &str)]) -> Command {
    let script = body(
        "depot-canary.yml",
        "runner",
        "Verify ephemeral runner resources",
    );
    assert_eq!(script.matches("${{ matrix.runner }}").count(), 1);
    let script = script.replace("${{ matrix.runner }}", label); // Sole source-owned matrix display projection.
    let mut c = Command::new("/bin/bash");
    c.env_clear().current_dir(f.path()).args(["-c", &script]);
    c.env(
        "PATH",
        format!("{}:/usr/bin:/bin", f.path().join("bin").display()),
    )
    .env("HOME", f.path())
    .env("DOCKER_CONFIG", f.path().join("docker"))
    .env("GITHUB_WORKSPACE", f.path())
    .env("GITHUB_STEP_SUMMARY", f.path().join("summary"))
    .env("CANARY_RUNNER_LABEL", label)
    .env(
        "FIXTURE_ARCH",
        if label.contains("-arm") {
            "aarch64"
        } else {
            "x86_64"
        },
    )
    .env("ImageOS", "inert-linux")
    .env("ImageVersion", "inert-v1");
    for (k, v) in extra {
        c.env(k, v);
    }
    c
}
#[test]
fn authority_actual_resource_audit_admits_declared_architectures_and_requires_image_identity() {
    let documents = workflows();
    let actual = job(&documents, "depot-canary.yml", "runner")
        .get("strategy")
        .unwrap()
        .get("matrix")
        .unwrap()
        .get("runner")
        .unwrap();
    let Node::Seq(labels) = actual else {
        panic!("runner matrix")
    };
    assert_eq!(
        labels.iter().map(|n| n.text().unwrap()).collect::<Vec<_>>(),
        LABELS
    );
    for label in LABELS {
        let f = prepare();
        let output = f.run(command(&f, label, &[]));
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let summary = fs::read_to_string(f.path().join("summary")).unwrap();
        assert!(summary.contains(&format!("### {label}")));
        assert!(summary.contains("| logical CPUs | 8 |"));
        assert!(summary.contains("| memory KiB | 1024 |"));
        assert!(summary.contains("| workspace free KiB | 4096 |"));
        assert!(summary.contains("| Depot cache credentials/endpoints injected | no |"));
        f.0.close().unwrap();
    }
    for fields in [
        vec![("FIXTURE_ARCH", "wrong")],
        vec![("ImageOS", "")],
        vec![("ImageVersion", "")],
    ] {
        let f = prepare();
        let output = f.run(command(&f, LABELS[0], &fields));
        assert!(!output.status.success());
        assert!(!f.path().join("summary").exists());
        f.0.close().unwrap();
    }
}
#[test]
fn authority_actual_resource_audit_refuses_each_injected_setting_without_printing_values() {
    for name in [
        "DEPOT_CACHE_TOKEN",
        "DEPOT_CACHE_URL",
        "DEPOT_CACHE_API_URL",
        "DEPOT_TOKEN",
        "DEPOT_REGISTRY_HOST",
        "DEPOT_REGISTRY_URL",
        "DEPOT_REGISTRY_PULL_TOKEN",
        "DEPOT_REGISTRY_TOKEN",
        "SCCACHE_WEBDAV_ENDPOINT",
        "SCCACHE_WEBDAV_TOKEN",
        "SCCACHE_WEBDAV_USERNAME",
        "SCCACHE_WEBDAV_PASSWORD",
        "SCCACHE_BUCKET",
        "SCCACHE_ENDPOINT",
        "TURBO_TOKEN",
        "TURBO_API",
        "TURBO_TEAM",
        "GOCACHEPROG",
        "REGISTRY_TOKEN",
        "REGISTRY_USERNAME",
        "REGISTRY_PASSWORD",
        "REGISTRY_AUTH_TOKEN",
        "NPM_TOKEN",
        "NODE_AUTH_TOKEN",
        "CARGO_REGISTRIES_CRATES_IO_TOKEN",
    ] {
        let f = prepare();
        let output = f.run(command(
            &f,
            LABELS[0],
            &[(name, "private-canary-fixture-token")],
        ));
        assert!(!output.status.success(), "{name}");
        assert!(String::from_utf8_lossy(&output.stderr).contains(name));
        assert!(!String::from_utf8_lossy(&output.stderr).contains("private-canary-fixture-token"));
        assert!(!f.path().join("summary").exists());
        f.0.close().unwrap();
    }
}
#[test]
fn authority_actual_resource_audit_endpoint_and_docker_refusal_remains_value_free() {
    for endpoint in [
        "https://results-receiver.actions.githubusercontent.com/fixture",
        "http://localhost:8080/fixture",
        "https://127.0.0.1:8080/fixture",
        "http://[::1]:8080/fixture",
    ] {
        let f = prepare();
        let output = f.run(command(
            &f,
            LABELS[0],
            &[
                ("ACTIONS_CACHE_URL", endpoint),
                ("ACTIONS_RESULTS_URL", endpoint),
                ("ACTIONS_RUNTIME_URL", endpoint),
            ],
        ));
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        f.0.close().unwrap();
    }
    for endpoint in [
        "http://remote.invalid:8080/secret-fixture",
        "https://actions.githubusercontent.com.evil/secret-fixture",
        "https://user:private-token@actions.githubusercontent.com/secret-fixture",
        "http://localhost/secret-fixture",
        "http://localhost:0/secret-fixture",
        "http://[::1]:65536/secret-fixture",
        "http://127.0.0.1:8080",
    ] {
        let f = prepare();
        let output = f.run(command(&f, LABELS[0], &[("ACTIONS_RESULTS_URL", endpoint)]));
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("variable=ACTIONS_RESULTS_URL"));
        assert!(
            !stderr.contains(endpoint)
                && !stderr.contains("secret-fixture")
                && !stderr.contains("private-token")
        );
        assert!(!f.path().join("summary").exists());
        f.0.close().unwrap();
    }
    for payload in [
        r#"{"auths":{}}"#,
        r#"{"credHelpers":{}}"#,
        r#"{"credsStore":"private-token"}"#,
    ] {
        let f = prepare();
        fs::write(f.path().join("docker/config.json"), payload).unwrap();
        let output = f.run(command(&f, LABELS[0], &[]));
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stderr).contains(payload));
        assert!(!f.path().join("summary").exists());
        f.0.close().unwrap();
    }
    let f = prepare();
    let output = f.run(command(
        &f,
        LABELS[0],
        &[("DOCKER_AUTH_CONFIG", "private-token")],
    ));
    assert!(!output.status.success());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("private-token"));
    assert!(!f.path().join("summary").exists());
    f.0.close().unwrap();
}
#[test]
fn authority_actual_trusted_image_body_checks_architecture_before_native_image_verifier() {
    let documents = workflows();
    let target = job(
        &documents,
        "ci-runner-contract-slice.yml",
        "trusted_runner_image",
    );
    assert_eq!(
        target.get("if").and_then(Node::text),
        Some("${{ inputs.profile == 'main' || inputs.profile == 'manual-full' }}")
    );
    let rows = target
        .get("strategy")
        .unwrap()
        .get("matrix")
        .unwrap()
        .get("include")
        .unwrap();
    let Node::Seq(rows) = rows else {
        panic!("image matrix")
    };
    assert_eq!(rows.len(), 2);
    for (row, (runner, machine)) in rows
        .iter()
        .zip([("mesh-llm-amd64", "x86_64"), ("mesh-llm-arm64", "aarch64")])
    {
        assert_eq!(row.get("runner").and_then(Node::text), Some(runner));
        assert_eq!(row.get("machine").and_then(Node::text), Some(machine));
        for (mode, actual, admitted) in [
            ("valid", machine, true),
            ("wrong-machine", "wrong", false),
            ("image-refusal", machine, false),
            ("compile-refusal", machine, false),
        ] {
            let f = prepare();
            f.executable("verify-runner-image",r#"[[ $* == self-hosted ]] || exit 41; [[ $FIXTURE_IMAGE_PASS == true ]] || exit 42; printf 'verified\n' > "$FIXTURE_IMAGE_RECEIPT""#);
            f.executable("cargo", r#"[[ $* == 'check --locked -p mesh-llm-config' ]] || exit 41; [[ -f $FIXTURE_IMAGE_RECEIPT ]] || exit 42; [[ $FIXTURE_COMPILE_PASS == true ]] || exit 43; printf 'checked\n' > "$FIXTURE_COMPILE_RECEIPT""#);
            let source = body(
                "ci-runner-contract-slice.yml",
                "trusted_runner_image",
                "Verify trusted runner image",
            );
            let mut c = Command::new("/bin/bash");
            c.env_clear()
                .current_dir(f.path())
                .env(
                    "PATH",
                    format!("{}:/usr/bin:/bin", f.path().join("bin").display()),
                )
                .env("EXPECTED_MACHINE", machine)
                .env("FIXTURE_ARCH", actual)
                .env(
                    "FIXTURE_IMAGE_PASS",
                    if mode == "image-refusal" {
                        "false"
                    } else {
                        "true"
                    },
                )
                .env("FIXTURE_IMAGE_RECEIPT", f.path().join("image-receipt"))
                .env(
                    "FIXTURE_COMPILE_PASS",
                    if mode == "compile-refusal" {
                        "false"
                    } else {
                        "true"
                    },
                )
                .env("FIXTURE_COMPILE_RECEIPT", f.path().join("compile-receipt"))
                .args(["-c", &source]);
            let output = f.run(c);
            assert_eq!(
                output.status.success(),
                admitted,
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(
                f.path().join("image-receipt").exists(),
                admitted || mode == "compile-refusal"
            );
            assert_eq!(f.path().join("compile-receipt").exists(), admitted);
            f.0.close().unwrap();
        }
    }
}

#[test]
fn authority_actual_resource_diagnostics_preserve_all_variable_classification_without_values() {
    for name in [
        "ACTIONS_CACHE_URL",
        "ACTIONS_RESULTS_URL",
        "ACTIONS_RUNTIME_URL",
    ] {
        for (endpoint, scheme, authority, port, path) in [
            (
                "https://cache.example.invalid/cache",
                "https",
                "other",
                "absent",
                "present",
            ),
            (
                "https://cache.depot.dev/cache",
                "https",
                "other",
                "absent",
                "present",
            ),
            (
                "https://user@attacker.example/cache",
                "https",
                "other",
                "absent",
                "present",
            ),
            (
                "https://actions.githubusercontent.com:443@attacker.example/",
                "https",
                "other",
                "absent",
                "present",
            ),
            (
                "https://cache.example.invalid:8443/cache",
                "https",
                "other",
                "present",
                "present",
            ),
            (
                "http://actions.githubusercontent.com/cache",
                "http",
                "github",
                "absent",
                "present",
            ),
            (
                "ftp://cache.example.invalid/cache",
                "other",
                "other",
                "absent",
                "present",
            ),
            (
                "http://localhost/cache",
                "http",
                "localhost",
                "absent",
                "present",
            ),
            (
                "http://127.0.0.1:12345",
                "http",
                "127.0.0.1",
                "present",
                "absent",
            ),
            (
                "http://[::1]:65536/cache",
                "http",
                "ipv6-loopback",
                "present",
                "present",
            ),
        ] {
            let f = prepare();
            let output = f.run(command(&f, LABELS[0], &[(name, endpoint)]));
            assert!(!output.status.success());
            assert!(output.stdout.is_empty());
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains(&format!("GitHub Actions endpoint rejected (variable={name} scheme={scheme} authority={authority} numeric_port={port} explicit_path={path})")),"{stderr}");
            for value in [
                endpoint,
                "cache.example.invalid",
                "cache.depot.dev",
                "attacker.example",
                "actions.githubusercontent.com",
                "/cache",
                "8443",
                "65536",
                "443",
                "12345",
            ] {
                assert!(!stderr.contains(value), "{stderr}");
            }
            assert!(!f.path().join("summary").exists());
            f.0.close().unwrap();
        }
    }
}
