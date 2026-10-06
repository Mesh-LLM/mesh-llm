//! Exact three owning wrappers with the real native endpoint frontend; no transport calls.
use super::*;
use std::process::Command;
#[test]
fn authority_all_three_actual_phase_wrappers_preserve_typed_refusal_and_redaction() {
    let phases = [
        ("ci-quality-slice.yml", "authority_sentinel"),
        ("depot-canary.yml", "seed_authority_marker"),
        ("depot-canary.yml", "verify_pr_write"),
    ];
    let scripts: Vec<_> = phases
        .iter()
        .map(|(workflow, job)| body(workflow, job, "Attest provider-injected cache backend"))
        .collect();
    let commands = |script: &str| {
        script
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(commands(&scripts[0]), commands(&scripts[1]));
    assert_eq!(commands(&scripts[1]), commands(&scripts[2]));
    let valid_cache = "http://cache.fixture.invalid:1234/cache";
    let valid_results = "http://results.fixture.invalid:5678/results";
    let cases = [
        ("http://127.0.0.0:1234/cache", "loopback"),
        ("http://127.1.2.3:1234/cache", "loopback"),
        ("http://127.255.255.255:1234/cache", "loopback"),
        ("http://[::1]:1234/cache", "loopback"),
        ("http://[::ffff:127.0.0.1]:1234/cache", "loopback"),
        ("http://[::ffff:7f00:1]:1234/cache", "loopback"),
        ("http://[0:0:0:0:0:ffff:127.0.0.1]:1234/cache", "loopback"),
        ("http://[0::1]:1234/cache", "loopback"),
        ("http://[not-an-ip]:1234/cache", "parser"),
        ("https://remote.fixture:1234/cache", "scheme"),
        ("http://actions.githubusercontent.com:1234/cache", "github"),
        (
            "http://user:private-fixture-token@remote.fixture:1234/cache",
            "userinfo",
        ),
        ("http://remote.fixture/cache", "port"),
        ("http://remote.fixture:0/cache", "port"),
        ("http://remote.fixture:65536/cache", "port"),
        ("http://remote.fixture:1234", "path"),
        (
            "http://remote.fixture:1234/private-fixture path",
            "whitespace",
        ),
        ("", "missing"),
    ];
    for script in scripts {
        let f = support::Fixture::new();
        let invoke = |cache: &str, results: &str, token: bool| {
            let mut c = Command::new("/bin/bash");
            c.env_clear()
                .current_dir(f.path())
                .env("PATH", "/usr/bin:/bin")
                .env("HOME", f.path())
                .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
                .env("ACTIONS_CACHE_URL", cache)
                .env("ACTIONS_RESULTS_URL", results)
                .args(["-c", &script]);
            if token {
                c.env("ACTIONS_RUNTIME_TOKEN", "private-fixture-token");
            }
            f.run(c)
        };
        for token in [false, true] {
            let output = invoke(valid_cache, valid_results, token);
            assert!(output.status.success());
            assert!(output.stdout.is_empty() && output.stderr.is_empty());
        }
        for name in ["ACTIONS_CACHE_URL", "ACTIONS_RESULTS_URL"] {
            for (endpoint, reason) in cases {
                let output = if name == "ACTIONS_CACHE_URL" {
                    invoke(endpoint, valid_results, true)
                } else {
                    invoke(valid_cache, endpoint, true)
                };
                assert!(!output.status.success());
                assert!(output.stdout.is_empty());
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(stderr.contains(&format!("{name}: {reason}")), "{stderr}");
                assert!(
                    !stderr.contains("private-fixture")
                        && !stderr.contains("cache.fixture.invalid")
                        && !stderr.contains("results.fixture.invalid")
                        && !stderr.contains("remote.fixture")
                );
            }
        }
        f.0.close().unwrap();
    }
}
