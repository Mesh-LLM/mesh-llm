use super::fixture::Fixture;
use std::fs;

const COMMON: &[&str] = &[
    "run_for",
    "remaining_verification_seconds",
    "run_verification_logged",
];

#[test]
fn actual_prepare_admits_relative_script_in_all_five_modes_and_refuses_wrong_stamp() {
    for mode in [
        "repair",
        "verify",
        "repair-build",
        "verify-build",
        "pinned-build",
    ] {
        for wrong in [false, true] {
            let fixture = Fixture::new();
            fixture.tool(
                "scripts/update-llama-pin.sh",
                "update",
                r#"printf '%s\n' "$1" > "$INERT_ROOT/third_party/llama.cpp/upstream.txt""#,
            );
            fixture.tool("scripts/prepare-llama.sh", "prepare", r#"printf '%s\n' "$INERT_PREPARED_SHA" > "$INERT_ROOT/.deps/llama.cpp/.mesh-llm-upstream-sha""#);
            let mut names = COMMON.to_vec();
            names.extend(["write_repair_pin", "verify_repair_pin", "run_prepare"]);
            let report = fixture.run(
                mode,
                &names,
                "",
                "run_prepare",
                &[(
                    "INERT_PREPARED_SHA",
                    if wrong { "b" } else { "a" }.repeat(40),
                )],
            );
            assert_eq!(
                report.process.status.unwrap().code(),
                Some(i32::from(wrong)),
                "mode={mode}, wrong={wrong}: {}",
                fixture.diagnostic(&report)
            );
            assert_eq!(fixture.args("prepare"), ["pinned"]);
            if mode == "pinned-build" {
                assert_eq!(fixture.order(), ["prepare"]);
                assert!(!fixture.root.join("update.args").exists());
            } else {
                assert_eq!(fixture.order(), ["update", "prepare"]);
                assert_eq!(fixture.args("update"), ["a".repeat(40)]);
            }
            assert_eq!(
                fs::read_to_string(fixture.root.join("third_party/llama.cpp/upstream.txt"))
                    .unwrap(),
                format!("{}\n", "a".repeat(40))
            );
            if wrong {
                assert!(
                    fs::read_to_string(fixture.root.join("prepare.log"))
                        .unwrap()
                        .contains("prepared upstream is")
                );
            }
            fixture.finish();
        }
        // The same actual timeout handoff preserves absolute argv and refuses a shell builtin.
        let fixture = Fixture::new();
        let report = fixture.run(mode, &["run_for"], "", r#"
run_for 'absolute argument custody' 3 /bin/bash -c 'printf "%s\0" "$@" > absolute.args' child '' 'space value' '*.txt'
if run_for 'builtin refusal' 3 printf forbidden; then exit 99; else status=$?; fi
[[ "$status" == 125 ]]
"#, &[]);
        assert_eq!(
            report.process.status.unwrap().code(),
            Some(0),
            "{}",
            fixture.diagnostic(&report)
        );
        assert_eq!(
            fixture.values("absolute.args"),
            ["", "space value", "*.txt"]
        );
        assert!(
            report
                .stderr
                .as_ref()
                .unwrap()
                .as_bytes()
                .windows(b"requires an absolute executable".len())
                .any(|part| part == b"requires an absolute executable")
        );
        fixture.finish();
    }
    let fixture = Fixture::new();
    let pin = fixture.root.join("third_party/llama.cpp/upstream.txt");
    fs::write(&pin, format!("{}\n", "b".repeat(40))).unwrap();
    let mut names = COMMON.to_vec();
    names.extend(["write_repair_pin", "verify_repair_pin", "run_prepare"]);
    let report = fixture.run("pinned-build", &names, "", "run_prepare", &[]);
    assert_eq!(report.process.status.unwrap().code(), Some(1));
    assert!(!fixture.root.join("order").exists());
    assert_eq!(
        fs::read_to_string(pin).unwrap(),
        format!("{}\n", "b".repeat(40))
    );
    fixture.finish();
}

#[test]
fn actual_full_build_preserves_arm64_generators_full_crates_and_both_oracle_environments() {
    let fixture = Fixture::new();
    fixture.build_tools();
    let mut names = COMMON.to_vec();
    names.push("run_full_build");
    let report = fixture.run("verify-build", &names, "", "run_full_build", &[]);
    assert_eq!(
        report.process.status.unwrap().code(),
        Some(0),
        "{}",
        fixture.diagnostic(&report)
    );
    assert_eq!(
        fixture.order(),
        [
            "uv",
            "archive",
            "generated",
            "cargo",
            "smoke",
            "oracles",
            "cargo",
            "systemone",
            "laya"
        ]
    );
    assert_eq!(
        fixture.args("uv"),
        [
            "run",
            "--no-project",
            "--with",
            "jinja2==3.1.6",
            "--",
            "arch",
            "-arm64",
            "bash",
            "scripts/build-llama.sh",
            "-DCMAKE_OSX_ARCHITECTURES=arm64",
            "-DGGML_METAL_EMBED_LIBRARY=ON"
        ]
    );
    assert_eq!(
        fs::read(fixture.root.join("upstream-tests")).unwrap(),
        b"ON"
    );
    assert_eq!(
        fixture.args("archive"),
        [
            "-archs".to_owned(),
            fixture
                .root
                .join("native/src/libllama.a")
                .display()
                .to_string()
        ]
    );
    assert_eq!(
        fixture.args("cargo-1"),
        [
            "build",
            "-p",
            "skippy-runtime",
            "-p",
            "skippy-server",
            "-p",
            "skippy-model-package",
            "-p",
            "skippy-correctness",
            "-p",
            "skippy-topology",
            "--bins"
        ]
    );
    assert_eq!(
        fixture.args("cargo-2"),
        [
            "test",
            "-p",
            "skippy-server",
            "--lib",
            "--no-run",
            "--message-format=json"
        ]
    );
    assert_eq!(
        fixture.args("oracles"),
        [
            "skippy-workload-oracles-build".to_owned(),
            format!("{}-workloads", fixture.root.join("native").display())
        ]
    );
    let work = fixture
        .root
        .join("target/skippy-system-one-smoke")
        .display()
        .to_string();
    assert_eq!(
        fixture.values("systemone.environment"),
        [
            work.clone(),
            "llama-bump".into(),
            "metal".into(),
            "metal".into(),
            "1".into()
        ]
    );
    assert_eq!(
        fixture.values("laya.environment"),
        [work, "llama-bump".into(), "CPU".into()]
    );
    assert!(fixture.root.join("mm-build.jsonl").is_file());
    fixture.finish();
    for (failure, expected_last) in [
        ("uv", "uv"),
        ("generated", "generated"),
        ("cargo", "cargo"),
        ("smoke", "smoke"),
        ("oracles", "oracles"),
        ("systemone", "systemone"),
        ("laya", "laya"),
    ] {
        let fixture = Fixture::new();
        fixture.build_tools();
        let report = fixture.run(
            "verify-build",
            &names,
            "",
            "run_full_build",
            &[("INERT_FAIL", failure.into())],
        );
        assert_eq!(report.process.status.unwrap().code(), Some(1));
        assert_eq!(fixture.order().last().unwrap(), expected_last);
        fixture.finish();
    }
    let fixture = Fixture::new();
    fixture.build_tools();
    let report = fixture.run(
        "pinned-build",
        &names,
        "",
        "run_full_build",
        &[("INERT_ARCHIVE", "x86_64".into())],
    );
    assert_eq!(report.process.status.unwrap().code(), Some(1));
    assert_eq!(fixture.order(), ["uv", "archive"]);
    assert!(
        fs::read_to_string(fixture.root.join("build.log"))
            .unwrap()
            .contains("candidate native archive must be arm64")
    );
    fixture.finish();
}
