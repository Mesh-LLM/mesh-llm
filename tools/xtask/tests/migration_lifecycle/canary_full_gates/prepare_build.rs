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
                r#"printf '%s\n' "$1" > "$INERT_ROOT/skippy/llama_cpp/upstream.txt""#,
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
                fs::read_to_string(fixture.root.join("skippy/llama_cpp/upstream.txt")).unwrap(),
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
    let pin = fixture.root.join("skippy/llama_cpp/upstream.txt");
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
            "uv",
            "generated",
            "cargo",
            "smoke",
            "oracles",
            "cargo",
            "systemone",
            "laya"
        ]
    );
    let jinja_project = fixture
        .root
        .join("ci/llama-canary/jinja")
        .display()
        .to_string();
    let jinja_python = "python3";
    let locked_jinja_run = vec![
        "run".to_owned(),
        "--locked".to_owned(),
        "--no-python-downloads".to_owned(),
        "--python".to_owned(),
        jinja_python.to_owned(),
        "--project".to_owned(),
        jinja_project,
        "--".to_owned(),
        "arch".to_owned(),
        "-arm64".to_owned(),
        "bash".to_owned(),
    ];
    let mut first_jinja_run = locked_jinja_run.clone();
    first_jinja_run.extend([
        "scripts/build-llama.sh".to_owned(),
        "--build-only".to_owned(),
        "-DCMAKE_OSX_ARCHITECTURES=arm64".to_owned(),
        "-DGGML_METAL_EMBED_LIBRARY=ON".to_owned(),
    ]);
    assert_eq!(fixture.args("uv-1"), first_jinja_run);
    let dependency_project = super::fixture::repository().join("ci/llama-canary/jinja");
    let pyproject = fs::read_to_string(dependency_project.join("pyproject.toml")).unwrap();
    let lock = fs::read_to_string(dependency_project.join("uv.lock")).unwrap();
    assert!(pyproject.contains("jinja2==3.1.6"));
    assert!(lock.contains("name = \"jinja2\"\nversion = \"3.1.6\""));
    assert!(lock.contains("name = \"markupsafe\""));
    assert!(lock.contains("hash = \"sha256:"));
    let expected_jinja_environment = fixture.root.join("jinja-env").display().to_string();
    assert_eq!(
        fixture.values("uv-environment-1"),
        [expected_jinja_environment.as_str()]
    );
    assert_eq!(
        fs::read(fixture.root.join("upstream-tests")).unwrap(),
        b"ON"
    );
    assert_eq!(fixture.args("generated"), ["--full-native-suite"]);
    let mut second_jinja_run = locked_jinja_run;
    second_jinja_run.extend([
        "scripts/check-skippy-generated-family-patch.sh".to_owned(),
        "--full-native-suite".to_owned(),
    ]);
    assert_eq!(fixture.args("uv-2"), second_jinja_run);
    assert_eq!(
        fixture.values("uv-environment-2"),
        [expected_jinja_environment.as_str()]
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
            "skippy-cli",
            "-p",
            "skippy-package-builder",
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
            "skippy-serving",
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

#[test]
fn actual_full_build_uses_controller_jinja_project_for_historical_source_without_manifest() {
    let fixture = Fixture::new();
    fixture.build_tools();
    let controller_manifest = fixture.root.join("ci/llama-canary/jinja/pyproject.toml");
    fs::create_dir_all(controller_manifest.parent().unwrap()).unwrap();
    fs::write(
        &controller_manifest,
        "[project]\nname = \"controller-jinja\"\n",
    )
    .unwrap();
    let historical_source = fixture.root.join("historical-source");
    fs::create_dir_all(&historical_source).unwrap();
    assert!(
        !historical_source
            .join("ci/llama-canary/jinja/pyproject.toml")
            .exists()
    );

    let source = super::fixture::source();
    let project_binding = source
        .find("JINJA_PROJECT=\"$TRUSTED_ROOT/ci/llama-canary/jinja\"")
        .unwrap();
    let selected_root_switch = source.find("ROOT=\"${CANARY_SOURCE_ROOT:").unwrap();
    assert!(project_binding < selected_root_switch);

    let mut names = COMMON.to_vec();
    names.push("run_full_build");
    let report = fixture.run(
        "pinned-build",
        &names,
        "ROOT=\"$ROOT/historical-source\"",
        "run_full_build",
        &[],
    );
    assert_eq!(
        report.process.status.unwrap().code(),
        Some(0),
        "{}",
        fixture.diagnostic(&report)
    );

    let controller_project = fixture
        .root
        .join("ci/llama-canary/jinja")
        .display()
        .to_string();
    assert!(controller_manifest.is_file());
    for invocation in ["uv-1", "uv-2"] {
        let args = fixture.args(invocation);
        let project_path = args
            .iter()
            .position(|argument| argument == "--project")
            .and_then(|index| args.get(index + 1));
        assert_eq!(project_path, Some(&controller_project));
    }
    assert_eq!(fixture.args("generated"), ["--full-native-suite"]);
    fixture.finish();
}
