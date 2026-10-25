//! Actual package-probe adapter execution with inert runtime observers.
use super::{document, job, steps, support, text};
use std::{fs, os::unix::fs::symlink, process::Command};

fn fixture() -> support::Fixture {
    let fixture = support::Fixture::new();
    let runtime = fixture
        .path()
        .join("product's tree with spaces/native-runtimes/cuda-test/tools");
    fs::create_dir_all(&runtime).unwrap();
    fs::create_dir(fixture.path().join("scripts")).unwrap();
    for (name, body) in [
        (
            "verify",
            r#"[[ $# == 1 && "$1" == "product's tree with spaces/native-runtimes/cuda-test" ]] || exit 91
[[ ${LD_LIBRARY_PATH-unset} == unset ]] || exit 92
printf 'verify:unset\n' >> events
[[ "$FAILURE" != verify ]] || exit 41"#,
        ),
        (
            "probe",
            r#"[[ $# == 1 && "$1" == --probe ]] || exit 93
printf 'probe:%s\n' "${LD_LIBRARY_PATH-unset}" >> events
[[ "$FAILURE" != probe ]] || exit 42"#,
        ),
    ] {
        fixture.executable(name, body);
    }
    fs::copy(
        fixture.path().join("bin/verify"),
        fixture
            .path()
            .join("scripts/verify-native-runtime-package.sh"),
    )
    .unwrap();
    fs::copy(
        fixture.path().join("bin/probe"),
        runtime.join("mesh-llm-gpu-benchmark"),
    )
    .unwrap();
    for name in ["find", "env"] {
        symlink(
            std::path::Path::new("/usr/bin").join(name),
            fixture.path().join("bin").join(name),
        )
        .unwrap();
    }
    fixture
}

#[test]
fn artifact_packaged_probe_verifies_then_tests_inherited_and_clean_loader_paths() {
    let document = document("smoke.yml");
    let matches = steps(job(&document, "smoke_tests"))
        .iter()
        .filter(|step| {
            text(step, "name")
                == Some("Verify packaged CUDA runtime without inherited toolkit paths")
        })
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1);
    assert_eq!(
        text(matches[0], "if"),
        Some("inputs.runner == 'gpu-nvidia'")
    );
    let run = text(matches[0], "run").unwrap();
    assert_eq!(run.matches("${{ inputs.artifact_path }}").count(), 1);
    let run = run.replace("${{ inputs.artifact_path }}", "product's tree with spaces");
    for (failure, status, events) in [
        (
            "none",
            0,
            "verify:unset\nprobe:finite-toolkit\nprobe:unset\n",
        ),
        ("verify", 41, "verify:unset\n"),
        ("probe", 42, "verify:unset\nprobe:finite-toolkit\n"),
    ] {
        let fixture = fixture();
        let mut command = Command::new("/bin/bash");
        command
            .env_clear()
            .current_dir(fixture.path())
            .env("PATH", fixture.path().join("bin"));
        command
            .env("HOME", fixture.path())
            .env("TMPDIR", fixture.path())
            .env("LD_LIBRARY_PATH", "finite-toolkit")
            .env("FAILURE", failure);
        command.args(["-c", &run]);
        let result = fixture.run(command);
        assert_eq!(result.status.code(), Some(status), "{result:?}");
        assert_eq!(
            fs::read_to_string(fixture.path().join("events")).unwrap(),
            events
        );
    }
}
