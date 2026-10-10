//! Exercise the checked-in SafeTensors compile-and-locate step without compilation.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .into()
}

fn actual_step() -> Result<String, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(root().join(".github/workflows/ci-rust-tests-slice.yml"))?;
    let section = text
        .split_once("        id: safetensors_smoke_test\n")
        .ok_or("missing owning step")?
        .1;
    let body = section
        .strip_prefix("        run: |\n")
        .ok_or("missing owning run scalar")?;
    let lines: Vec<_> = body
        .lines()
        .take_while(|line| line.is_empty() || line.starts_with("          "))
        .collect();
    if lines.is_empty() {
        return Err("empty owning run scalar".into());
    }
    Ok(lines
        .into_iter()
        .map(|line| line.strip_prefix("          ").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n"))
}

fn executable(path: &Path, body: &str) -> TestResult {
    fs::write(path, body)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    Ok(())
}

fn fixture(directory: &Path, adapter: bool, present: bool) -> TestResult {
    let owner = if adapter {
        "mesh-llm-skippy-adapter"
    } else {
        "mesh-llm-host-runtime"
    };
    let module = if adapter {
        "config::hardware_translation_tests"
    } else {
        "inference::skippy::resolver::tests"
    };
    let name = format!("{module}::safetensors_checkpoint_reaches_mesh_host_runtime");
    fs::write(
        directory.join("metadata.json"),
        serde_json::to_vec(&json!({
            "workspace_members":[owner], "packages":[{"id":owner,"name":owner},
            {"id":"external","name":"mesh-llm-skippy-adapter"}]
        }))?,
    )?;
    fs::write(
        directory.join("listing"),
        if present {
            format!("{name}: test\n")
        } else {
            format!("{name}_wrong: test\nother: test\n")
        },
    )?;
    let binary = directory.join("selected test executable");
    executable(
        &binary,
        "#!/bin/sh\nset -eu\nprintf '%s\\n' \"$@\" > \"$FIXTURE/list-argv\"\ncat \"$FIXTURE/listing\"\n",
    )?;
    fs::write(
        directory.join("artifact.json"),
        serde_json::to_vec(&json!({
            "reason":"compiler-artifact", "target":{"name":owner.replace('-',"_")},
            "profile":{"test":true}, "executable":binary
        }))?,
    )?;
    executable(
        &directory.join("cargo"),
        "#!/bin/sh\nset -eu\ncase \"$1\" in\n metadata) printf '%s\\n' \"$@\" > \"$FIXTURE/metadata-argv\"; cat \"$FIXTURE/metadata.json\";;\n test) printf 'compile\\n' >> \"$FIXTURE/compilations\"; printf '%s\\n' \"$@\" > \"$FIXTURE/compile-argv\"; cat \"$FIXTURE/artifact.json\";;\n *) exit 91;;\nesac\n",
    )?;
    Ok(())
}

fn run(directory: &Path, body: String) -> process::ProcessReport {
    let mut environment = BTreeMap::new();
    environment.insert(
        "PATH".into(),
        Value::Public(format!("{}:{}", directory.display(), std::env::var("PATH").unwrap()).into()),
    );
    environment.insert("FIXTURE".into(), Value::Public(directory.into()));
    environment.insert(
        "GITHUB_OUTPUT".into(),
        Value::Public(directory.join("output").into()),
    );
    let spec = ProcessSpec {
        executable: "/bin/bash".into(),
        arguments: vec![Value::Public("-c".into()), Value::Public(body.into())],
        cwd: directory.into(),
        environment,
    };
    let limits = Limits {
        execution: Duration::from_secs(10),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        report.cleanup.complete && !report.stdout.truncated && !report.stderr.truncated,
        "{report:?}"
    );
    report
}

fn verify(directory: &Path, adapter: bool, present: bool) -> TestResult {
    let owner = if adapter {
        "mesh-llm-skippy-adapter"
    } else {
        "mesh-llm-host-runtime"
    };
    let module = if adapter {
        "config::hardware_translation_tests"
    } else {
        "inference::skippy::resolver::tests"
    };
    assert_eq!(
        fs::read_to_string(directory.join("compilations"))?,
        "compile\n"
    );
    assert_eq!(
        fs::read_to_string(directory.join("metadata-argv"))?,
        "metadata\n--locked\n--no-deps\n--format-version=1\n"
    );
    assert_eq!(
        fs::read_to_string(directory.join("compile-argv"))?,
        format!(
            "test\n--locked\n-p\n{owner}\n--no-default-features\n--lib\n--no-run\n--message-format=json\n"
        )
    );
    assert_eq!(
        fs::read_to_string(directory.join("list-argv"))?,
        "--list\n--ignored\n--format\nterse\n"
    );
    if present {
        assert_eq!(
            fs::read_to_string(directory.join("output"))?,
            format!(
                "test_binary={}\ntest_name={module}::safetensors_checkpoint_reaches_mesh_host_runtime\n",
                directory.join("selected test executable").display()
            )
        );
    } else {
        assert!(
            !directory.join("output").exists(),
            "missing exact test exported success outputs"
        );
    }
    Ok(())
}

#[test]
fn actual_safetensors_step_compiles_selected_owner_once_and_requires_exact_ignored_test()
-> TestResult {
    for adapter in [false, true] {
        for present in [false, true] {
            let directory = tempfile::Builder::new()
                .prefix("safetensors caller space ")
                .tempdir()?;
            fixture(directory.path(), adapter, present)?;
            let report = run(directory.path(), actual_step()?);
            assert_eq!(
                report.success(),
                present,
                "adapter={adapter}, present={present}: {report:?}"
            );
            verify(directory.path(), adapter, present)?;
        }
    }
    Ok(())
}
