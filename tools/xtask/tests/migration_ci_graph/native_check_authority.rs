//! Actual parsed ten-entry graph authority; finite CLI only, no GitHub execution.
use crate::{
    process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    },
    support::{entrypoint_files, repository_root},
};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, time::Duration};
fn actual(workflows: &Path) -> (i32, String) {
    let environment = ["PATH", "SystemRoot", "WINDIR", "TEMP", "TMP"]
        .into_iter()
        .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
        .collect::<BTreeMap<_, _>>();
    let result = process::supervise_raw(
        &ProcessSpec {
            executable: Path::new(env!("CARGO_BIN_EXE_xtask")).to_owned(),
            cwd: repository_root(),
            environment,
            arguments: vec![
                Value::Public("ci".into()),
                Value::Public("validate-graph".into()),
                Value::Public("--workflows".into()),
                Value::Public(workflows.as_os_str().to_owned()),
            ],
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(result.process.failure.is_none(), "{:?}", result.process);
    assert!(result.process.cleanup.complete);
    assert!(result.stdout.unwrap().as_bytes().is_empty());
    (
        result.process.status.unwrap().code().unwrap(),
        String::from_utf8_lossy(result.stderr.unwrap().as_bytes()).into_owned(),
    )
}
fn copied(file: &str, transform: impl FnOnce(String) -> String) -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    let root = repository_root().join(".github/workflows");
    for name in entrypoint_files() {
        fs::copy(root.join(&name), temp.path().join(name)).unwrap();
    }
    let original = fs::read_to_string(root.join(file)).unwrap();
    fs::write(temp.path().join(file), transform(original)).unwrap();
    temp
}
fn all_files() -> Vec<String> {
    ["pr", "main"]
        .into_iter()
        .flat_map(|entry| {
            ["quality", "website", "linux", "macos", "windows"]
                .into_iter()
                .map(move |lane| format!("{entry}_{lane}.yml"))
        })
        .collect()
}
fn rejection(workflows: &Path, file: &str, message: &str) {
    let (status, diagnostic) = actual(workflows);
    assert_eq!(status, 2, "{diagnostic}");
    assert!(
        diagnostic.contains(file) && diagnostic.contains(message),
        "{diagnostic}"
    );
}
#[test]
fn native_check_authority_actual_ten_entries_and_read_only_ancillary_steps_pass() {
    let (status, diagnostic) = actual(&repository_root().join(".github/workflows"));
    assert_eq!(status, 0, "{diagnostic}");
    for file in all_files() {
        let temp = copied(&file, |mut source| {
            source.push_str("\n      - name: Ancillary read-only metadata\n        run: echo 'literal checks.create mention is harmless'\n");
            source
        });
        let (status, diagnostic) = actual(temp.path());
        assert_eq!(status, 0, "{file}: {diagnostic}");
    }
}
#[test]
fn native_check_authority_actual_ten_entries_refuse_permission_escalation_and_inheritance() {
    for file in all_files() {
        for (from, to, job) in [
            (
                "    permissions:\n      contents: read\n",
                "    permissions:\n      checks: write\n      contents: read\n",
                "plan",
            ),
            ("    permissions:\n      contents: read\n", "", "plan"),
            (
                "    permissions: {}\n",
                "    permissions: write-all\n",
                "required",
            ),
            ("    permissions: {}\n", "", "required"),
        ] {
            let temp = copied(&file, |source| {
                assert_eq!(source.matches(from).count(), 1);
                source.replace(from, to)
            });
            rejection(
                temp.path(),
                &file,
                &format!("native entry job {job} must have effective checks permission none"),
            );
        }
    }
}
#[test]
fn native_check_authority_actual_ten_entries_refuse_supplied_controller_fields_even_empty() {
    for file in all_files() {
        for key in ["lane_check_id", "overall_check_id", "correlation_id"] {
            for value in ["'701'", "''"] {
                let temp = copied(&file, |source| {
                    let anchor = "    with:\n      lane_plan_json:";
                    assert_eq!(source.matches(anchor).count(), 1);
                    source.replace(
                        anchor,
                        &format!("    with:\n      {key}: {value}\n      lane_plan_json:"),
                    )
                });
                rejection(
                    temp.path(),
                    &file,
                    &format!("direct native lane must omit controller-owned {key}"),
                );
            }
        }
    }
}

#[test]
fn native_check_authority_actual_ten_entries_bind_quoted_permissions() {
    for file in all_files() {
        for (mapping, admitted, message) in [
            (
                "{'checks': write}",
                false,
                "effective checks permission none",
            ),
            (
                r#"{"checks": write}"#,
                false,
                "effective checks permission none",
            ),
            (
                "{checks: none, 'checks': write}",
                false,
                "duplicate permission checks",
            ),
            (
                r#"{"checks": none, checks: write}"#,
                false,
                "duplicate permission checks",
            ),
            (
                r#"{"\u0063hecks": write}"#,
                false,
                "unsupported permission key syntax",
            ),
            ("{'checks': none}", true, ""),
            (r#"{"checks": none}"#, true, ""),
            ("{'contents': read}", true, ""),
            (r#"{"contents": read}"#, true, ""),
        ] {
            let temp = copied(&file, |source| {
                let anchor = "    permissions: {}\n";
                assert_eq!(source.matches(anchor).count(), 1);
                source.replace(anchor, &format!("    permissions: {mapping}\n"))
            });
            let (status, diagnostic) = actual(temp.path());
            if admitted {
                assert_eq!(status, 0, "{file}: {mapping}: {diagnostic}");
            } else {
                assert_eq!(status, 2, "{file}: {mapping}: {diagnostic}");
                assert!(
                    diagnostic.contains(message),
                    "{file}: {mapping}: {diagnostic}"
                );
            }
        }
    }
}
