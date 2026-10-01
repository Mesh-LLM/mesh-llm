use super::{fixture, request};
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{collections::BTreeMap, path::Path, time::Duration};

pub(super) fn invoke(root: &Path, options: &[String]) -> process::RawProcessReport {
    let binary = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("xtask");
    let executable = binary;
    let mut arguments = vec!["automation".into(), "split-evidence".into()];
    for name in super::NAMES {
        arguments.push(format!("--{}", name.replace('_', "-")).into());
        arguments.push(
            root.join(format!("{}.json", name.replace('_', "-")))
                .into_os_string(),
        );
    }
    arguments.extend(["--model-label".into(), "dense".into()]);
    arguments.extend(options.iter().map(Into::into));
    let spec = ProcessSpec {
        executable,
        arguments: arguments.into_iter().map(Value::Public).collect(),
        cwd: root.into(),
        environment: BTreeMap::new(),
    };
    let limits = Limits {
        execution: Duration::from_secs(10),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    process::supervise_raw(
        &spec,
        &limits,
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap()
}

fn census(root: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(root)
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            (
                path.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read(path).unwrap(),
            )
        })
        .collect()
}

#[test]
fn compiled_candidate_when_numeric_and_container_evidence_varies() {
    numeric_cases();
}

fn numeric_cases() {
    let fallback = tempfile::tempdir().unwrap();
    let evidence = std::env::var_os("SPLIT_VALIDATION_EVIDENCE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| fallback.path().to_owned());
    let parent = evidence.as_path();
    for (case, replacement, expected) in [
        ("bool", "true", true),
        ("float", "1.0", true),
        ("wrong", "2", false),
        ("fraction", "1.5", false),
        ("nan", "NaN", false),
        ("infinity", "Infinity", false),
        ("negative-infinity", "-Infinity", false),
        ("overflow", "1e400", false),
        ("signed-zero", "1", true),
        ("nested", "1", true),
        ("ordered", "1", false),
        ("decoded-key", "1", true),
        ("distinct-key", "1", false),
        ("adjacent53", "1", false),
        ("exact53", "1", true),
    ] {
        let root = tempfile::Builder::new()
            .prefix(case)
            .tempdir_in(parent)
            .unwrap()
            .keep();
        let _request = request(&root);
        let bound = match case {
            "exact53" | "adjacent53" => "9007199254740992",
            "exact128" | "adjacent128" => "340282366920938463463374607431768211456",
            _ => "24",
        };
        if bound != "24" {
            for name in ["seed-stages.json", "worker-stages.json"] {
                let path = root.join(name);
                let bytes = std::fs::read_to_string(&path)
                    .unwrap()
                    .replace("\"layer_end\":24", &format!("\"layer_end\":{bound}"));
                std::fs::write(path, bytes).unwrap();
            }
        }
        let candidate = invoke(&root, &["--output".into(), "candidate.json".into()]);
        assert!(candidate.process.success(), "{case}");
        if bound == "24" {
            assert_eq!(
                std::fs::read(root.join("candidate.json")).unwrap(),
                std::fs::read(fixture().join("expected-ready.json")).unwrap(),
                "{case}"
            );
        }
        let original = std::fs::read_to_string(root.join("candidate.json")).unwrap();
        let mut changed = original.replace(
            "\"schema_version\": 1",
            &format!("\"schema_version\": {replacement}"),
        );
        changed = match case {
            "signed-zero" => changed.replace("\"layer_start\": 0", "\"layer_start\": -0.0"),
            "nested" => changed
                .replace("\"stage_index\": 0", "\"stage_index\": false")
                .replace("\"stage_index\": 1", "\"stage_index\": true")
                .replace("\"layer_end\": 24", "\"layer_end\": 24.0"),
            "ordered" => {
                let mut value: serde_json::Value = serde_json::from_str(&changed).unwrap();
                value["topology"]["stages"]
                    .as_array_mut()
                    .unwrap()
                    .reverse();
                serde_json::to_string(&value).unwrap()
            }
            "decoded-key" => changed.replace("\"schema_version\"", "\"schema_\\u0076ersion\""),
            "distinct-key" => changed.replace("\"schema_version\"", "\"schema_\\ud800ersion\""),
            "exact53" => changed.replace(
                &format!("\"layer_end\": {bound}"),
                "\"layer_end\": 9007199254740992.0",
            ),
            "adjacent53" => changed.replace(
                &format!("\"layer_end\": {bound}"),
                "\"layer_end\": 9007199254740993",
            ),
            "exact128" => changed.replace(
                &format!("\"layer_end\": {bound}"),
                "\"layer_end\": 3.402823669209385e38",
            ),
            "adjacent128" => changed.replace(
                &format!("\"layer_end\": {bound}"),
                "\"layer_end\": 340282366920938463463374607431768211457",
            ),
            _ => changed,
        };
        std::fs::write(root.join("verify.json"), changed).unwrap();
        let before = census(&root);
        let candidate = invoke(&root, &["--verify".into(), "verify.json".into()]);
        assert_eq!(candidate.process.success(), expected, "{case}");
        assert_eq!(census(&root), before, "{case}");
        std::fs::write(
            root.join("captures.txt"),
            format!("candidate={candidate:?}\n"),
        )
        .unwrap();
    }
}
