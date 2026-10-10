use super::{execute, mutate, request};
use serde_json::json;

#[test]
fn persists_failure_when_each_legacy_mismatch_occurs() {
    for case in 0..13 {
        let root = tempfile::tempdir().unwrap();
        let request = request(root.path());
        match case {
            0 => mutate(&request, 4, |value| {
                value["topologies"][0]["run_id"] = json!("run-b")
            }),
            1 => mutate(&request, 1, |value| {
                let stage = value["topologies"][0]["stages"][1].clone();
                value["topologies"][0]["stages"]
                    .as_array_mut()
                    .unwrap()
                    .push(stage);
            }),
            2 => mutate(&request, 1, |value| {
                value["topologies"][0]["stages"][1]["layer_start"] = json!(13)
            }),
            3 => mutate(&request, 3, |value| value["node_id"] = json!("seed-node")),
            4 => mutate(&request, 4, |value| {
                value["statuses"][1]["state"] = json!("starting")
            }),
            5 => mutate(&request, 5, |value| {
                value["data"][0]["id"] = json!("model-b")
            }),
            6 => mutate(&request, 4, |value| {
                value["topologies"][0]["package_ref"] = json!("other")
            }),
            7 => mutate(&request, 4, |value| {
                value["topologies"][0]["stages"][1]["endpoint"]["bind_addr"] = json!("other")
            }),
            8..=10 => {
                for index in [1, 4] {
                    mutate(&request, index, |value| match case {
                        8 => value["statuses"][0]["manifest_sha256"] = json!("b".repeat(64)),
                        9 => value["statuses"][0]["package_ref"] = json!("other"),
                        10 => value["statuses"][1]["bind_addr"] = json!("other"),
                        _ => unreachable!(),
                    });
                }
            }
            11 => mutate(&request, 3, |value| value["mesh_id"] = json!("mesh-b")),
            12 => mutate(&request, 4, |value| {
                value["statuses"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("package_ref");
            }),
            _ => unreachable!(),
        }
        if let Some(parent) = std::env::var_os("SPLIT_VALIDATION_EVIDENCE") {
            let retained = tempfile::Builder::new()
                .prefix(&format!("historic-{case}-"))
                .tempdir_in(parent)
                .unwrap()
                .keep();
            for path in &request.paths {
                std::fs::copy(path, retained.join(path.file_name().unwrap())).unwrap();
            }
            let candidate = super::executable_parity::invoke(
                &retained,
                &["--output".into(), "candidate.json".into()],
            );
            std::fs::write(
                retained.join("captures.txt"),
                format!("candidate={candidate:?}\n"),
            )
            .unwrap();
            assert!(!candidate.process.success(), "case {case}");
            {
                let name = "candidate.json";
                let failure: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(retained.join(name)).unwrap()).unwrap();
                assert_eq!(failure["status"], "failed", "case {case}");
                assert_eq!(failure.as_object().unwrap().len(), 5);
            }
        }
        assert!(execute(&request).is_err(), "case {case}");
        let failure: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.path().join("split-evidence.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(failure["status"], "failed", "case {case}");
        assert_eq!(failure.as_object().unwrap().len(), 5);
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 7);
    }
}
