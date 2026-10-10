use serde_json::{Value, json};
use std::{
    path::Path,
    process::{Command, Output},
};

fn execute(output: &Path, layers: &str, end: &str, sha: &str, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "workload-smoke-config", "--output"])
        .arg(output)
        .args([
            "--model-id",
            "quoted \"model\"\n雪",
            "--model-path",
            "/abs/model path.gguf",
            "--model-sha256",
            sha,
            "--layer-end",
            end,
            "--n-gpu-layers",
            layers,
        ])
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn cpu_unsplit_config_preserves_full_shape_and_escaped_model_identity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    let result = execute(&path, "0", "32", &"a".repeat(64), &[]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.ends_with(b"\n"));
    let actual: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        actual,
        json!({
            "run_id":"workload-http-smoke", "topology_id":"workload-http-smoke-local",
            "model_id":"quoted \"model\"\n雪", "model_path":"/abs/model path.gguf", "source_model_sha256":"a".repeat(64),
            "stage_id":"stage-0", "stage_index":0, "layer_start":0, "layer_end":32,
            "ctx_size":2048, "lane_count":1, "n_batch":2048, "n_ubatch":2048, "n_gpu_layers":0,
            "selected_device":{"backend_device":"CPU"}, "kv_offload":false, "op_offload":false,
            "resident_tensor_names":[], "execution_contract":"", "native_mtp_enabled":false,
            "load_mode":"runtime-slice", "bind_addr":"127.0.0.1:0"
        })
    );
}

#[test]
fn cpu_gpu_and_all_gpu_projector_configs_preserve_backend_selection() {
    let dir = tempfile::tempdir().unwrap();
    for layers in ["0", "99", "-1"] {
        let path = dir.path().join(format!("{layers}.json"));
        let result = execute(
            &path,
            layers,
            "32",
            &"b".repeat(64),
            &["--projector-path", "/abs/projector \"雪\".gguf"],
        );
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        if layers == "0" {
            assert_eq!(actual["selected_device"], json!({"backend_device":"CPU"}));
            assert_eq!(actual["kv_offload"], false);
            assert_eq!(actual["op_offload"], false);
        } else {
            for field in ["selected_device", "kv_offload", "op_offload"] {
                assert!(actual[field].is_null());
            }
        }
        assert_eq!(actual["n_gpu_layers"], layers.parse::<i32>().unwrap());
        assert_eq!(actual["projector_path"], "/abs/projector \"雪\".gguf");
    }
}

#[test]
fn invalid_input_cannot_truncate_retained_config() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("retained.json");
    for (layers, end, sha, extra) in [
        ("bad", "32", "a".repeat(64), vec![]),
        ("0", "-1", "a".repeat(64), vec![]),
        ("0", "32", "unverified".into(), vec![]),
        ("0", "32", "a".repeat(64), vec!["--invented-option"]),
    ] {
        std::fs::write(&path, b"retained\n").unwrap();
        let result = execute(&path, layers, end, &sha, &extra);
        assert!(!result.status.success());
        assert_eq!(std::fs::read(&path).unwrap(), b"retained\n");
    }
}
