//! Actual config CLI and extracted shell callers; no product/model execution.
#![cfg(unix)]
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
fn invoke(
    root: &Path,
    executable: &str,
    args: Vec<String>,
    env: Vec<(&str, String)>,
) -> process::ProcessReport {
    let mut environment = BTreeMap::from([(
        "PATH".into(),
        Value::Public("/usr/bin:/bin:/opt/homebrew/bin".into()),
    )]);
    for (key, value) in env {
        environment.insert(key.into(), Value::Public(value.into()));
    }
    let spec = ProcessSpec {
        executable: executable.into(),
        arguments: args
            .into_iter()
            .map(|arg| Value::Public(arg.into()))
            .collect(),
        cwd: root.into(),
        environment,
    };
    let limits = Limits {
        execution: Duration::from_secs(8),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let output = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(output.cleanup.complete, "{output:?}");
    output
}
fn arguments(model: &Path, output: &Path) -> Vec<String> {
    [
        "automation",
        "openai-smoke-config",
        "cache",
        "--output",
        output.to_str().unwrap(),
        "--model-id",
        "model \"雪\"\nidentity",
        "--model-path",
        model.to_str().unwrap(),
        "--layer-end",
        "32",
        "--ctx-size",
        "8192",
        "--bind-addr",
        "127.0.0.1:12345",
        "--payload",
        "resident-kv",
        "--flash-attn",
        "disabled",
        "--n-batch",
        "128",
        "--n-ubatch",
        "256",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}
fn read(path: &Path) -> Json {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn source_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn source() -> String {
    fs::read_to_string(source_root().join("skippy/scripts/skippy-ci-smoke.sh")).unwrap()
}
fn function(source: &str, name: &str) -> String {
    let start = source.find(&format!("{name}() {{")).unwrap();
    let end = source[start..].find("\n}\n").unwrap() + start + 3;
    source[start..end].into()
}
#[test]
fn cache_profile_hashes_real_bytes_and_preserves_exact_cpu_cache_contract() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model \"雪\".gguf");
    let output = root.path().join("config.json");
    let bytes = [b"GGUF".as_slice(), &vec![0xa5; 1024 * 1024 + 17]].concat();
    fs::write(&model, &bytes).unwrap();
    for driver in [false, true] {
        let mut args = arguments(&model, &output);
        if driver {
            args.extend(["--upstream-endpoint".into(), "driver".into()]);
        }
        let result = invoke(root.path(), env!("CARGO_BIN_EXE_xtask"), args, vec![]);
        assert!(result.success(), "{result:?}");
        assert_eq!(
            read(&output),
            json!({"run_id":"skippy-ci-smoke","topology_id":"skippy-ci-smoke-single-stage","model_id":"model \"雪\"\nidentity","model_path":model,"source_model_sha256":hex::encode(Sha256::digest(&bytes)),"stage_id":"stage-0","stage_index":0,"layer_start":0,"layer_end":32,"ctx_size":8192,"lane_count":4,"n_batch":128,"n_ubatch":256,"n_gpu_layers":0,"cache_type_k":"f16","cache_type_v":"f16","flash_attn_type":"disabled","load_mode":"runtime-slice","execution_contract":"","bind_addr":"127.0.0.1:12345","upstream":if driver {json!({"stage_id":"stage-0","stage_index":0,"endpoint":"driver"})}else{Json::Null},"downstream":null,"kv_cache":{"mode":"lookup-record","payload":"resident-kv","max_entries":32,"max_bytes":0,"min_tokens":64,"shared_prefix_stride_tokens":128,"shared_prefix_record_limit":2}})
        );
        assert!(fs::read(&output).unwrap().ends_with(b"\n"));
    }
    let previous = read(&output)["source_model_sha256"].clone();
    fs::write(&model, b"changed actual model fixture").unwrap();
    assert!(
        invoke(
            root.path(),
            env!("CARGO_BIN_EXE_xtask"),
            arguments(&model, &output),
            vec![]
        )
        .success()
    );
    assert_eq!(
        read(&output)["source_model_sha256"],
        hex::encode(Sha256::digest(b"changed actual model fixture"))
    );
    assert_ne!(read(&output)["source_model_sha256"], previous);
}
#[test]
fn invalid_cache_settings_do_not_truncate_config_or_change_model() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.gguf");
    let output = root.path().join("config.json");
    fs::write(&model, b"original bytes").unwrap();
    for (key, value) in [
        ("--layer-end", "0"),
        ("--ctx-size", "-1"),
        ("--n-batch", "0"),
        ("--n-ubatch", "1.5"),
        ("--payload", "unknown"),
        ("--flash-attn", "maybe"),
        ("--bind-addr", "0.0.0.0:12345"),
        ("--bind-addr", "127.0.0.1:0"),
    ] {
        fs::write(&output, b"preserve prior output\n").unwrap();
        let mut args = arguments(&model, &output);
        let index = args.iter().position(|arg| arg == key).unwrap();
        args[index + 1] = value.into();
        let result = invoke(root.path(), env!("CARGO_BIN_EXE_xtask"), args, vec![]);
        assert!(!result.success(), "{key}={value}");
        assert_eq!(fs::read(&output).unwrap(), b"preserve prior output\n");
        assert_eq!(fs::read(&model).unwrap(), b"original bytes");
    }
    let mut args = arguments(&model, &output);
    args.extend(["--upstream-endpoint".into(), "other".into()]);
    assert!(!invoke(root.path(), env!("CARGO_BIN_EXE_xtask"), args, vec![]).success());
    fs::remove_file(&model).unwrap();
    assert!(
        !invoke(
            root.path(),
            env!("CARGO_BIN_EXE_xtask"),
            arguments(&model, &output),
            vec![]
        )
        .success()
    );
    assert_eq!(fs::read(output).unwrap(), b"preserve prior output\n");
}
#[test]
fn actual_stage_config_helper_routes_driver_and_normal_http_settings() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("selected.gguf");
    fs::write(&model, b"GGUF actual selected file bytes").unwrap();
    let source = source();
    // Admit the complete production region, including everything after the shell
    // close through the next helper. A leftover heredoc/Python tail must fail.
    let start = source.find("write_stage_config() {").unwrap();
    let end = source[start..].find("make_long_prompt_file() {").unwrap() + start;
    let helper = &source[start..end];
    for driver in ["driver", ""] {
        let output = root.path().join("stage.json");
        let script = format!(
            "set -euo pipefail\nautomation=(\"$OWNER\")\nSMOKE_N_BATCH=128\nSMOKE_N_UBATCH=64\nSMOKE_FLASH_ATTN=enabled\n{helper}\nwrite_stage_config \"$1\" 'quoted model' \"$2\" 16 4096 127.0.0.1:23456 resident-kv 256 128 \"$3\"\n"
        );
        let result = invoke(
            root.path(),
            "/bin/bash",
            vec![
                "-c".into(),
                script,
                "fixture".into(),
                output.display().to_string(),
                model.display().to_string(),
                driver.into(),
            ],
            vec![("OWNER", env!("CARGO_BIN_EXE_xtask").into())],
        );
        assert!(result.success(), "{result:?}");
        let value = read(&output);
        assert_eq!(value["n_batch"], 256);
        assert_eq!(value["n_ubatch"], 128);
        assert_eq!(value["flash_attn_type"], "enabled");
        assert_eq!(value["ctx_size"], 4096);
        assert_eq!(value["bind_addr"], "127.0.0.1:23456");
        assert_eq!(
            value["source_model_sha256"],
            hex::encode(Sha256::digest(b"GGUF actual selected file bytes"))
        );
        assert_eq!(
            value["upstream"],
            if driver.is_empty() {
                Json::Null
            } else {
                json!({"stage_id":"stage-0","stage_index":0,"endpoint":"driver"})
            }
        );
    }
}
#[test]
fn actual_corpus_callers_preserve_current_paragraph_and_same_system_prefix() {
    let root = tempfile::tempdir().unwrap();
    let source = source();
    let input = root.path().join("prompt input.txt");
    let make = function(&source, "make_long_prompt_file");
    let prefix = source
        .lines()
        .find(|line| line.starts_with("openai_shared_prefix="))
        .unwrap();
    let start = source.find("openai_prefix_seed_request=").unwrap();
    let end = source[start..]
        .find("openai_prefix_seed_response=")
        .unwrap()
        + start;
    let request_builders = &source[start..end];
    let script = format!(
        "set -euo pipefail\n{make}\nmake_long_prompt_file \"$1\"\n{prefix}\nDENSE_MODEL_ID='fixture model'\n{request_builders}\nprintf '%s\\n' \"$openai_prefix_seed_request\" > \"$2\"\nprintf '%s\\n' \"$openai_prefix_hit_request\" > \"$3\"\n"
    );
    let result = invoke(
        root.path(),
        "/bin/bash",
        vec![
            "-c".into(),
            script,
            "fixture".into(),
            input.display().to_string(),
            root.path().join("seed.json").display().to_string(),
            root.path().join("hit.json").display().to_string(),
        ],
        vec![("ROOT", source_root().display().to_string())],
    );
    assert!(result.success(), "{result:?}");
    let prompt = fs::read_to_string(&input).unwrap();
    let lines: Vec<_> = prompt.lines().collect();
    let asset = fs::read_to_string(source_root().join("ci/fixtures/skippy-cache-prompt-input.txt"))
        .unwrap();
    assert_eq!(prompt.as_bytes(), asset.as_bytes());
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].len(), 2735);
    assert!(prompt.ends_with('\n'));
    for index in 0..12 {
        assert_eq!(lines[0].matches(&format!("{index:03}. ")).count(), 1);
    }
    assert_eq!(
        lines[0]
            .matches("We are validating exact prefix cache reuse")
            .count(),
        12
    );
    let requests = [
        read(&root.path().join("seed.json")),
        read(&root.path().join("hit.json")),
    ];
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["messages"][0], requests[1]["messages"][0]);
    let text = requests[0]["messages"][0]["content"].as_str().unwrap();
    assert_eq!(
        text.matches("Cache smoke shared system prefix.").count(),
        32
    );
    assert!(text.len() > 1000);
    assert_eq!(
        requests[0]["messages"][1]["content"],
        "Answer with the word seed."
    );
    assert_eq!(
        requests[1]["messages"][1]["content"],
        "Answer with the word hit."
    );
    assert_eq!(requests[0]["max_tokens"], 1);
    assert_eq!(requests[1]["max_tokens"], 1);
}

#[test]
fn actual_corpus_callers_fail_when_required_static_assets_are_missing() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("existing prompt.txt");
    let source = source();
    let make = function(&source, "make_long_prompt_file");
    let prefix = source
        .lines()
        .find(|line| line.starts_with("openai_shared_prefix="))
        .unwrap();
    for body in [
        format!("{make}\nmake_long_prompt_file \"$1\"\n"),
        format!("{prefix}\n"),
    ] {
        fs::write(&output, b"retained output\n").unwrap();
        let script = format!("set -euo pipefail\n{body}\nprintf 'must-not-continue'\n");
        let report = invoke(
            root.path(),
            "/bin/bash",
            vec![
                "-c".into(),
                script,
                "fixture".into(),
                output.display().to_string(),
            ],
            vec![("ROOT", root.path().display().to_string())],
        );
        assert!(!report.success());
        assert!(report.stdout.bytes_retained.is_empty());
        assert_eq!(fs::read(&output).unwrap(), b"retained output\n");
    }
}

#[test]
fn complete_production_smoke_script_parses_as_bash_without_executing_gate_code() {
    let root = tempfile::tempdir().unwrap();
    let script = source_root().join("skippy/scripts/skippy-ci-smoke.sh");
    let report = invoke(
        root.path(),
        "/bin/bash",
        vec!["-n".into(), script.display().to_string()],
        vec![],
    );
    assert!(report.success(), "{report:?}");
    assert!(report.stdout.bytes_retained.is_empty());
}

#[test]
fn actual_staged_correctness_caller_preserves_prompt_argv_and_refuses_failure() {
    let root = tempfile::tempdir().unwrap();
    let source = source();
    let start = source
        .find("run_with_timeout \"staged cache reuse smoke\"")
        .unwrap();
    let start = source[..start]
        .rfind("if ! LLAMA_STAGE_BUILD_DIR=")
        .unwrap();
    let end = source[start..].find("\ncleanup\nSERVER_PID=").unwrap() + start;
    let caller = &source[start..end];
    let prompt = root.path().join("prompt with spaces.txt");
    let asset = fs::read(source_root().join("ci/fixtures/skippy-cache-prompt-input.txt")).unwrap();
    fs::write(&prompt, &asset).unwrap();
    let trace = root.path().join("caller argv");
    for status in [0, 23] {
        let script = format!(
            "set -euo pipefail\nrun_with_timeout() {{ printf '%s\\0' \"$LLAMA_STAGE_BUILD_DIR\" \"$@\" > \"$TRACE\"; return \"$STATUS\"; }}\n{caller}\nprintf caller-complete\n"
        );
        let result = invoke(
            root.path(),
            "/bin/bash",
            vec!["-c".into(), script],
            vec![
                (
                    "LLAMA_BUILD_DIR",
                    root.path().join("native build").display().to_string(),
                ),
                ("PROMPT_OPENAI_URL", "http://127.0.0.1:12345/v1".into()),
                ("DENSE_MODEL_ID", "selected model with spaces".into()),
                ("PROMPT_IN", prompt.display().to_string()),
                ("PROMPT_MAX_NEW_TOKENS", "17".into()),
                (
                    "PROMPT_OUT",
                    root.path().join("prompt output").display().to_string(),
                ),
                (
                    "PROMPT_LOG",
                    root.path().join("missing stage log").display().to_string(),
                ),
                ("TRACE", trace.display().to_string()),
                ("STATUS", status.to_string()),
            ],
        );
        let bytes = fs::read(&trace).unwrap();
        let actual = bytes
            .split(|byte| *byte == 0)
            .filter(|part| !part.is_empty())
            .map(|part| std::str::from_utf8(part).unwrap())
            .collect::<Vec<_>>();
        let build = root.path().join("native build").display().to_string();
        let input = prompt.display().to_string();
        assert_eq!(
            actual,
            [
                build.as_str(),
                "staged cache reuse smoke",
                "target/debug/skippy-correctness",
                "open-ai-cache-reuse",
                "--base-url",
                "http://127.0.0.1:12345/v1",
                "--model",
                "selected model with spaces",
                "--prompt-file",
                input.as_str(),
                "--max-tokens",
                "17"
            ]
        );
        assert_eq!(fs::read(&prompt).unwrap(), asset);
        assert_eq!(result.success(), status == 0, "{result:?}");
        if status == 0 {
            assert_eq!(result.stdout.bytes_retained, b"caller-complete");
        } else {
            assert!(result.stdout.bytes_retained.is_empty());
            assert!(
                String::from_utf8_lossy(&result.stderr.bytes_retained)
                    .contains("staged cache reuse smoke failed")
            );
        }
    }
}
