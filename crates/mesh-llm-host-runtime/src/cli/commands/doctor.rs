use anyhow::{Context, Result};
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cli::DoctorCommand;
use crate::runtime::instance::{LocalInstanceSnapshot, runtime_root, scan_local_instances};

const SKIPPY_NATIVE_LOG_FILE: &str = "skippy-native.log";
const SKIPPY_DIAGNOSTIC_ENDPOINTS: &[(&str, &str, &str)] = &[
    ("status", "/api/status", "status.json"),
    ("runtime", "/api/runtime", "runtime.json"),
    (
        "runtime_stages",
        "/api/runtime/stages",
        "runtime-stages.json",
    ),
    (
        "runtime_endpoints",
        "/api/runtime/endpoints",
        "runtime-endpoints.json",
    ),
    ("runtime_llama", "/api/runtime/llama", "runtime-llama.json"),
    ("plugins", "/api/plugins", "plugins.json"),
    (
        "plugin_endpoints",
        "/api/plugins/endpoints",
        "plugin-endpoints.json",
    ),
    (
        "plugin_providers",
        "/api/plugins/providers",
        "plugin-providers.json",
    ),
];

pub(crate) async fn dispatch_doctor_command(command: &DoctorCommand) -> Result<()> {
    match command {
        DoctorCommand::Split {
            model_ref,
            port,
            json,
            output_dir,
        } => run_split_doctor(model_ref, *port, *json, output_dir.as_deref()).await,
    }
}

async fn run_split_doctor(
    model_ref: &str,
    port: u16,
    json_output: bool,
    output_dir: Option<&Path>,
) -> Result<()> {
    let report = fetch_split_readiness_report(model_ref, port).await?;
    let captured_files = match output_dir {
        Some(output_dir) => write_split_doctor_bundle(output_dir, port, &report).await?,
        None => Vec::new(),
    };
    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        for line in split_readiness_lines(&report) {
            println!("{line}");
        }
        if !captured_files.is_empty() {
            println!();
            println!("Captured diagnostics:");
            for path in captured_files {
                println!("  - {}", path.display());
            }
        }
    }
    Ok(())
}

async fn fetch_split_readiness_report(model_ref: &str, port: u16) -> Result<serde_json::Value> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let encoded = urlencoding::encode(model_ref);
    let url =
        format!("http://127.0.0.1:{port}/api/diagnostics/split-readiness?model_ref={encoded}");
    client
        .get(&url)
        .send()
        .await
        .with_context(|| {
            format!("Can't connect to mesh-llm console on port {port}. Is it running?")
        })?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await
        .map_err(Into::into)
}

async fn write_split_doctor_bundle(
    output_dir: &Path,
    port: u16,
    report: &Value,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create split doctor output dir {}", output_dir.display()))?;
    let mut files = Vec::new();
    files.push(write_split_readiness_report(output_dir, report)?);

    let mut diagnostics = Map::new();
    diagnostics.insert("captured_at_unix_ms".to_string(), json!(unix_time_ms()));
    diagnostics.insert("console_port".to_string(), json!(port));
    diagnostics.insert(
        "api".to_string(),
        capture_skippy_api_diagnostics(output_dir, port, &mut files).await?,
    );
    diagnostics.insert(
        "runtime".to_string(),
        capture_skippy_runtime_diagnostics(output_dir, port, &mut files).await?,
    );
    files.push(write_json_file(
        output_dir,
        "skippy-diagnostics.json",
        &Value::Object(diagnostics),
    )?);
    Ok(files)
}

async fn capture_skippy_api_diagnostics(
    output_dir: &Path,
    port: u16,
    files: &mut Vec<PathBuf>,
) -> Result<Value> {
    let mut endpoints = Map::new();
    for (name, path, file_name) in SKIPPY_DIAGNOSTIC_ENDPOINTS {
        let (file_payload, summary) = match fetch_console_json(port, path).await {
            Ok(value) => (
                value,
                json!({
                    "ok": true,
                    "path": path,
                    "file": file_name,
                }),
            ),
            Err(error) => {
                let payload = json!({
                    "ok": false,
                    "path": path,
                    "file": file_name,
                    "error": error.to_string(),
                });
                (payload.clone(), payload)
            }
        };
        files.push(write_json_file(output_dir, file_name, &file_payload)?);
        endpoints.insert((*name).to_string(), summary);
    }
    Ok(Value::Object(endpoints))
}

async fn capture_skippy_runtime_diagnostics(
    output_dir: &Path,
    port: u16,
    files: &mut Vec<PathBuf>,
) -> Result<Value> {
    let runtime_root = match runtime_root() {
        Ok(root) => root,
        Err(error) => {
            return Ok(json!({
                "ok": false,
                "error": error.to_string(),
            }));
        }
    };
    let instances = match scan_local_instances(&runtime_root, std::process::id()).await {
        Ok(instances) => instances,
        Err(error) => {
            return Ok(json!({
                "ok": false,
                "runtime_root": runtime_root,
                "error": error.to_string(),
            }));
        }
    };
    let local_instances = serde_json::to_value(&instances)?;
    let (selected, selection) = select_runtime_instance(&instances, port)
        .map(|(instance, selection)| (Some(instance), Some(selection)))
        .unwrap_or((None, None));
    let native_log = capture_skippy_native_log(output_dir, selected, files)?;
    Ok(json!({
        "ok": true,
        "runtime_root": runtime_root,
        "local_instances": local_instances,
        "selected_instance": selected,
        "selection": selection,
        "native_log": native_log,
    }))
}

fn select_runtime_instance(
    instances: &[LocalInstanceSnapshot],
    port: u16,
) -> Option<(&LocalInstanceSnapshot, &'static str)> {
    instances
        .iter()
        .find(|instance| instance.api_port == Some(port))
        .map(|instance| (instance, "api_port"))
        .or_else(|| {
            (instances.len() == 1).then(|| {
                (
                    instances.first().expect("len checked"),
                    "single_live_instance",
                )
            })
        })
}

fn capture_skippy_native_log(
    output_dir: &Path,
    instance: Option<&LocalInstanceSnapshot>,
    files: &mut Vec<PathBuf>,
) -> Result<Value> {
    let Some(instance) = instance else {
        return Ok(json!({
            "ok": false,
            "error": "no live local runtime instance matched the requested console port",
        }));
    };
    let source_path = instance
        .runtime_dir
        .join("logs")
        .join(SKIPPY_NATIVE_LOG_FILE);
    if !source_path.is_file() {
        return Ok(json!({
            "ok": false,
            "source_path": source_path,
            "error": "skippy native log was not found",
        }));
    }

    let capture_path = output_dir.join(SKIPPY_NATIVE_LOG_FILE);
    std::fs::copy(&source_path, &capture_path).with_context(|| {
        format!(
            "copy skippy native log from {} to {}",
            source_path.display(),
            capture_path.display()
        )
    })?;
    let bytes = std::fs::metadata(&capture_path)
        .with_context(|| format!("read copied skippy native log {}", capture_path.display()))?
        .len();
    files.push(capture_path.clone());
    Ok(json!({
        "ok": true,
        "source_path": source_path,
        "file": SKIPPY_NATIVE_LOG_FILE,
        "bytes": bytes,
    }))
}

fn write_split_readiness_report(output_dir: &Path, report: &Value) -> Result<PathBuf> {
    write_json_file(output_dir, "split-readiness.json", report)
}

fn write_json_file(output_dir: &Path, file_name: &str, value: &Value) -> Result<PathBuf> {
    let path = output_dir.join(file_name);
    let json = serde_json::to_string_pretty(value)?;
    std::fs::write(&path, json)
        .with_context(|| format!("write diagnostic file {}", path.display()))?;
    Ok(path)
}

async fn fetch_console_json(port: u16, path: &str) -> Result<Value> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let url = format!("http://127.0.0.1:{port}{path}");
    client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("fetch diagnostic endpoint {path} from console port {port}"))?
        .error_for_status()?
        .json::<Value>()
        .await
        .map_err(Into::into)
}

fn unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn split_readiness_lines(report: &Value) -> Vec<String> {
    let model = report["model_ref"].as_str().unwrap_or("unknown");
    let verdict = report["verdict"].as_str().unwrap_or("unknown");
    let participants = report["participant_count"].as_u64().unwrap_or_default();
    let exclusions = report["exclusion_count"].as_u64().unwrap_or_default();

    let mut lines = vec![
        format!("🩺 Split readiness: {verdict}"),
        String::new(),
        format!("Model: {model}"),
        format!("Eligible participants: {participants}"),
        format!("Excluded peers: {exclusions}"),
    ];

    if let Some(items) = report["recommendations"].as_array() {
        let recommendations = items
            .iter()
            .filter_map(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .collect::<Vec<_>>();
        if !recommendations.is_empty() {
            lines.push(String::new());
            lines.push("Recommended next steps:".to_string());
            lines.extend(
                recommendations
                    .into_iter()
                    .map(|item| format!("  - {item}")),
            );
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::{
        SKIPPY_DIAGNOSTIC_ENDPOINTS, capture_skippy_native_log, select_runtime_instance,
        split_readiness_lines, write_split_readiness_report,
    };
    use crate::runtime::instance::LocalInstanceSnapshot;
    use serde_json::json;
    use std::path::PathBuf;

    #[test]
    fn split_readiness_lines_show_waiting_guidance() {
        let report = json!({
            "model_ref": "meshllm/Qwen3-8B-Q4_K_M-layers",
            "verdict": "waiting_for_peers",
            "participant_count": 1,
            "exclusion_count": 1,
            "recommendations": [
                "Start at least one more worker/host with --model meshllm/Qwen3-8B-Q4_K_M-layers --split and join it to this mesh."
            ]
        });

        let lines = split_readiness_lines(&report);

        assert!(lines.iter().any(|line| line.contains("waiting_for_peers")));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Eligible participants: 1"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("--model meshllm/Qwen3-8B-Q4_K_M-layers"))
        );
    }

    #[test]
    fn split_readiness_capture_writes_report_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let report = json!({"verdict": "ready"});

        let path = write_split_readiness_report(dir.path(), &report).expect("write report");

        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("split-readiness.json")
        );
        let written = std::fs::read_to_string(path).expect("read report");
        assert!(written.contains("\"verdict\": \"ready\""));
    }

    #[test]
    fn split_doctor_captures_plugin_startup_surfaces() {
        let paths = SKIPPY_DIAGNOSTIC_ENDPOINTS
            .iter()
            .map(|(_, path, _)| *path)
            .collect::<Vec<_>>();

        assert!(paths.contains(&"/api/runtime/endpoints"));
        assert!(paths.contains(&"/api/plugins"));
        assert!(paths.contains(&"/api/plugins/providers"));
    }

    #[test]
    fn select_runtime_instance_prefers_matching_console_port() {
        let first = instance_snapshot(100, Some(3131), "/tmp/mesh-100");
        let second = instance_snapshot(200, Some(3145), "/tmp/mesh-200");
        let instances = vec![first, second];

        let (selected, selection) =
            select_runtime_instance(&instances, 3145).expect("selected instance");

        assert_eq!(selected.pid, 200);
        assert_eq!(selection, "api_port");
    }

    #[test]
    fn select_runtime_instance_falls_back_to_single_live_instance() {
        let instances = vec![instance_snapshot(100, None, "/tmp/mesh-100")];

        let (selected, selection) =
            select_runtime_instance(&instances, 3131).expect("selected instance");

        assert_eq!(selected.pid, 100);
        assert_eq!(selection, "single_live_instance");
    }

    #[test]
    fn skippy_native_log_capture_copies_selected_instance_log() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime_dir = dir.path().join("runtime").join("123");
        let log_dir = runtime_dir.join("logs");
        std::fs::create_dir_all(&log_dir).expect("create logs");
        std::fs::write(log_dir.join("skippy-native.log"), "native line\n").expect("write log");
        let output_dir = dir.path().join("doctor");
        std::fs::create_dir_all(&output_dir).expect("create output");
        let instance = instance_snapshot(123, Some(3131), runtime_dir);
        let mut files = Vec::new();

        let payload =
            capture_skippy_native_log(&output_dir, Some(&instance), &mut files).expect("capture");

        assert_eq!(payload["ok"], true);
        assert_eq!(payload["bytes"], 12);
        assert_eq!(
            std::fs::read_to_string(output_dir.join("skippy-native.log")).expect("read captured"),
            "native line\n"
        );
        assert_eq!(files, vec![output_dir.join("skippy-native.log")]);
    }

    fn instance_snapshot(
        pid: u32,
        api_port: Option<u16>,
        runtime_dir: impl Into<PathBuf>,
    ) -> LocalInstanceSnapshot {
        LocalInstanceSnapshot {
            pid,
            api_port,
            version: Some("0.99.0-test".to_string()),
            started_at_unix: 1_700_000_000,
            runtime_dir: runtime_dir.into(),
            is_self: false,
        }
    }
}
