use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::cli::DoctorCommand;

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
    if let Some(output_dir) = output_dir {
        write_split_readiness_report(output_dir, &report)?;
    }
    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        for line in split_readiness_lines(&report) {
            println!("{line}");
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

fn write_split_readiness_report(output_dir: &Path, report: &serde_json::Value) -> Result<PathBuf> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create split doctor output dir {}", output_dir.display()))?;
    let path = output_dir.join("split-readiness.json");
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(&path, json)
        .with_context(|| format!("write split readiness report {}", path.display()))?;
    Ok(path)
}

fn split_readiness_lines(report: &serde_json::Value) -> Vec<String> {
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
    use super::{split_readiness_lines, write_split_readiness_report};
    use serde_json::json;

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
}
