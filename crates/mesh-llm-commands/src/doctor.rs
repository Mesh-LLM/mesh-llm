//! `mesh-llm doctor network` handler: fetches the console's network
//! diagnostics report and renders it for humans or as JSON.

use anyhow::{Context, Result};
use serde_json::Value;

pub async fn run_network_doctor(port: u16, json_output: bool) -> Result<()> {
    let report = fetch_network_report(port).await?;
    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    for line in network_report_lines(&report) {
        println!("{line}");
    }
    Ok(())
}

async fn fetch_network_report(port: u16) -> Result<Value> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let url = format!("http://127.0.0.1:{port}/api/diagnostics/network");
    client
        .get(&url)
        .send()
        .await
        .with_context(|| {
            format!("Can't connect to mesh-llm console on port {port}. Is it running?")
        })?
        .error_for_status()?
        .json()
        .await
        .map_err(Into::into)
}

fn bool_display(value: &Value, default: bool) -> String {
    value.as_bool().unwrap_or(default).to_string()
}

fn network_report_lines(report: &Value) -> Vec<String> {
    let node_id = report["node_id"].as_str().unwrap_or("unknown");
    let advertisement = &report["advertisement"];
    let verdict = advertisement["verdict"].as_str().unwrap_or("unknown");
    let summary = advertisement["summary"].as_str().unwrap_or("");
    let raw_stun = bool_display(&advertisement["raw_stun_enabled"], false);

    let mut lines = vec![
        format!("🩺 Network advertisement: {verdict}"),
        String::new(),
        format!("Node: {node_id}"),
        format!("Raw STUN discovery: {raw_stun}"),
    ];

    match advertisement["observed_public"].as_str() {
        Some(observed) => lines.push(format!("Observed public address: {observed}")),
        None => lines.push("Observed public address: none".to_string()),
    }

    let candidates = advertisement["public_candidates"].as_array();
    match candidates {
        Some(items) if !items.is_empty() => {
            lines.push(String::new());
            lines.push("Advertised public candidates:".to_string());
            for item in items {
                let addr = item["addr"].as_str().unwrap_or("unknown");
                let verified = item["externally_verified"].as_bool().unwrap_or(false);
                let marker = if verified {
                    "verified by relay probe"
                } else {
                    "UNVERIFIED (locally bound port)"
                };
                lines.push(format!("  - {addr} — {marker}"));
            }
        }
        _ => {
            lines.push("Advertised public candidates: none".to_string());
        }
    }

    if let Some(items) = advertisement["lan_candidates"].as_array()
        && !items.is_empty()
    {
        lines.push(String::new());
        lines.push("LAN candidates:".to_string());
        for item in items {
            let addr = item.as_str().unwrap_or("unknown");
            lines.push(format!("  - {addr}"));
        }
    }

    if let Some(items) = report["peers"].as_array()
        && !items.is_empty()
    {
        lines.push(String::new());
        lines.push("Peer paths:".to_string());
        for item in items {
            let short = item["short_node_id"].as_str().unwrap_or("unknown");
            let path = item["path"].as_str().unwrap_or("unknown");
            match item["rtt_ms"].as_u64() {
                Some(rtt) => lines.push(format!("  - {short}: {path} ({rtt} ms)")),
                None => lines.push(format!("  - {short}: {path}")),
            }
        }
    }

    if verdict == "unverified_candidate" {
        lines.push(String::new());
        lines.push(format!("⚠️  {summary}"));
        lines.push(
            "On a port-remapping NAT (e.g. Vast.ai), peers exclude this node from direct \
             splits (stage_path_relay_only). Restarting the node re-runs discovery; check \
             the relay probe log line for the observed address."
                .to_string(),
        );
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::network_report_lines;
    use serde_json::json;

    #[test]
    fn network_report_lines_render_verified_advertisement() {
        let report = json!({
            "node_id": "abc123def456",
            "advertisement": {
                "verdict": "verified",
                "summary": "advertised public address was externally observed",
                "public_candidates": [
                    {"addr": "213.5.72.196:23555", "externally_verified": true}
                ],
                "lan_candidates": ["192.168.1.8:45678"],
                "observed_public": "213.5.72.196:23555",
                "raw_stun_enabled": true
            },
            "peers": [
                {"node_id": "full", "short_node_id": "peer0000", "path": "direct", "rtt_ms": 4}
            ]
        });

        let lines = network_report_lines(&report);

        let text = lines.join("\n");
        assert!(text.contains("Network advertisement: verified"));
        assert!(text.contains("Node: abc123def456"));
        assert!(text.contains("Raw STUN discovery: true"));
        assert!(text.contains("Observed public address: 213.5.72.196:23555"));
        assert!(text.contains("213.5.72.196:23555 — verified by relay probe"));
        assert!(text.contains("192.168.1.8:45678"));
        assert!(text.contains("peer0000: direct (4 ms)"));
        assert!(!text.contains("UNVERIFIED"));
        assert!(!text.contains("⚠️"));
    }

    #[test]
    fn network_report_lines_warn_on_unverified_candidate() {
        // The exact #1300 field shape: advertised tuple carries the bound
        // port and nothing observed it.
        let report = json!({
            "node_id": "abc123def456",
            "advertisement": {
                "verdict": "unverified_candidate",
                "summary": "a advertised public candidate carries the locally bound port",
                "public_candidates": [
                    {"addr": "213.5.72.196:41842", "externally_verified": false}
                ],
                "raw_stun_enabled": true
            },
            "peers": [
                {"node_id": "full", "short_node_id": "peer0000", "path": "relay"}
            ]
        });

        let lines = network_report_lines(&report);

        let text = lines.join("\n");
        assert!(text.contains("Network advertisement: unverified_candidate"));
        assert!(text.contains("Observed public address: none"));
        assert!(text.contains("213.5.72.196:41842 — UNVERIFIED (locally bound port)"));
        assert!(text.contains("peer0000: relay"));
        assert!(text.contains("stage_path_relay_only"));
        assert!(text.contains("⚠️"));
    }

    #[test]
    fn network_report_lines_render_no_public_advertised() {
        let report = json!({
            "node_id": "abc123def456",
            "advertisement": {
                "verdict": "no_public_advertised",
                "summary": "no public IPv4 is advertised",
                "public_candidates": [],
                "raw_stun_enabled": false
            },
            "peers": []
        });

        let lines = network_report_lines(&report);

        let text = lines.join("\n");
        assert!(text.contains("Network advertisement: no_public_advertised"));
        assert!(text.contains("Advertised public candidates: none"));
        assert!(!text.contains("Peer paths:"));
    }
}
