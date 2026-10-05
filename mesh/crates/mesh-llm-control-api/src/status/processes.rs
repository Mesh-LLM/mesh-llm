//! Runtime model/process management payloads with stable process ordering.
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeModelPayload {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub profile: String,
    pub backend: String,
    pub status: String,
    pub port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeProcessPayload {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub profile: String,
    pub backend: String,
    pub status: String,
    pub port: u16,
    pub pid: u32,
    pub slots: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeProcessesPayload {
    pub processes: Vec<RuntimeProcessPayload>,
}

pub fn build_runtime_processes_payload(
    mut local_processes: Vec<RuntimeProcessPayload>,
) -> RuntimeProcessesPayload {
    local_processes.sort_by(|left, right| {
        (
            left.name.to_lowercase(),
            left.instance_id.as_deref().unwrap_or(""),
            left.port,
        )
            .cmp(&(
                right.name.to_lowercase(),
                right.instance_id.as_deref().unwrap_or(""),
                right.port,
            ))
    });
    RuntimeProcessesPayload {
        processes: local_processes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_build_runtime_processes_payload_sorts_processes() {
        let payload = build_runtime_processes_payload(vec![
            RuntimeProcessPayload {
                name: "Zulu".into(),
                instance_id: None,
                backend: "llama".into(),
                status: "ready".into(),
                port: 9444,
                pid: 11,
                slots: 4,
                context_length: None,
                profile: String::new(),
            },
            RuntimeProcessPayload {
                name: "Alpha".into(),
                instance_id: None,
                backend: "llama".into(),
                status: "ready".into(),
                port: 9337,
                pid: 10,
                slots: 4,
                context_length: None,
                profile: String::new(),
            },
        ]);

        assert_eq!(payload.processes.len(), 2);
        assert_eq!(payload.processes[0].name, "Alpha");
        assert_eq!(payload.processes[1].name, "Zulu");
    }

    #[test]
    fn test_runtime_processes_payload_includes_context_length() {
        let payload = build_runtime_processes_payload(vec![
            RuntimeProcessPayload {
                name: "model-a".into(),
                instance_id: None,
                backend: "llama".into(),
                status: "ready".into(),
                port: 9337,
                pid: 10,
                slots: 4,
                context_length: Some(65536),
                profile: String::new(),
            },
            RuntimeProcessPayload {
                name: "model-b".into(),
                instance_id: None,
                backend: "llama".into(),
                status: "ready".into(),
                port: 9444,
                pid: 11,
                slots: 2,
                context_length: None,
                profile: String::new(),
            },
        ]);

        assert_eq!(payload.processes.len(), 2);
        assert_eq!(payload.processes[0].name, "model-a");
        assert_eq!(payload.processes[0].context_length, Some(65536));
        assert_eq!(payload.processes[0].slots, 4);
        assert_eq!(payload.processes[1].context_length, None);

        // Verify serialization includes context_length when present
        let json = serde_json::to_string(&payload).expect("serialize payload");
        assert!(json.contains(r#""context_length":65536"#));
        // Verify context_length is omitted when None (skip_serializing_if)
        let model_b_section: serde_json::Value = serde_json::from_str(&json).expect("parse json");
        let processes = model_b_section["processes"]
            .as_array()
            .expect("processes array");
        assert!(
            processes[1].get("context_length").is_none()
                && processes[1]["context_length"].is_null()
        );
    }
}
