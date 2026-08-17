use super::{
    InferenceEndpointRoute, PluginCapabilityProvider, PluginEndpointSummary, PluginManager,
    PluginSummary, endpoint_kind_name, endpoint_transport_kind_name, plugin_manifest_overview,
    proto,
};
use anyhow::Result;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use url::Url;

pub(super) const HEALTH_CHECK_INTERVAL_SECS: u64 = 15;
const ENDPOINT_STARTUP_GRACE_SECS: u64 = 30;
const ENDPOINT_FAILURE_THRESHOLD: u32 = 2;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct EndpointHealthRecord {
    pub(super) state: String,
    pub(super) available: bool,
    pub(super) detail: Option<String>,
    pub(super) models: Vec<String>,
}

#[derive(Clone, Debug)]
pub(super) struct EndpointHealthState {
    record: EndpointHealthRecord,
    first_checked_at: Instant,
    consecutive_failures: u32,
}

impl PluginManager {
    pub(super) async fn refresh_plugin_endpoints(&self, plugin_name: &str) -> Result<()> {
        let summary = if let Some(plugin) = self.inner.plugins.get(plugin_name) {
            plugin.summary().await
        } else if let Some(summary) = self.inner.inactive.get(plugin_name) {
            summary.clone()
        } else {
            self.clear_plugin_endpoint_health(plugin_name).await;
            return Ok(());
        };
        self.publish_plugin_summary(&summary);

        let manifest = if let Some(plugin) = self.inner.plugins.get(plugin_name) {
            plugin.manifest_snapshot().await
        } else {
            self.manifest(plugin_name).await.ok().flatten()
        };
        let Some(manifest) = manifest else {
            self.clear_plugin_endpoint_health(plugin_name).await;
            self.publish_plugin_summary(&summary);
            self.publish_plugin_providers(plugin_name, Vec::new());
            return Ok(());
        };

        let now = Instant::now();
        let prefix = format!("{plugin_name}:");
        let previous = self
            .inner
            .endpoint_health
            .lock()
            .await
            .iter()
            .filter_map(|(key, value)| {
                key.strip_prefix(&prefix)
                    .map(|endpoint_id| (endpoint_id.to_string(), value.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let plugin_default = endpoint_record_from_plugin_status(&summary);
        let mut providers = manifest
            .capabilities
            .iter()
            .map(|capability| PluginCapabilityProvider {
                capability: capability.clone(),
                plugin_name: summary.name.clone(),
                plugin_status: summary.status.clone(),
                endpoint_id: None,
                available: plugin_default.available,
                detail: plugin_default.detail.clone(),
            })
            .collect::<Vec<_>>();
        let mut endpoint_states = BTreeMap::new();
        let mut endpoint_summaries = Vec::new();
        for endpoint in &manifest.endpoints {
            let key = endpoint.endpoint_id.clone();
            let health =
                endpoint_health_for_summary(&summary, endpoint, previous.get(&key), now).await;
            for capability in endpoint_declared_capabilities(endpoint) {
                providers.push(PluginCapabilityProvider {
                    capability,
                    plugin_name: summary.name.clone(),
                    plugin_status: summary.status.clone(),
                    endpoint_id: Some(endpoint.endpoint_id.clone()),
                    available: health.record.available,
                    detail: health.record.detail.clone(),
                });
            }
            endpoint_summaries.push(PluginEndpointSummary {
                plugin_name: summary.name.clone(),
                plugin_status: summary.status.clone(),
                endpoint_id: endpoint.endpoint_id.clone(),
                state: health.record.state.clone(),
                available: health.record.available,
                kind: endpoint_kind_name(endpoint.kind).to_string(),
                transport_kind: endpoint_transport_kind_name(endpoint.transport_kind).to_string(),
                protocol: endpoint.protocol.clone(),
                address: endpoint.address.clone(),
                args: endpoint.args.clone(),
                namespace: endpoint.namespace.clone(),
                supports_streaming: endpoint.supports_streaming,
                managed_by_plugin: endpoint.managed_by_plugin,
                detail: health.record.detail.clone(),
                models: health.record.models.clone(),
            });
            endpoint_states.insert(endpoint_key(plugin_name, &key), health);
        }

        self.clear_plugin_endpoint_health(plugin_name).await;
        self.publish_plugin_summary(&summary);
        self.publish_plugin_manifest(plugin_name, Some(plugin_manifest_overview(&manifest)));
        self.publish_plugin_providers(plugin_name, providers);
        for endpoint_summary in endpoint_summaries {
            self.plugin_endpoint_producer(plugin_name, &endpoint_summary.endpoint_id)
                .publish_plugin_endpoint(endpoint_summary);
        }

        let mut registry = self.inner.endpoint_health.lock().await;
        registry.extend(endpoint_states);
        Ok(())
    }

    async fn clear_plugin_endpoint_health(&self, plugin_name: &str) {
        let mut registry = self.inner.endpoint_health.lock().await;
        registry.retain(|key, _| !key.starts_with(&format!("{plugin_name}:")));
        drop(registry);
        self.plugin_summary_producer(plugin_name)
            .clear_plugin_reports(plugin_name);
    }

    pub async fn inference_endpoints(&self) -> Result<Vec<InferenceEndpointRoute>> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.inactive.is_empty() {
            let mut endpoints = self.inner.test_inference_endpoints.lock().await.clone();
            endpoints.sort_by(|a, b| {
                a.plugin_name
                    .cmp(&b.plugin_name)
                    .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
            });
            if !endpoints.is_empty() {
                return Ok(endpoints);
            }
        }
        let endpoint_summaries = self.endpoints().await?;
        let mut endpoints = Vec::new();
        for endpoint in endpoint_summaries {
            if endpoint.kind != "inference" || !endpoint.available {
                continue;
            }
            let Some(address) = endpoint.address.clone() else {
                continue;
            };
            endpoints.push(InferenceEndpointRoute {
                plugin_name: endpoint.plugin_name,
                endpoint_id: endpoint.endpoint_id,
                address,
                models: endpoint.models,
            });
        }
        Ok(endpoints)
    }
}

pub(super) fn endpoint_record_from_plugin_status(summary: &PluginSummary) -> EndpointHealthRecord {
    if !summary.enabled || summary.status == "disabled" {
        return EndpointHealthRecord {
            state: "unavailable".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        };
    }

    match summary.status.as_str() {
        "running" => EndpointHealthRecord {
            state: "healthy".into(),
            available: true,
            detail: None,
            models: Vec::new(),
        },
        "starting" | "restarting" => EndpointHealthRecord {
            state: "starting".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
        "degraded" => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
        _ => EndpointHealthRecord {
            state: "unavailable".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
    }
}

fn endpoint_state_from_plugin_status(summary: &PluginSummary, now: Instant) -> EndpointHealthState {
    EndpointHealthState {
        record: endpoint_record_from_plugin_status(summary),
        first_checked_at: now,
        consecutive_failures: 0,
    }
}

fn endpoint_key(plugin_name: &str, endpoint_id: &str) -> String {
    format!("{plugin_name}:{endpoint_id}")
}

pub(super) fn endpoint_declared_capabilities(endpoint: &proto::EndpointManifest) -> Vec<String> {
    match proto::EndpointKind::try_from(endpoint.kind).unwrap_or(proto::EndpointKind::Unspecified) {
        proto::EndpointKind::Inference => {
            let mut capabilities = vec!["endpoint:inference".into()];
            if let Some(protocol) = endpoint.protocol.as_deref() {
                capabilities.push(format!("endpoint:inference/{protocol}"));
            }
            capabilities
        }
        proto::EndpointKind::Mcp => {
            let mut capabilities = vec!["endpoint:mcp".into()];
            if let Some(namespace) = endpoint.namespace.as_deref() {
                capabilities.push(format!("endpoint:mcp/{namespace}"));
            }
            capabilities
        }
        proto::EndpointKind::Unspecified => Vec::new(),
    }
}

async fn endpoint_health_for_summary(
    summary: &PluginSummary,
    endpoint: &proto::EndpointManifest,
    previous: Option<&EndpointHealthState>,
    now: Instant,
) -> EndpointHealthState {
    if summary.status != "running" {
        return endpoint_state_from_plugin_status(summary, now);
    }

    let probe = probe_endpoint(endpoint)
        .await
        .unwrap_or(EndpointHealthRecord {
            state: "healthy".into(),
            available: true,
            detail: None,
            models: Vec::new(),
        });
    apply_endpoint_probe(previous, probe, now)
}

fn apply_endpoint_probe(
    previous: Option<&EndpointHealthState>,
    probe: EndpointHealthRecord,
    now: Instant,
) -> EndpointHealthState {
    let first_checked_at = previous.map(|state| state.first_checked_at).unwrap_or(now);

    if probe.available {
        return EndpointHealthState {
            record: probe,
            first_checked_at,
            consecutive_failures: 0,
        };
    }

    let failure_streak = previous
        .map(|state| state.consecutive_failures.saturating_add(1))
        .unwrap_or(1);
    let within_startup_grace =
        now.duration_since(first_checked_at) < Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS);
    let was_available = previous
        .map(|state| state.record.available)
        .unwrap_or(false);

    let record = if !was_available && within_startup_grace {
        EndpointHealthRecord {
            state: "starting".into(),
            available: false,
            detail: probe.detail,
            models: Vec::new(),
        }
    } else if was_available && failure_streak < ENDPOINT_FAILURE_THRESHOLD {
        EndpointHealthRecord {
            state: "degraded".into(),
            available: true,
            detail: probe.detail,
            models: Vec::new(),
        }
    } else {
        EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: probe.detail,
            models: Vec::new(),
        }
    };

    EndpointHealthState {
        record,
        first_checked_at,
        consecutive_failures: failure_streak,
    }
}

async fn probe_endpoint(endpoint: &proto::EndpointManifest) -> Option<EndpointHealthRecord> {
    match (
        proto::EndpointKind::try_from(endpoint.kind).unwrap_or(proto::EndpointKind::Unspecified),
        proto::EndpointTransportKind::try_from(endpoint.transport_kind)
            .unwrap_or(proto::EndpointTransportKind::Unspecified),
    ) {
        (proto::EndpointKind::Inference, proto::EndpointTransportKind::EndpointTransportHttp) => {
            let protocol = endpoint.protocol.as_deref().unwrap_or_default();
            if protocol.eq_ignore_ascii_case("openai_compatible") {
                return Some(
                    probe_openai_compatible_http_endpoint(endpoint.address.as_deref()?).await,
                );
            }
            None
        }
        _ => None,
    }
}

async fn probe_openai_compatible_http_endpoint(address: &str) -> EndpointHealthRecord {
    let models_url = match endpoint_models_url(address) {
        Some(url) => url,
        None => {
            return EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some(format!("invalid endpoint address '{address}'")),
                models: Vec::new(),
            };
        }
    };

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            return EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some(format!("build health probe client: {err}")),
                models: Vec::new(),
            };
        }
    };

    match client.get(models_url.clone()).send().await {
        Ok(response) if response.status().is_success() => EndpointHealthRecord {
            state: "healthy".into(),
            available: true,
            detail: Some(format!("GET {} -> {}", models_url, response.status())),
            models: parse_models_response(response).await.unwrap_or_default(),
        },
        Ok(response) => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: Some(format!("GET {} -> {}", models_url, response.status())),
            models: Vec::new(),
        },
        Err(err) => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: Some(format!("GET {} failed: {}", models_url, err)),
            models: Vec::new(),
        },
    }
}

async fn parse_models_response(response: reqwest::Response) -> Result<Vec<String>> {
    let body = response.json::<Value>().await?;
    let models = body
        .get("data")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("id").and_then(|id| id.as_str()))
        .map(|id| id.to_string())
        .collect::<Vec<_>>();
    Ok(models)
}

fn endpoint_models_url(address: &str) -> Option<Url> {
    let mut url = Url::parse(address).ok()?;
    let mut path = url.path().trim_end_matches('/').to_string();
    if path.is_empty() {
        path = "/v1".into();
    }
    if !path.ends_with("/models") {
        if path.ends_with("/v1") || path.ends_with("/api/v1") {
            path.push_str("/models");
        } else {
            path.push_str("/v1/models");
        }
    }
    url.set_path(&path);
    url.set_query(None);
    Some(url)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::PluginWebUiState;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn running_summary() -> PluginSummary {
        PluginSummary {
            name: "demo".into(),
            kind: "external".into(),
            enabled: true,
            status: "running".into(),
            pid: None,
            version: None,
            capabilities: Vec::new(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            web_ui: PluginWebUiState::default(),
            startup: None,
            error: None,
        }
    }

    async fn spawn_fake_models_server(
        responses: Vec<(&'static str, &'static str)>,
    ) -> (String, tokio::task::JoinHandle<()>, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let requests_seen = requests.clone();
        let handle = tokio::spawn(async move {
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 4096];
                let _ = stream.read(&mut buf).await.unwrap();
                requests_seen.fetch_add(1, Ordering::SeqCst);
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                let _ = stream.shutdown().await;
            }
        });
        (format!("http://{addr}/api/v1"), handle, requests)
    }

    #[test]
    fn running_plugin_endpoints_are_healthy() {
        let summary = running_summary();
        assert_eq!(
            endpoint_record_from_plugin_status(&summary),
            EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: Vec::new(),
            }
        );
    }

    #[test]
    fn restarting_plugin_endpoints_are_not_available() {
        let summary = PluginSummary {
            status: "restarting".into(),
            error: Some("timed out".into()),
            ..running_summary()
        };
        assert_eq!(
            endpoint_record_from_plugin_status(&summary),
            EndpointHealthRecord {
                state: "starting".into(),
                available: false,
                detail: Some("timed out".into()),
                models: Vec::new(),
            }
        );
    }

    #[test]
    fn first_probe_failure_stays_in_startup_grace() {
        let now = Instant::now();
        let state = apply_endpoint_probe(
            None,
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("GET /models failed".into()),
                models: Vec::new(),
            },
            now,
        );
        assert_eq!(state.record.state, "starting");
        assert!(!state.record.available);
        assert_eq!(state.consecutive_failures, 1);
    }

    #[test]
    fn healthy_endpoint_degrades_before_becoming_unhealthy() {
        let now = Instant::now();
        let healthy = EndpointHealthState {
            record: EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: vec!["demo".into()],
            },
            first_checked_at: now - Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS + 1),
            consecutive_failures: 0,
        };

        let degraded = apply_endpoint_probe(
            Some(&healthy),
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            now,
        );
        assert_eq!(degraded.record.state, "degraded");
        assert!(degraded.record.available);
        assert_eq!(degraded.consecutive_failures, 1);

        let unhealthy = apply_endpoint_probe(
            Some(&degraded),
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            now + Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS),
        );
        assert_eq!(unhealthy.record.state, "unhealthy");
        assert!(!unhealthy.record.available);
        assert_eq!(unhealthy.consecutive_failures, 2);
    }

    #[test]
    fn unhealthy_endpoint_recovers_immediately_on_success() {
        let now = Instant::now();
        let unhealthy = EndpointHealthState {
            record: EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            first_checked_at: now - Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS + 1),
            consecutive_failures: ENDPOINT_FAILURE_THRESHOLD,
        };

        let recovered = apply_endpoint_probe(
            Some(&unhealthy),
            EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: vec!["demo".into()],
            },
            now,
        );
        assert_eq!(recovered.record.state, "healthy");
        assert!(recovered.record.available);
        assert_eq!(recovered.record.models, vec!["demo".to_string()]);
        assert_eq!(recovered.consecutive_failures, 0);
    }

    #[test]
    fn models_probe_url_extends_openai_v1_base() {
        let url = endpoint_models_url("http://localhost:8000/v1").unwrap();
        assert_eq!(url.as_str(), "http://localhost:8000/v1/models");
    }

    #[test]
    fn models_probe_url_extends_api_v1_base() {
        let url = endpoint_models_url("http://localhost:8000/api/v1").unwrap();
        assert_eq!(url.as_str(), "http://localhost:8000/api/v1/models");
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_extracts_models_from_fake_server() {
        let (address, handle, requests) = spawn_fake_models_server(vec![(
            "200 OK",
            r#"{"data":[{"id":"lemonade-small"},{"id":"lemonade-large"}]}"#,
        )])
        .await;

        let health = probe_openai_compatible_http_endpoint(&address).await;
        assert!(health.available);
        assert_eq!(health.state, "healthy");
        assert_eq!(
            health.models,
            vec!["lemonade-small".to_string(), "lemonade-large".to_string()]
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_marks_503_unavailable() {
        let (address, handle, requests) =
            spawn_fake_models_server(vec![("503 Service Unavailable", r#"{"error":"warming"}"#)])
                .await;

        let health = probe_openai_compatible_http_endpoint(&address).await;
        assert!(!health.available);
        assert_eq!(health.state, "unhealthy");
        assert!(health.models.is_empty());
        assert!(
            health
                .detail
                .as_deref()
                .unwrap_or_default()
                .contains("503 Service Unavailable")
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_recovers_when_fake_server_recovers() {
        let (address, handle, requests) = spawn_fake_models_server(vec![
            ("503 Service Unavailable", r#"{"error":"warming"}"#),
            ("200 OK", r#"{"data":[{"id":"lemonade-recovered"}]}"#),
        ])
        .await;

        let first = probe_openai_compatible_http_endpoint(&address).await;
        assert!(!first.available);
        assert_eq!(first.state, "unhealthy");

        let second = probe_openai_compatible_http_endpoint(&address).await;
        assert!(second.available);
        assert_eq!(second.state, "healthy");
        assert_eq!(second.models, vec!["lemonade-recovered".to_string()]);
        assert_eq!(requests.load(Ordering::SeqCst), 2);

        handle.await.unwrap();
    }

    #[test]
    fn endpoint_declares_inference_capabilities() {
        let endpoint = proto::EndpointManifest {
            endpoint_id: "demo".into(),
            kind: proto::EndpointKind::Inference as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportHttp as i32,
            protocol: Some("openai_compatible".into()),
            address: Some("http://localhost:8000/api/v1".into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: true,
            managed_by_plugin: false,
        };
        assert_eq!(
            endpoint_declared_capabilities(&endpoint),
            vec![
                "endpoint:inference".to_string(),
                "endpoint:inference/openai_compatible".to_string()
            ]
        );
    }
}
