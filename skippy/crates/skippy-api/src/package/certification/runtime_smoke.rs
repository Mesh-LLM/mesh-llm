use super::{
    CertificationGate, CertificationGateStatus, SkippyCertificationRequest, StagePackageInfo,
};
use reqwest::StatusCode;
use serde_json::json;
use std::time::Duration;
const RUNTIME_SMOKE_TIMEOUT: Duration = Duration::from_secs(120);

pub(super) async fn runtime_smoke_gates(
    request: &SkippyCertificationRequest,
    package: &StagePackageInfo,
) -> Vec<CertificationGate> {
    if request.package_only {
        return required_runtime_gate_names()
            .iter()
            .map(|name| CertificationGate {
                name: (*name).to_string(),
                status: CertificationGateStatus::NotRequired,
                details: Some("package-only certification requested".to_string()),
            })
            .collect();
    }

    let Some(api_base) = request.api_base.as_deref() else {
        return required_runtime_gate_names()
            .iter()
            .map(|name| CertificationGate {
                name: (*name).to_string(),
                status: CertificationGateStatus::Incomplete,
                details: Some("pass --api-base to run runtime OpenAI smoke gates".to_string()),
            })
            .collect();
    };

    let client = match reqwest::Client::builder()
        .timeout(RUNTIME_SMOKE_TIMEOUT)
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return required_runtime_gate_names()
                .iter()
                .map(|name| failed_gate(name, &error))
                .collect();
        }
    };
    vec![
        smoke_v1_models(&client, api_base, &package.package_ref).await,
        smoke_chat_completions(&client, api_base, package, request).await,
        smoke_responses(&client, api_base, package, request).await,
    ]
}

async fn smoke_v1_models(
    client: &reqwest::Client,
    api_base: &str,
    served_model_id: &str,
) -> CertificationGate {
    let url = format!("{}/v1/models", api_base.trim_end_matches('/'));
    match client.get(url).send().await {
        Ok(response) if response.status() == StatusCode::OK => {
            match response.json::<serde_json::Value>().await {
                Ok(value) if models_response_contains(&value, served_model_id) => {
                    CertificationGate {
                        name: "v1_models".to_string(),
                        status: CertificationGateStatus::Passed,
                        details: None,
                    }
                }
                Ok(_) => CertificationGate {
                    name: "v1_models".to_string(),
                    status: CertificationGateStatus::Failed,
                    details: Some(format!(
                        "model {served_model_id:?} was not present in /v1/models"
                    )),
                },
                Err(error) => failed_gate("v1_models", error),
            }
        }
        Ok(response) => failed_gate_message("v1_models", format!("HTTP {}", response.status())),
        Err(error) => failed_gate("v1_models", error),
    }
}

pub(super) async fn smoke_chat_completions(
    client: &reqwest::Client,
    api_base: &str,
    package: &StagePackageInfo,
    request: &SkippyCertificationRequest,
) -> CertificationGate {
    let url = format!("{}/v1/chat/completions", api_base.trim_end_matches('/'));
    let body = json!({
        "model": package.package_ref,
        "messages": [{ "role": "user", "content": request.prompt }],
        "max_tokens": request.max_tokens,
        "stream": false
    });
    smoke_post_json(
        client,
        &url,
        body,
        "v1_chat_completions",
        response_has_chat_choice_content,
        "chat completion choice content",
    )
    .await
}

pub(super) async fn smoke_responses(
    client: &reqwest::Client,
    api_base: &str,
    package: &StagePackageInfo,
    request: &SkippyCertificationRequest,
) -> CertificationGate {
    let url = format!("{}/v1/responses", api_base.trim_end_matches('/'));
    let body = json!({
        "model": package.package_ref,
        "input": request.prompt,
        "max_output_tokens": request.max_tokens
    });
    smoke_post_json(
        client,
        &url,
        body,
        "v1_responses",
        response_has_responses_output,
        "Responses output",
    )
    .await
}

async fn smoke_post_json(
    client: &reqwest::Client,
    url: &str,
    body: serde_json::Value,
    name: &str,
    valid_response: fn(&serde_json::Value) -> bool,
    expected: &'static str,
) -> CertificationGate {
    match client.post(url).json(&body).send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<serde_json::Value>().await {
                Ok(value) if valid_response(&value) => CertificationGate {
                    name: name.to_string(),
                    status: CertificationGateStatus::Passed,
                    details: None,
                },
                Ok(_) => failed_gate_message(name, format!("response missing {expected}")),
                Err(error) => failed_gate(name, error),
            }
        }
        Ok(response) => failed_gate_message(name, format!("HTTP {}", response.status())),
        Err(error) => failed_gate(name, error),
    }
}

pub(super) fn response_has_chat_choice_content(value: &serde_json::Value) -> bool {
    value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .is_some_and(|choices| {
            choices.iter().any(|choice| {
                choice
                    .pointer("/message/content")
                    .is_some_and(response_content_has_text)
            })
        })
}

pub(super) fn response_has_responses_output(value: &serde_json::Value) -> bool {
    value
        .get("output_text")
        .and_then(|output_text| output_text.as_str())
        .is_some_and(|output_text| !output_text.trim().is_empty())
        || value
            .get("output")
            .and_then(|output| output.as_array())
            .is_some_and(|items| {
                items.iter().any(|item| {
                    item.get("content")
                        .and_then(|content| content.as_array())
                        .is_some_and(|content| {
                            content.iter().any(|part| {
                                part.get("type").and_then(|kind| kind.as_str())
                                    == Some("output_text")
                                    && part
                                        .get("text")
                                        .and_then(|text| text.as_str())
                                        .is_some_and(|text| !text.trim().is_empty())
                            })
                        })
                })
            })
}

fn response_content_has_text(value: &serde_json::Value) -> bool {
    value.as_str().is_some_and(|text| !text.trim().is_empty())
        || value.as_array().is_some_and(|parts| {
            parts.iter().any(|part| {
                part.as_str().is_some_and(|text| !text.trim().is_empty())
                    || part
                        .get("text")
                        .and_then(|text| text.as_str())
                        .is_some_and(|text| !text.trim().is_empty())
            })
        })
}

pub(super) fn models_response_contains(value: &serde_json::Value, model_id: &str) -> bool {
    value
        .get("data")
        .and_then(|data| data.as_array())
        .is_some_and(|models| {
            models
                .iter()
                .any(|model| model.get("id").and_then(|id| id.as_str()) == Some(model_id))
        })
}

fn required_runtime_gate_names() -> &'static [&'static str] {
    &["v1_models", "v1_chat_completions", "v1_responses"]
}

fn failed_gate(name: &str, error: impl std::fmt::Display) -> CertificationGate {
    failed_gate_message(name, error.to_string())
}

fn failed_gate_message(name: &str, details: String) -> CertificationGate {
    CertificationGate {
        name: name.to_string(),
        status: CertificationGateStatus::Failed,
        details: Some(details),
    }
}
