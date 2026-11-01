//! Vendor-compatible projections of the Mesh discovery catalog.
//!
//! Rich metadata remains on `/mesh/v1/models`; these projections deliberately
//! contain only fields belonging to the selected compatibility protocol.

use mesh_llm_types::models::discovery::MeshModel;
use serde_json::{Value, json};

#[derive(Default)]
struct AnthropicPage {
    limit: Option<usize>,
    before: Option<String>,
    after: Option<String>,
}

/// Project list or exact-ID lookup from the same confirmed LLM inventory.
pub(crate) fn models_response(models: &[MeshModel], path: &str, anthropic: bool) -> (u16, Value) {
    let (path, query) = path.split_once('?').unwrap_or((path, ""));
    let mut models = models
        .iter()
        .filter(|model| model.is_available_llm())
        .collect::<Vec<_>>();
    models.sort_by(|left, right| left.id.cmp(&right.id));
    models.dedup_by(|left, right| left.id == right.id);
    if let Some(encoded_id) = path
        .strip_prefix("/v1/models/")
        .or_else(|| path.strip_prefix("/models/"))
    {
        return model_detail(&models, encoded_id, query, anthropic);
    }
    if anthropic {
        anthropic_list(&models, query)
    } else if query.is_empty() {
        let data = models
            .iter()
            .map(|model| model_json(model, false))
            .collect::<Vec<_>>();
        (200, json!({"object": "list", "data": data}))
    } else {
        error_response(400, "model list query parameters are not supported", false)
    }
}

fn model_detail(
    models: &[&MeshModel],
    encoded_id: &str,
    query: &str,
    anthropic: bool,
) -> (u16, Value) {
    let Ok(id) = decode_model_id(encoded_id) else {
        return error_response(400, "invalid model ID", anthropic);
    };
    if !query.is_empty() {
        return error_response(
            400,
            "model lookup does not accept query parameters",
            anthropic,
        );
    }
    match models.iter().find(|model| model.id == id) {
        Some(model) => (200, model_json(model, anthropic)),
        None => error_response(404, "model not found", anthropic),
    }
}

fn model_json(model: &MeshModel, anthropic: bool) -> Value {
    if anthropic {
        json!({
            "id": model.id,
            "type": "model",
            "display_name": model.display_name,
            "created_at": "1970-01-01T00:00:00Z",
            "lifecycle": "active",
        })
    } else {
        json!({
            "id": model.id,
            "object": "model",
            "created": 0,
            "owned_by": "mesh-llm",
        })
    }
}

fn anthropic_list(models: &[&MeshModel], query: &str) -> (u16, Value) {
    let page = match parse_page(query) {
        Ok(page) => page,
        Err(message) => return error_response(400, &message, true),
    };
    let limit = page.limit.unwrap_or(20);
    let (start, end) = match page_bounds(models, &page, limit) {
        Ok(bounds) => bounds,
        Err(message) => return error_response(400, &message, true),
    };
    let selected = &models[start..end];
    let data = selected
        .iter()
        .map(|model| model_json(model, true))
        .collect::<Vec<_>>();
    let has_more = if page.before.is_some() {
        start > 0
    } else {
        end < models.len()
    };
    (
        200,
        json!({
            "data": data,
            "has_more": has_more,
            "first_id": selected.first().map(|model| &model.id),
            "last_id": selected.last().map(|model| &model.id),
        }),
    )
}

fn parse_page(query: &str) -> Result<AnthropicPage, String> {
    let mut page = AnthropicPage::default();
    let parameters = crate::models::discovery::parse_query_parameters(query)
        .map_err(|error| error.to_string())?;
    for (key, value) in parameters {
        match key.as_str() {
            "limit" if page.limit.is_none() => {
                let limit = value.parse::<usize>().map_err(|_| "invalid limit")?;
                if !(1..=1000).contains(&limit) {
                    return Err("limit must be between 1 and 1000".into());
                }
                page.limit = Some(limit);
            }
            "before_id" if page.before.is_none() => page.before = Some(value),
            "after_id" if page.after.is_none() => page.after = Some(value),
            _ => return Err(format!("invalid or duplicate model list parameter: {key}")),
        }
    }
    if page.before.is_some() && page.after.is_some() {
        return Err("before_id and after_id cannot be combined".into());
    }
    Ok(page)
}

fn page_bounds(
    models: &[&MeshModel],
    page: &AnthropicPage,
    limit: usize,
) -> Result<(usize, usize), String> {
    if let Some(before) = &page.before {
        let end = models
            .iter()
            .position(|model| &model.id == before)
            .ok_or("unknown before_id")?;
        Ok((end.saturating_sub(limit), end))
    } else {
        let start = match &page.after {
            Some(after) => {
                models
                    .iter()
                    .position(|model| &model.id == after)
                    .ok_or("unknown after_id")?
                    + 1
            }
            None => 0,
        };
        Ok((start, start.saturating_add(limit).min(models.len())))
    }
}

/// Decode a path suffix exactly once, preserving literal `+` in model IDs.
fn decode_model_id(encoded: &str) -> Result<String, ()> {
    let mut decoded = Vec::with_capacity(encoded.len());
    let mut bytes = encoded.bytes();
    while let Some(byte) = bytes.next() {
        if byte == b'%' {
            let high = bytes
                .next()
                .and_then(|value| (value as char).to_digit(16))
                .ok_or(())?;
            let low = bytes
                .next()
                .and_then(|value| (value as char).to_digit(16))
                .ok_or(())?;
            decoded.push((high * 16 + low) as u8);
        } else {
            decoded.push(byte);
        }
    }
    let id = String::from_utf8(decoded).map_err(|_| ())?;
    if id.is_empty() || id.chars().any(char::is_control) {
        return Err(());
    }
    Ok(id)
}

pub(crate) fn error_response(status: u16, message: &str, anthropic: bool) -> (u16, Value) {
    let body = if anthropic {
        let kind = match status {
            404 => "not_found_error",
            500..=599 => "api_error",
            _ => "invalid_request_error",
        };
        json!({"type": "error", "error": {"type": kind, "message": message}})
    } else {
        json!({"error": {"message": message, "type": "invalid_request_error", "param": null,
            "code": if status == 404 { Some("model_not_found") } else { None }}})
    };
    (status, body)
}

#[cfg(test)]
mod tests;
