//! Validated catalog queries shared by HTTP and in-process consumers.

use super::{MeshModel, MeshModelKind, MeshModelPage, MeshModelQuery, collect};
use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const DEFAULT_LIMIT: usize = 100;
const MAX_LIMIT: usize = 500;

/// Select a bounded page from one collected snapshot. Cursors expire when the
/// matching snapshot changes, rather than silently skipping changed records.
pub fn query_models(mut models: Vec<MeshModel>, query: MeshModelQuery) -> Result<MeshModelPage> {
    let limit = query.limit.unwrap_or(DEFAULT_LIMIT);
    ensure!(
        (1..=MAX_LIMIT).contains(&limit),
        "limit must be between 1 and {MAX_LIMIT}"
    );
    ensure!(
        query.kind != Some(MeshModelKind::Unknown),
        "unknown model kind"
    );
    models.retain(|model| matches_query(model, &query));
    models.sort_by(|a, b| a.id.cmp(&b.id));
    let fingerprint = snapshot_fingerprint(&models, &query)?;
    let start = match query.cursor.as_deref() {
        Some(cursor) => cursor_offset(cursor, &fingerprint, models.len())?,
        None => 0,
    };
    let end = start.saturating_add(limit).min(models.len());
    let next_cursor = (end < models.len()).then(|| format!("{fingerprint}:{end}"));
    Ok(MeshModelPage {
        data: models.into_iter().skip(start).take(limit).collect(),
        next_cursor,
    })
}

fn matches_query(model: &MeshModel, query: &MeshModelQuery) -> bool {
    if query.kind == Some(MeshModelKind::Llm)
        && query
            .workload_class
            .is_some_and(|class| class != super::ModelWorkloadClass::CausalGeneration)
    {
        return false;
    }
    let class = query.workload_class.or_else(|| {
        (query.kind == Some(MeshModelKind::Llm))
            .then_some(super::ModelWorkloadClass::CausalGeneration)
    });
    let providers: Vec<_> = model
        .providers
        .iter()
        .filter(|provider| class.is_none_or(|class| provider.workload_class == Some(class)))
        .collect();
    query.kind.is_none_or(|kind| model.kind == kind)
        && !providers.is_empty()
        && query.available.is_none_or(|available| {
            providers.iter().any(|provider| provider.available) == available
        })
}

fn snapshot_fingerprint(models: &[MeshModel], query: &MeshModelQuery) -> Result<String> {
    let bytes = serde_json::to_vec(&(models, query.workload_class, query.kind, query.available))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn cursor_offset(cursor: &str, fingerprint: &str, length: usize) -> Result<usize> {
    ensure!(cursor.len() <= 90, "invalid cursor");
    let (snapshot, offset) = cursor.split_once(':').context("invalid cursor")?;
    ensure!(
        snapshot == fingerprint,
        "cursor snapshot expired; restart the model list"
    );
    let offset: usize = offset.parse().context("invalid cursor offset")?;
    ensure!(offset > 0 && offset < length, "invalid cursor offset");
    Ok(offset)
}

/// Resolve an exact public serving ID without interpreting it as a path or ref.
pub fn lookup(models: Vec<MeshModel>, id: &str) -> Result<MeshModel> {
    validate_id(id)?;
    models
        .into_iter()
        .find(|model| model.id == id)
        .context("model not found")
}

fn validate_id(id: &str) -> Result<()> {
    ensure!(
        !id.trim().is_empty() && id.len() <= 4096 && !id.chars().any(char::is_control),
        "invalid model id"
    );
    Ok(())
}

/// Produce Mesh discovery responses on either inference ingress without doing
/// any transport I/O or crossing into the management console.
pub(crate) async fn http_response(
    node: &crate::mesh::Node,
    method: &str,
    path: &str,
) -> Option<(u16, serde_json::Value)> {
    let (path, raw_query) = path.split_once('?').unwrap_or((path, ""));
    if !matches!(path, "/mesh/v1/models" | "/mesh/v1/models/lookup") {
        return None;
    }
    if method != "GET" {
        return Some(error(405, "method not allowed"));
    }
    let params = match parse_params(raw_query) {
        Ok(params) => params,
        Err(err) => return Some(error(400, &err.to_string())),
    };
    let models = match collect(node).await {
        Ok(models) => models,
        Err(err) => {
            tracing::warn!(%err, "mesh model discovery failed");
            return Some(error(503, "model discovery unavailable"));
        }
    };
    if path.ends_with("/lookup") {
        if params.len() != 1 || !params.contains_key("id") {
            return Some(error(400, "lookup requires only id"));
        }
        let id = &params["id"];
        if let Err(err) = validate_id(id) {
            return Some(error(400, &err.to_string()));
        }
        return Some(match lookup(models, id) {
            Ok(model) => (200, serde_json::json!(model)),
            Err(_) => error(404, "model not found"),
        });
    }
    Some(
        match parse_query(params).and_then(|query| query_models(models, query)) {
            Ok(page) => (200, serde_json::json!(page)),
            Err(err) => error(400, &err.to_string()),
        },
    )
}

pub(crate) fn parse_params(raw: &str) -> Result<BTreeMap<String, String>> {
    ensure!(raw.len() <= 8192, "query is too long");
    let bytes = raw.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut offset = 0;
    while offset < bytes.len() {
        if bytes[offset] == b'%' {
            let pair = bytes
                .get(offset + 1..offset + 3)
                .context("invalid percent encoding")?;
            ensure!(
                pair.iter().all(u8::is_ascii_hexdigit),
                "invalid percent encoding"
            );
            let pair = std::str::from_utf8(pair).context("invalid percent encoding")?;
            decoded.push(u8::from_str_radix(pair, 16).context("invalid percent encoding")?);
            offset += 3;
        } else {
            decoded.push(bytes[offset]);
            offset += 1;
        }
    }
    // form_urlencoded otherwise substitutes U+FFFD for malformed UTF-8,
    // potentially turning an invalid lookup into another model's exact ID.
    std::str::from_utf8(&decoded).context("invalid UTF-8 in query")?;
    let mut params = BTreeMap::new();
    for (name, value) in url::form_urlencoded::parse(bytes) {
        ensure!(
            !params.contains_key(name.as_ref()),
            "duplicate query parameter"
        );
        params.insert(name.into_owned(), value.into_owned());
    }
    Ok(params)
}

fn parse_query(params: BTreeMap<String, String>) -> Result<MeshModelQuery> {
    let mut values = serde_json::Map::new();
    for (name, value) in params {
        let value = match name.as_str() {
            "workload_class" => {
                ensure!(
                    matches!(
                        value.as_str(),
                        "causal_generation"
                            | "embedding"
                            | "rerank"
                            | "encoder_decoder"
                            | "speech_synthesis"
                            | "decision"
                            | "speech_recognition"
                            | "document_ocr"
                            | "unknown"
                    ),
                    "invalid workload_class"
                );
                serde_json::Value::String(value)
            }
            "kind" => {
                ensure!(
                    matches!(value.as_str(), "llm" | "model" | "virtual"),
                    "invalid kind"
                );
                serde_json::Value::String(value)
            }
            "available" => {
                serde_json::Value::Bool(value.parse().context("available must be true or false")?)
            }
            "limit" => serde_json::json!(value.parse::<usize>().context("invalid limit")?),
            "cursor" => serde_json::Value::String(value),
            _ => bail!("unknown query parameter"),
        };
        values.insert(name, value);
    }
    serde_json::from_value(serde_json::Value::Object(values)).context("invalid model query")
}

fn error(status: u16, message: &str) -> (u16, serde_json::Value) {
    (status, serde_json::json!({"error": {"message": message}}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_decodes_reserved_characters_once_and_rejects_ambiguous_queries() {
        let params = parse_params("id=org%2Fmodel%40revision%23profile%252F").unwrap();
        assert_eq!(params["id"], "org/model@revision#profile%2F");
        assert!(parse_params("id=a&id=b").is_err());
        assert!(parse_params("id=%XY").is_err());
        assert!(parse_params("id=%FF").is_err());
        assert!(parse_params("id=%C3%28").is_err());
        assert_eq!(parse_params("id=%EF%BF%BD").unwrap()["id"], "\u{fffd}");
        assert!(validate_id("").is_err());
        assert!(validate_id("a\0b").is_err());
    }

    #[test]
    fn filters_are_strict_and_native_query_fields_are_typed() {
        for raw in [
            "kind=chat",
            "available=yes",
            "limit=-1",
            "workload_class=not_a_workload",
            "typo=value",
        ] {
            assert!(parse_query(parse_params(raw).unwrap()).is_err(), "{raw}");
        }
        let query = parse_query(
            parse_params("workload_class=document_ocr&available=true&limit=5").unwrap(),
        )
        .unwrap();
        assert_eq!(
            query.workload_class,
            Some(crate::mesh::ModelWorkloadClass::DocumentOcr)
        );
        assert_eq!(query.available, Some(true));
        assert_eq!(query.limit, Some(5));
    }
}
