//! Exact local catalog variant update; transport failures must never be interpreted as absent entry.
use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};
pub struct CatalogInput<'a> {
    pub source_repo: &'a str,
    pub source_revision: &'a str,
    pub source_file: &'a str,
    pub target_repo: &'a str,
    pub model_id: &'a str,
    pub layer_count: u64,
}
fn variant_name(file: &str) -> Result<String> {
    let base = file
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("source filename absent"))?;
    let stem = base.replace(".gguf", "");
    let bytes = stem.as_bytes();
    let n = bytes.len();
    let suffix = n >= 15
        && bytes[n - 15] == b'-'
        && bytes[n - 9..n - 5] == *b"-of-"
        && bytes[n - 14..n - 9].iter().all(u8::is_ascii_digit)
        && bytes[n - 5..].iter().all(u8::is_ascii_digit);
    Ok(if suffix { stem[..n - 15].into() } else { stem })
}
/// None means an independently classified EntryNotFound, never auth/transport/malformed JSON fallback.
pub fn project(existing: Option<Value>, input: &CatalogInput<'_>) -> Result<(String, Value)> {
    let plan = super::Plan {
        repo: input.target_repo.into(),
        kind: super::RepositoryKind::Model,
        revision: "main".into(),
        path: input.source_file.into(),
        create_pr: false,
        maximum_attempts: 1,
        expected_parent: None,
    };
    plan.validate()?;
    if input.layer_count == 0
        || input.source_revision.len() != 40
        || !input
            .source_revision
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || input.model_id.len() > 4096
        || input.model_id.chars().any(char::is_control)
    {
        bail!("catalog source/layers/model policy refused");
    }
    let source_plan = super::Plan {
        repo: input.source_repo.into(),
        ..plan
    };
    source_plan.validate()?;
    let entry_path = format!("entries/{}.json", input.source_repo);
    let mut entry = existing.unwrap_or_else(
        || json!({"schema_version":1,"source_repo":input.source_repo,"variants":{}}),
    );
    if !entry.is_object()
        || entry
            .get("source_repo")
            .is_some_and(|v| v != input.source_repo)
    {
        bail!("catalog source identity refused");
    }
    if entry.get("variants").is_none() {
        entry["variants"] = json!({});
    }
    let name = variant_name(input.source_file)?;
    let package = json!({"type":"layer-package","repo":input.target_repo,"layer_count":input.layer_count,"source_revision":input.source_revision});
    let source =
        json!({"repo":input.source_repo,"file":input.source_file,"revision":input.source_revision});
    let fresh = json!({"source":source,"curated":{"name":name,"size":format!("{} layers",input.layer_count),"description":format!("Layer package for {}",input.model_id)},"packages":[package]});
    let variants = &mut entry["variants"];
    if let Some(map) = variants.as_object_mut() {
        if let Some(row) = map.get_mut(&name) {
            update(row, &source, &package, input.target_repo)?;
        } else {
            map.insert(name, fresh);
        }
    } else if let Some(rows) = variants.as_array_mut() {
        if let Some(row) = rows.iter_mut().find(|row| row["curated"]["name"] == name) {
            update(row, &source, &package, input.target_repo)?;
        } else {
            rows.push(fresh);
        }
    } else {
        bail!("catalog variants must be object or array");
    }
    if serde_json::to_vec(&entry)?.len() > 1024 * 1024 {
        bail!("catalog entry exceeds publication bound");
    }
    Ok((entry_path, entry))
}
fn update(row: &mut Value, source: &Value, package: &Value, target: &str) -> Result<()> {
    if !row.is_object() {
        bail!("catalog variant must be object");
    }
    if row.get("packages").is_none() {
        row["packages"] = json!([]);
    }
    let packages = row["packages"]
        .as_array_mut()
        .ok_or_else(|| anyhow!("catalog packages must be array"))?;
    if packages.iter().any(|p| !p.is_object()) {
        bail!("catalog package row malformed");
    }
    packages.retain(|p| p["repo"] != target);
    packages.push(package.clone());
    row["source"] = source.clone();
    Ok(())
}
