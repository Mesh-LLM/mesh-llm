//! Selected input policies join the existing cadence-authorized model manifest owner.
use super::Request;
use crate::model_registry::{fields, manifest};
use crate::{ci_plan::document::Json, command::DynResult};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::Path,
};

pub(super) struct Target {
    pub(super) label: String,
    pub(super) repo: String,
    pub(super) revision: Option<String>,
    pub(super) includes: Vec<String>,
    pub(super) verified: Option<Json>,
    pub(super) artifact_id: String,
}
pub(super) struct Plan {
    pub(super) targets: Vec<Target>,
    pub(super) missing: Vec<String>,
}
fn document(path: &Path) -> DynResult<Vec<u8>> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("manifest must be a regular file".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("opened manifest must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err("manifest exceeds eight MiB".into());
    }
    Ok(bytes)
}
fn csv(text: &str) -> BTreeSet<&str> {
    text.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}
fn text<'a>(row: &'a Value, field: &str) -> DynResult<&'a str> {
    match row.get(field) {
        None | Some(Value::Null) => Ok(""),
        Some(Value::String(value)) if !value.contains(['\0', '\n', '\r']) => Ok(value),
        _ => Err(format!("candidate {field} must be single-line text").into()),
    }
}
fn repo(value: &str) -> bool {
    let parts: Vec<_> = value.split('/').collect();
    parts.len() == 2
        && parts.iter().all(|part| {
            !part.is_empty()
                && part.as_bytes()[0].is_ascii_alphanumeric()
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        })
}
fn admit_names(names: &[String], exact: bool) -> DynResult<()> {
    if names.is_empty()
        || names.iter().any(|name| {
            name.is_empty()
                || name.starts_with('-')
                || fields::escapes_root(name)
                || name.contains('\\')
                || name.chars().any(char::is_control)
                || (exact && name.contains(['*', '?', '[', ']']))
        })
    {
        return Err("candidate includes must be safe relative names or patterns".into());
    }
    if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
        return Err("candidate includes must be distinct".into());
    }
    Ok(())
}
fn direct(row: &Value, id: &str) -> DynResult<(Json, bool)> {
    let revision = text(row, "revision")?;
    if !revision.is_empty() && !fields::is_lower_hex(revision, 40..=40) {
        return Err("candidate revision must be an immutable forty-hex commit".into());
    }
    let names = match row.get("include") {
        None | Some(Value::Null) => vec!["*.gguf".into()],
        Some(Value::String(value)) => vec![value.clone()],
        Some(Value::Array(values)) => values
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or("include entries must be strings")
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("include must be string or string list".into()),
    };
    admit_names(&names, false)?;
    let pinned = row
        .get("file_integrity")
        .is_some_and(|value| !value.is_null());
    if pinned
        && (revision.is_empty() || names.iter().any(|name| name.contains(['*', '?', '[', ']'])))
    {
        return Err(
            "integrity-bearing candidates require an immutable revision and exact files".into(),
        );
    }
    let repo = text(row, "repo")?;
    let artifact = json!({"id":id,"repo":repo,"revision":revision,"selector":"manual-parity-input","model_ref":format!("{repo}@{revision}"),"cadences":["manual"],"files":names,"file_integrity":row.get("file_integrity"),"urls":names.iter().map(|name| format!("https://huggingface.co/{repo}/resolve/{revision}/{name}")).collect::<Vec<_>>()});
    let view = json!({"manifest_kind":"test-model-artifacts","artifacts":[artifact]});
    Ok((Json::parse(&serde_json::to_vec(&view)?)?, pinned))
}
fn target(row: &Value, registry: &Json, priority: &str) -> DynResult<Option<Target>> {
    let artifact_id = text(row, "artifact_id")?;
    let (view, pinned, id) = if artifact_id.is_empty() {
        if text(row, "repo")?.is_empty() {
            return Ok(None);
        }
        let (view, pinned) = direct(row, "parity-direct")?;
        (view, pinned, "parity-direct")
    } else {
        (registry.clone(), true, artifact_id)
    };
    let artifact = if pinned {
        Some(
            manifest::resolve(
                &view,
                &manifest::Selection {
                    artifact_id: Some(id),
                    cadence: "manual",
                },
            )
            .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let selected = view
        .get("artifacts")
        .and_then(Json::as_array)
        .unwrap()
        .iter()
        .find(|value| value.get("id").and_then(Json::as_str) == Some(id))
        .ok_or("selected artifact missing")?;
    let repo_value = selected
        .get("repo")
        .and_then(Json::as_str)
        .ok_or("artifact repo missing")?;
    if !repo(repo_value) {
        return Err("artifact repo must be an owner/name identifier".into());
    }
    let revision = selected
        .get("revision")
        .and_then(Json::as_str)
        .unwrap_or_default();
    if !revision.is_empty() && !fields::is_lower_hex(revision, 40..=40) {
        return Err("artifact revision must be immutable forty-hex commit".into());
    }
    if pinned && revision.is_empty() {
        return Err("verified artifact revision missing".into());
    }
    let includes: Vec<String> = if let Some(artifact) = artifact {
        artifact.files.into_iter().map(|file| file.name).collect()
    } else {
        selected
            .get("files")
            .and_then(Json::as_array)
            .unwrap()
            .iter()
            .map(|file| file.as_str().unwrap().to_owned())
            .collect()
    };
    admit_names(&includes, pinned)?;
    Ok(Some(Target {
        label: format!(
            "{} / {} ({priority}, {})",
            text(row, "llama_model")?,
            text(row, "family")?,
            text(row, "status")?
        ),
        repo: repo_value.to_owned(),
        revision: (!revision.is_empty()).then(|| revision.to_owned()),
        includes,
        verified: pinned.then_some(view),
        artifact_id: id.to_owned(),
    }))
}
fn priority_lookup(data: &Value) -> DynResult<BTreeMap<(&str, &str), &str>> {
    if !data.is_object() {
        return Err("parity manifest must be an object".into());
    }
    if data
        .get("support_priority")
        .is_some_and(|value| !value.is_object())
    {
        return Err("support_priority must be an object".into());
    }
    let mut lookup = BTreeMap::new();
    for priority in ["p0", "p1", "p2"] {
        if data["support_priority"]
            .get(priority)
            .is_some_and(|value| !value.is_object())
        {
            return Err("priority group must be an object".into());
        }
        for (plural, kind) in [("families", "family"), ("llama_models", "llama_model")] {
            if data["support_priority"][priority]
                .get(plural)
                .is_some_and(|value| !value.is_array())
            {
                return Err("priority membership must be an array".into());
            }
            if let Some(names) = data["support_priority"][priority][plural].as_array() {
                for name in names {
                    lookup.insert(
                        (kind, name.as_str().ok_or("priority names must be strings")?),
                        priority,
                    );
                }
            }
        }
    }
    Ok(lookup)
}
pub(super) fn plan(request: &Request) -> DynResult<Plan> {
    let data: Value = serde_json::from_slice(&document(&request.manifest)?)?;
    let registry = Json::parse(&document(&request.registry)?)?;
    let statuses = csv(&request.statuses);
    let priorities = csv(&request.priorities);
    let lookup = priority_lookup(&data)?;
    let mut ordered = Vec::new();
    let mut missing = Vec::new();
    for row in data["candidates"]
        .as_array()
        .ok_or("manifest requires candidates array")?
    {
        if !row.is_object() {
            return Err("candidate row must be an object".into());
        }
        let status = text(row, "status")?;
        if !statuses.contains(status) {
            continue;
        }
        let family = text(row, "family")?;
        let model = text(row, "llama_model")?;
        let priority = lookup
            .get(&("family", family))
            .or_else(|| lookup.get(&("llama_model", model)))
            .copied()
            .unwrap_or("p2");
        if !priorities.is_empty() && !priorities.contains(priority) {
            continue;
        }
        match target(row, &registry, priority)? {
            Some(target) => ordered.push((
                (
                    priority.to_owned(),
                    status.to_owned(),
                    model.to_owned(),
                    family.to_owned(),
                    target.repo.clone(),
                ),
                target,
            )),
            None => missing.push(format!("{model} / {family} ({priority}, {status})")),
        }
    }
    ordered.sort_by(|(a, _), (b, _)| a.cmp(b));
    missing.sort();
    Ok(Plan {
        targets: ordered.into_iter().map(|(_, target)| target).collect(),
        missing,
    })
}
