use super::document::{self, Json};
use super::failure::Failure;
use super::fields::{PlanResult, exact, valid_label};
use super::model::{self, Model};
use super::policy::{self, CORE};
use super::projection::{ToJson, object};
use super::shard_count::ShardCount;
use super::shards::{GithubMatrix, SelectedFamily, Shard, shard_families};
use super::text::FamilyString;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub(super) struct Plan {
    schema_version: u8,
    generated_by: &'static str,
    manifest: String,
    pub(super) manifest_sha256: String,
    required_certification_lanes: Vec<&'static str>,
    model_class_lanes: BTreeMap<&'static str, Vec<&'static str>>,
    requested_families: Option<String>,
    pub(super) selected_family_count: usize,
    selected_models: Vec<Model>,
    shards: Vec<Shard>,
    pub(super) github_matrix: GithubMatrix,
}

impl ToJson for Plan {
    fn to_json(&self) -> Json {
        object([
            ("schema_version", self.schema_version.to_json()),
            ("generated_by", self.generated_by.to_json()),
            ("manifest", self.manifest.to_json()),
            ("manifest_sha256", self.manifest_sha256.to_json()),
            (
                "required_certification_lanes",
                self.required_certification_lanes.to_json(),
            ),
            ("model_class_lanes", self.model_class_lanes.to_json()),
            ("requested_families", self.requested_families.to_json()),
            (
                "selected_family_count",
                self.selected_family_count.to_json(),
            ),
            ("selected_models", self.selected_models.to_json()),
            ("shards", self.shards.to_json()),
            ("github_matrix", self.github_matrix.to_json()),
        ])
    }
}

pub(super) struct Selection<'a> {
    pub(super) families: &'a FamilyString,
    pub(super) shard_count: &'a ShardCount,
}

pub(super) fn build(
    root: &Path,
    manifest_path: &Path,
    selection: Selection<'_>,
) -> Result<Plan, Failure> {
    let Selection {
        families,
        shard_count,
    } = selection;
    let raw = fs::read(manifest_path)
        .map_err(|error| format!("unable to load {}: {error}", manifest_path.display()))?;
    let text = std::str::from_utf8(&raw).map_err(|error| Failure::Policy(error.to_string()))?;
    let manifest = document::parse(text).map_err(Failure::Policy)?;
    if manifest.as_object().is_none() {
        return Err(format!("{} must contain an object", manifest_path.display()).into());
    }
    if !manifest.get("schema_version").is_some_and(Json::equals_one) {
        return Err(format!("{} has unsupported schema_version", manifest_path.display()).into());
    }
    exact(
        &manifest,
        &["schema_version", "policy", "models"],
        "manifest",
    )?;
    let policy = policy::parse(manifest.get("policy"))?;
    let models = model::parse_all(manifest.get("models"), &policy)?;
    let families = families
        .scalar_text()
        .ok_or_else(|| "--families must contain unique comma-separated family labels".to_owned())?;
    let selected_models = select(models, &families)?;
    if selected_models.is_empty() {
        return Err("family selection produced no models".to_owned().into());
    }
    let projection = selected_models
        .iter()
        .map(|model| SelectedFamily {
            family: &model.family,
            manifest_index: model.manifest_index,
            estimated_model_bytes: model.resources.estimated_model_bytes.clone(),
        })
        .collect::<Vec<_>>();
    let sharded = shard_families(&projection, shard_count.effective(selected_models.len())?)
        .map_err(|error| error.to_string())?;
    let manifest_source = manifest_path
        .canonicalize()
        .ok()
        .and_then(|resolved| {
            resolved.strip_prefix(root).ok().map(|relative| {
                relative
                    .components()
                    .map(|part| part.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/")
            })
        })
        .unwrap_or_else(|| {
            manifest_path
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
        });
    Ok(Plan {
        schema_version: 1,
        generated_by: "scripts/plan-family-battery.py",
        manifest: manifest_source,
        manifest_sha256: hex::encode(Sha256::digest(&raw)),
        required_certification_lanes: CORE.to_vec(),
        model_class_lanes: model::CLASSES
            .into_iter()
            .map(|(name, lanes)| (name, lanes.to_vec()))
            .collect(),
        requested_families: (!families.is_empty()).then(|| families.to_owned()),
        selected_family_count: selected_models.len(),
        selected_models,
        shards: sharded.shards,
        github_matrix: sharded.github_matrix,
    })
}

fn select(models: Vec<Model>, families: &str) -> PlanResult<Vec<Model>> {
    if families.is_empty() {
        return Ok(models);
    }
    let requested = families.split(',').collect::<Vec<_>>();
    let set = requested.iter().copied().collect::<BTreeSet<_>>();
    if requested.iter().any(|family| !valid_label(family)) || requested.len() != set.len() {
        return Err("--families must contain unique comma-separated family labels".into());
    }
    let known = models
        .iter()
        .map(|model| model.family.as_str())
        .collect::<BTreeSet<_>>();
    let unknown = requested
        .iter()
        .filter(|family| !known.contains(**family))
        .copied()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(format!("unknown selected families: {}", unknown.join(", ")));
    }
    Ok(models
        .into_iter()
        .filter(|model| set.contains(model.family.as_str()))
        .collect())
}

pub(super) fn verify(
    root: &Path,
    manifest_path: &Path,
    supplied_path: &Path,
) -> Result<(), Failure> {
    let text = fs::read_to_string(supplied_path).map_err(Failure::io)?;
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let supplied = document::parse(&text).map_err(Failure::Runtime)?;
    if supplied.as_object().is_none() {
        return Err("plan must be an object".to_owned().into());
    }
    let families = match supplied.get("requested_families") {
        None | Some(Json::Null) => FamilyString::from(""),
        Some(Json::String(value)) => value.clone(),
        _ => {
            return Err("plan.requested_families must be a string or null"
                .to_owned()
                .into());
        }
    };
    let shards = supplied
        .get("shards")
        .and_then(Json::as_array)
        .filter(|shards| !shards.is_empty())
        .ok_or_else(|| "plan.shards must be a nonempty list".to_owned())?;
    let count = ShardCount::from(shards.len());
    let canonical = build(
        root,
        manifest_path,
        Selection {
            families: &families,
            shard_count: &count,
        },
    )?;
    let canonical_value = canonical.to_json();
    if !super::equality::equal(&supplied, &canonical_value) {
        return Err(
            "policy plan differs from the canonical manifest and selection"
                .to_owned()
                .into(),
        );
    }
    Ok(())
}
