use super::artifact::{self, Artifact};
use super::document::Json;
use super::execution::{self, Execution};
use super::fields::{PlanResult, choice, exact, label, object, string};
use super::policy::{CORE, Profile};
use super::resources::{self, Resources};
use super::text::FamilyString;
use std::collections::{BTreeMap, BTreeSet};

mod projection;

pub(super) const CLASSES: [(&str, &[&str]); 7] = [
    ("causal_generation", &CORE),
    ("embedding", &["embedding-smoke", "embedding-oracle"]),
    ("rerank", &["rerank-smoke", "rerank-oracle"]),
    (
        "encoder_decoder",
        &["encoder-decoder-smoke", "encoder-decoder-oracle"],
    ),
    ("ocr", &["ocr-smoke", "ocr-oracle"]),
    (
        "speech_synthesis",
        &["speech-synthesis-smoke", "speech-synthesis-oracle"],
    ),
    (
        "speech_recognition",
        &["speech-recognition-smoke", "speech-recognition-oracle"],
    ),
];

struct Evidence {
    fixture: FamilyString,
    comparison: FamilyString,
}

impl Evidence {
    fn parse(row: &Json, profile: &str, field: &str) -> PlanResult<Option<Self>> {
        if profile == "workload-oracle" {
            let name = format!("{field}.evidence");
            let evidence = object(row.get("evidence"), &name)?;
            exact(evidence, &["fixture", "comparison"], &name)?;
            Ok(Some(Self {
                fixture: string(evidence.get("fixture"), &format!("{name}.fixture"))?,
                comparison: string(evidence.get("comparison"), &format!("{name}.comparison"))?,
            }))
        } else if row.get("evidence").is_some() {
            Err(format!("{field}.evidence requires workload-oracle"))
        } else {
            Ok(None)
        }
    }
}

pub(super) struct Model {
    pub(super) family: String,
    model_class: String,
    architecture: String,
    profile: String,
    certification_status: String,
    oracle: String,
    certification_lanes: Vec<FamilyString>,
    artifact: Artifact,
    draft_artifact: Option<Artifact>,
    mmproj_artifact: Option<Artifact>,
    execution: Execution,
    pub(super) resources: Resources,
    notes: FamilyString,
    evidence: Option<Evidence>,
    pub(super) manifest_index: usize,
}

pub(super) fn parse_all(
    value: Option<&Json>,
    policy: &BTreeMap<String, Profile>,
) -> PlanResult<Vec<Model>> {
    let rows = value
        .and_then(Json::as_array)
        .filter(|rows| !rows.is_empty())
        .ok_or_else(|| "models must be a non-empty array".to_owned())?;
    let mut parser = ModelParser {
        policy,
        seen: BTreeSet::new(),
    };
    rows.iter()
        .enumerate()
        .map(|(index, row)| parser.parse(row, index))
        .collect()
}

struct ModelParser<'a> {
    policy: &'a BTreeMap<String, Profile>,
    seen: BTreeSet<String>,
}

impl ModelParser<'_> {
    fn parse(&mut self, value: &Json, index: usize) -> PlanResult<Model> {
        let field = format!("models[{index}]");
        let row = object(Some(value), &field)?;
        exact(
            row,
            &[
                "family",
                "class",
                "architecture",
                "profile",
                "artifact",
                "draft_artifact",
                "mmproj_artifact",
                "execution",
                "resources",
                "notes",
                "evidence",
            ],
            &field,
        )?;
        let family = label(row.get("family"), &format!("{field}.family"))?;
        if !self.seen.insert(family.clone()) {
            return Err(format!("duplicate family: {family}"));
        }
        let model_class = choice(
            row.get("class"),
            &format!("{field}.class"),
            &CLASSES.map(|(name, _)| name),
        )?;
        let architecture = label(row.get("architecture"), &format!("{field}.architecture"))?;
        let profile = choice(
            row.get("profile"),
            &format!("{field}.profile"),
            &super::policy::NAMES,
        )?;
        if model_class != "causal_generation"
            && !matches!(profile.as_str(), "workload-smoke" | "workload-oracle")
        {
            return Err(format!(
                "{field}.class {model_class} requires a class-specific workload profile"
            ));
        }
        if model_class == "causal_generation"
            && matches!(profile.as_str(), "workload-smoke" | "workload-oracle")
        {
            return Err(format!(
                "{field}.class causal_generation cannot use workload profiles"
            ));
        }
        let evidence = Evidence::parse(row, &profile, &field)?;
        let artifact = artifact::parse_serving(row.get("artifact"), &format!("{field}.artifact"))?;
        let draft_artifact = row
            .get("draft_artifact")
            .map(|value| artifact::parse_serving(Some(value), &format!("{field}.draft_artifact")))
            .transpose()?;
        let mmproj_artifact = row
            .get("mmproj_artifact")
            .map(|value| artifact::parse(Some(value), &format!("{field}.mmproj_artifact")))
            .transpose()?;
        if mmproj_artifact
            .as_ref()
            .is_some_and(|sidecar| sidecar.files.len() != 1)
        {
            return Err(format!(
                "{field}.mmproj_artifact.files must name exactly one projector GGUF"
            ));
        }
        if matches!(
            model_class.as_str(),
            "ocr" | "speech_synthesis" | "speech_recognition"
        ) && mmproj_artifact.is_none()
        {
            return Err(format!(
                "{field}.class {model_class} requires an mmproj_artifact"
            ));
        }
        if model_class != "causal_generation" && artifact.files.len() != 1 {
            return Err(format!(
                "{field}.class {model_class} requires exactly one target GGUF"
            ));
        }
        let execution = execution::parse(row.get("execution"), &field, &model_class)?;
        let resources = resources::parse(row.get("resources"), &field)?;
        let notes = string(row.get("notes"), &format!("{field}.notes"))?;
        let profile_policy = self
            .policy
            .get(&profile)
            .ok_or_else(|| format!("unknown profile: {profile}"))?;
        let mut certification_lanes = self.lanes(&model_class, &profile)?;
        if model_class == "causal_generation" && !execution.mtp_layers.is_zero() {
            certification_lanes.push("native-mtp-heads".into());
        }
        Ok(Model {
            family,
            model_class,
            architecture,
            profile,
            certification_status: profile_policy.status.clone(),
            oracle: profile_policy.oracle.clone(),
            certification_lanes,
            artifact,
            draft_artifact,
            mmproj_artifact,
            execution,
            resources,
            notes,
            evidence,
            manifest_index: index,
        })
    }

    fn lanes(&self, model_class: &str, profile: &str) -> PlanResult<Vec<FamilyString>> {
        if model_class == "causal_generation" {
            return self
                .policy
                .get(profile)
                .map(|policy| policy.lanes.clone())
                .ok_or_else(|| format!("unknown profile: {profile}"));
        }
        let lanes = CLASSES
            .iter()
            .find(|(name, _)| *name == model_class)
            .map(|(_, lanes)| *lanes)
            .ok_or_else(|| format!("unknown model class: {model_class}"))?;
        Ok(lanes
            .iter()
            .take(if profile == "workload-smoke" { 1 } else { 2 })
            .map(|lane| (*lane).into())
            .collect())
    }
}
