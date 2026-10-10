use super::boundary::{ModelClass, Truth, object_last_wins, object_rows_last_wins};
use super::{Error, ErrorKind, Family};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Deserialize)]
pub(crate) struct FamilyModel {
    pub(super) family: Family,
    #[serde(rename = "class")]
    pub(super) class: ModelClass,
    pub(super) certification_lanes: Vec<String>,
    #[serde(default)]
    pub(super) mmproj_artifact: Truth,
}

#[derive(Deserialize)]
#[serde(remote = "Self")]
struct PlanDocument {
    #[serde(deserialize_with = "object_rows_last_wins")]
    selected_models: Vec<FamilyModel>,
    required_certification_lanes: Vec<String>,
    #[serde(deserialize_with = "object_last_wins")]
    github_matrix: Matrix,
    #[serde(deserialize_with = "object_rows_last_wins")]
    shards: Vec<Shard>,
}

impl<'de> Deserialize<'de> for PlanDocument {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        super::boundary::ordered_object_last_wins(deserializer, Self::deserialize)
    }
}

#[derive(Deserialize)]
struct Matrix {
    #[serde(deserialize_with = "object_rows_last_wins")]
    include: Vec<MatrixRow>,
}

#[derive(Deserialize)]
struct MatrixRow {
    shard_index: u64,
    families: Family,
}

#[derive(Deserialize)]
struct Shard {
    shard_index: u64,
    families: Vec<Family>,
}

pub(crate) struct SourceFamilyPlan {
    pub(super) models: BTreeMap<Family, FamilyModel>,
    source_matrix: Vec<Value>,
    canonical_bytes: Vec<u8>,
}

impl SourceFamilyPlan {
    /// Read only the controller's legacy validation projection. Never regenerate
    /// or serialize this as a replacement for the source-owned canonical plan.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, Error> {
        if bytes
            .iter()
            .copied()
            .find(|byte| !byte.is_ascii_whitespace())
            != Some(b'{')
        {
            return Err(Error::new(ErrorKind::Json, "expected a source plan object"));
        }
        let plan: PlanDocument = serde_json::from_slice(bytes)?;
        let model_count = plan.selected_models.len();
        let models: BTreeMap<_, _> = plan
            .selected_models
            .into_iter()
            .map(|model| (model.family.clone(), model))
            .collect();
        if !(1..=256).contains(&models.len()) || models.len() != model_count {
            return Err(Error::new(
                ErrorKind::Plan,
                "family matrix must contain 1..256 unique families",
            ));
        }
        let rows = plan.github_matrix.include;
        let families: BTreeSet<_> = rows.iter().map(|row| &row.families).collect();
        if rows.len() != models.len() || families != models.keys().collect() {
            return Err(Error::new(
                ErrorKind::Plan,
                "exactly one matrix job is required per family",
            ));
        }
        let indexes: BTreeSet<_> = rows.iter().map(|row| row.shard_index).collect();
        if indexes.len() != rows.len() {
            return Err(Error::new(ErrorKind::Plan, "duplicate shard index"));
        }
        for row in rows {
            let mut shards = plan
                .shards
                .iter()
                .filter(|shard| shard.shard_index == row.shard_index);
            match (shards.next(), shards.next()) {
                (Some(shard), None)
                    if shard.families.as_slice() == std::slice::from_ref(&row.families) => {}
                _ => {
                    return Err(Error::new(
                        ErrorKind::Plan,
                        "matrix and shard membership disagree",
                    ));
                }
            }
        }
        let lanes: BTreeSet<_> = plan
            .required_certification_lanes
            .iter()
            .map(String::as_str)
            .collect();
        if lanes != BTreeSet::from(["single-step", "chain", "state-handoff"]) {
            return Err(Error::new(
                ErrorKind::Plan,
                "required lane contract changed",
            ));
        }
        let source: Value = serde_json::from_slice(bytes)?;
        let source_matrix = source["github_matrix"]["include"]
            .as_array()
            .ok_or_else(|| Error::new(ErrorKind::Plan, "source matrix rows missing"))?
            .clone();
        Ok(Self {
            models,
            source_matrix,
            canonical_bytes: bytes.to_vec(),
        })
    }

    /// Retain source rows and order, overlaying only controller-owned placement.
    pub(super) fn retry_matrix(&self, families: &BTreeSet<Family>) -> Result<Value, Error> {
        for family in families {
            self.model(family)?;
        }
        if families.is_empty() {
            return Ok(serde_json::json!({"include": []}));
        }
        let placements = super::placement::by_family(&self.canonical_bytes)
            .map_err(|error| Error::new(ErrorKind::Plan, error))?;
        let mut include = Vec::new();
        for source in &self.source_matrix {
            let name = source["families"]
                .as_str()
                .ok_or_else(|| Error::new(ErrorKind::Plan, "source row family missing"))?;
            if !families.iter().any(|family| family.as_str() == name) {
                continue;
            }
            let placement = placements
                .get(name)
                .and_then(Value::as_object)
                .ok_or_else(|| Error::new(ErrorKind::Plan, "retry placement family missing"))?;
            let mut row = source.clone();
            let fields = row
                .as_object_mut()
                .ok_or_else(|| Error::new(ErrorKind::Plan, "retry row must be an object"))?;
            for key in super::placement::PLACEMENT_FIELDS {
                fields.remove(key);
            }
            fields.extend(
                placement
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
            include.push(row);
        }
        if include.len() != families.len() {
            return Err(Error::new(
                ErrorKind::Plan,
                "retry matrix membership changed",
            ));
        }
        Ok(serde_json::json!({"include": include}))
    }

    pub(super) fn model(&self, family: &Family) -> Result<&FamilyModel, Error> {
        self.models
            .get(family)
            .ok_or_else(|| Error::new(ErrorKind::UnplannedFamily, "unplanned family"))
    }
}

#[cfg(test)]
#[path = "plan_retry_tests.rs"]
mod retry_tests;
