//! Reviewed numeric identity anchors. Inactive preparation grants no authority.
use crate::command::DynResult;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Activation {
    pub schema_version: u32,
    pub repository: String,
    pub active: bool,
    pub repository_id: Option<u64>,
    pub reviewer_team_id: Option<u64>,
    pub administration_environment_id: Option<u64>,
    pub observer_app_id: Option<u64>,
    pub observer_installation_id: Option<u64>,
    pub publisher_app_id: Option<u64>,
    pub publisher_installation_id: Option<u64>,
    pub native_ruleset_ids: Vec<u64>,
    pub human_ids: BTreeMap<String, Option<u64>>,
}

#[derive(Clone, Debug)]
pub(crate) struct Anchors {
    pub repository_id: u64,
    pub reviewer_team_id: u64,
    pub administration_environment_id: u64,
    pub observer_app_id: u64,
    pub observer_installation_id: u64,
    pub publisher_app_id: u64,
    pub publisher_installation_id: u64,
    pub native_ruleset_ids: Vec<u64>,
    pub human_ids: BTreeMap<String, u64>,
}

pub(crate) fn from_bytes(bytes: &[u8]) -> DynResult<Activation> {
    if bytes.len() > 64 * 1024 {
        return Err("governance activation exceeds 64 KiB".into());
    }
    let activation: Activation = serde_json::from_slice(bytes).map_err(|error| {
        format!(
            "invalid governance activation schema at line {}, column {}",
            error.line(),
            error.column()
        )
    })?;
    activation.validate()?;
    Ok(activation)
}

impl Activation {
    fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1 || self.repository != "Mesh-LLM/mesh-llm" {
            return Err("unsupported governance activation identity or schema".into());
        }
        let ids = [
            self.repository_id,
            self.reviewer_team_id,
            self.administration_environment_id,
            self.observer_app_id,
            self.observer_installation_id,
            self.publisher_app_id,
            self.publisher_installation_id,
        ];
        if ids.into_iter().flatten().any(|id| id == 0)
            || self.native_ruleset_ids.len() > 100
            || self.native_ruleset_ids.contains(&0)
            || self
                .native_ruleset_ids
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != self.native_ruleset_ids.len()
        {
            return Err(
                "governance activation IDs must be positive, bounded and unique within each roster"
                    .into(),
            );
        }
        Ok(())
    }

    pub(crate) fn anchors(&self) -> DynResult<Anchors> {
        self.validate()?;
        if !self.active {
            return Err("CI governance activation is inactive; reviewed identity anchors and external protections remain pending".into());
        }
        if self.native_ruleset_ids.is_empty() {
            return Err("active governance requires reviewed native ruleset IDs".into());
        }
        if self.observer_app_id != self.publisher_app_id
            || self.observer_installation_id != self.publisher_installation_id
        {
            return Err("this governance activation requires one App/installation with separately downscoped tokens".into());
        }
        let mut human_ids = BTreeMap::new();
        for login in ["micspiral", "ndizazzo", "i386"] {
            let id = self
                .human_ids
                .get(login)
                .copied()
                .flatten()
                .filter(|id| *id > 0)
                .ok_or("active governance requires every reviewed positive human user ID")?;
            human_ids.insert(login.to_owned(), id);
        }
        if self.human_ids.len() != 3 || human_ids.values().collect::<BTreeSet<_>>().len() != 3 {
            return Err("active governance human identity anchors must be the exact unique three-person roster".into());
        }
        Ok(Anchors {
            repository_id: required(self.repository_id, "repository_id")?,
            reviewer_team_id: required(self.reviewer_team_id, "reviewer_team_id")?,
            administration_environment_id: required(
                self.administration_environment_id,
                "administration_environment_id",
            )?,
            observer_app_id: required(self.observer_app_id, "observer_app_id")?,
            observer_installation_id: required(
                self.observer_installation_id,
                "observer_installation_id",
            )?,
            publisher_app_id: required(self.publisher_app_id, "publisher_app_id")?,
            publisher_installation_id: required(
                self.publisher_installation_id,
                "publisher_installation_id",
            )?,
            native_ruleset_ids: self.native_ruleset_ids.clone(),
            human_ids,
        })
    }
}

fn required(value: Option<u64>, name: &str) -> DynResult<u64> {
    value.ok_or_else(|| format!("active governance is missing reviewed {name}").into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn activation() -> Value {
        json!({"schema_version":1,"repository":"Mesh-LLM/mesh-llm","active":true,
            "repository_id":1,"reviewer_team_id":2,"administration_environment_id":3,
            "observer_app_id":4,"observer_installation_id":5,"publisher_app_id":4,
            "publisher_installation_id":5,"native_ruleset_ids":[8],"human_ids":{"micspiral":1,"ndizazzo":2,"i386":3}})
    }

    #[test]
    fn inactive_or_incomplete_configuration_never_grants_anchors() {
        let mut value = activation();
        value["active"] = json!(false);
        assert!(
            serde_json::from_value::<Activation>(value)
                .unwrap()
                .anchors()
                .is_err()
        );
        for field in [
            "repository_id",
            "reviewer_team_id",
            "administration_environment_id",
            "observer_app_id",
            "observer_installation_id",
            "publisher_app_id",
            "publisher_installation_id",
        ] {
            let mut value = activation();
            value[field] = Value::Null;
            assert!(
                serde_json::from_value::<Activation>(value)
                    .unwrap()
                    .anchors()
                    .is_err(),
                "{field}"
            );
        }
    }

    #[test]
    fn unknown_or_invalid_numeric_identity_configuration_is_rejected() {
        for (field, replacement) in [
            ("repository_id", json!(0)),
            ("native_ruleset_ids", json!([8, 8])),
            ("native_ruleset_ids", json!([])),
            ("repository", json!("foreign/repository")),
            ("schema_version", json!(2)),
        ] {
            let mut value = activation();
            value[field] = replacement;
            assert!(
                serde_json::from_value::<Activation>(value)
                    .unwrap()
                    .anchors()
                    .is_err()
            );
        }
        let mut value = activation();
        value["endpoint"] = json!("https://example.invalid");
        assert!(serde_json::from_value::<Activation>(value).is_err());
        assert!(
            serde_json::from_value::<Activation>(activation())
                .unwrap()
                .anchors()
                .is_ok()
        );
    }
}
