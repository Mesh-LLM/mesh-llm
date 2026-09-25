//! The slice half of `_validate_manifests`: profiles, slice definitions,
//! domain rules, row catalogs, batch limits and the dependency graph. Checks run in the legacy order so the first reported
//! problem is the same.

use crate::ci_plan::catalog::Ownership;
use crate::ci_plan::diagnostics::{
    PlanResult, fail, nonempty_string, positive_int, repr, repr_list, sorted_unknown, string_list,
};
use crate::ci_plan::document::Json;
use crate::ci_plan::profile_catalog::{self, ProfileDefinition};
use crate::ci_plan::request::Profile;
use crate::ci_plan::row_catalog::{self, RowCatalogs};
use crate::ci_plan::slice_graph;
use std::collections::{BTreeMap, BTreeSet};

const CACHE_MODES: [&str; 4] = [
    "none",
    "pr-isolated",
    "trusted-readonly",
    "trusted-readwrite",
];

pub(super) struct SliceDefinition {
    pub(super) id: String,
    pub(super) runner_role: String,
    pub(super) cache_mode: String,
    pub(super) depends_on: Vec<String>,
}

/// The validated `ci/slices.yml`.
pub(super) struct SliceCatalog {
    pub(super) profiles: BTreeMap<Profile, ProfileDefinition>,
    pub(super) slices: Vec<SliceDefinition>,
    pub(super) domain_rules: BTreeMap<String, Vec<String>>,
    pub(super) rows: RowCatalogs,
    pub(super) clippy_batches: usize,
    pub(super) rust_test_batches: usize,
}

impl SliceCatalog {
    pub(super) fn slice(&self, id: &str) -> Option<&SliceDefinition> {
        self.slices.iter().find(|slice| slice.id == id)
    }
}

pub(super) fn validate(slices: &Json, ownership: &Ownership) -> PlanResult<SliceCatalog> {
    let raw_profiles = slices.get("profiles").and_then(Json::as_object);
    let profile_names = raw_profiles.map(|entries| {
        entries
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>()
    });
    let expected = Profile::ALL
        .map(Profile::name)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let Some(raw_profiles) = raw_profiles.filter(|_| profile_names.as_ref() == Some(&expected))
    else {
        return fail("slices.profiles must define exactly the four supported profiles");
    };
    let definitions = slice_definitions(slices)?;
    let known = definitions
        .iter()
        .map(|slice| slice.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut profiles = BTreeMap::new();
    for (name, definition) in raw_profiles {
        let Some(profile) = Profile::parse(name) else {
            return fail("slices.profiles must define exactly the four supported profiles");
        };
        profiles.insert(
            profile,
            profile_catalog::profile_definition(name, definition, &known)?,
        );
    }
    let domain_rules = domain_rules(slices, ownership, &known)?;
    let mut rows = row_catalog::validate(slices)?;
    let batch_limits = slices.get("batch_limits");
    if batch_limits.and_then(Json::as_object).is_none() {
        return fail("slices.batch_limits must be an object");
    }
    let mut limits = [0_usize; 2];
    for (slot, key) in limits.iter_mut().zip(["clippy", "rust_tests"]) {
        let limit = positive_int(batch_limits.and_then(|limits| limits.get(key)));
        let Some(limit) = limit.and_then(|limit| usize::try_from(limit).ok()) else {
            return fail(format!("slices.batch_limits.{key} must be positive"));
        };
        *slot = limit;
    }
    row_catalog::validate_mappings(slices, &ownership.domain_set(), &mut rows)?;
    slice_graph::check_dependencies(&definitions, &known)?;
    Ok(SliceCatalog {
        profiles,
        slices: definitions,
        domain_rules,
        rows,
        clippy_batches: limits[0],
        rust_test_batches: limits[1],
    })
}

fn slice_definitions(slices: &Json) -> PlanResult<Vec<SliceDefinition>> {
    let Some(items) = slices.get("slices").and_then(Json::as_array) else {
        return fail("slices.slices must be an array");
    };
    let mut definitions = Vec::with_capacity(items.len());
    for (index, definition) in items.iter().enumerate() {
        if definition.as_object().is_none() {
            return fail(format!("slices.slices[{index}] must be an object"));
        }
        let id = nonempty_string(definition.get("id"), &format!("slices.slices[{index}].id"))?;
        nonempty_string(definition.get("kind"), &format!("slice {id}.kind"))?;
        let runner_role = nonempty_string(
            definition.get("runner_role"),
            &format!("slice {id}.runner_role"),
        )?;
        let cache_mode = nonempty_string(
            definition.get("cache_mode"),
            &format!("slice {id}.cache_mode"),
        )?;
        if !CACHE_MODES.contains(&cache_mode.as_str()) {
            return fail(format!(
                "slice {id} has invalid cache_mode {}",
                repr(&cache_mode)
            ));
        }
        let empty = Json::Array(Vec::new());
        let depends_on = string_list(
            Some(definition.get("depends_on").unwrap_or(&empty)),
            &format!("slice {id}.depends_on"),
        )?;
        if depends_on.contains(&id) {
            return fail(format!("slice {id} cannot depend on itself"));
        }
        definitions.push(SliceDefinition {
            id,
            runner_role,
            cache_mode,
            depends_on,
        });
    }
    let ids = definitions
        .iter()
        .map(|slice| &slice.id)
        .collect::<BTreeSet<_>>();
    if ids.len() != definitions.len() {
        return fail("slices.slices contains duplicate IDs");
    }
    Ok(definitions)
}

fn domain_rules(
    slices: &Json,
    ownership: &Ownership,
    known: &BTreeSet<&str>,
) -> PlanResult<BTreeMap<String, Vec<String>>> {
    let Some(items) = slices.get("domain_rules").and_then(Json::as_array) else {
        return fail("slices.domain_rules must be an array");
    };
    let domains = ownership.domain_set();
    let mut rules = BTreeMap::new();
    for (index, rule) in items.iter().enumerate() {
        if rule.as_object().is_none() {
            return fail(format!("slices.domain_rules[{index}] must be an object"));
        }
        let domain = nonempty_string(rule.get("domain"), &format!("domain_rules[{index}].domain"))?;
        if !domains.contains(domain.as_str()) {
            return fail(format!(
                "slice rule references unknown domain {}",
                repr(&domain)
            ));
        }
        if rules.contains_key(&domain) {
            return fail(format!("duplicate slice rule for domain {}", repr(&domain)));
        }
        let selected = string_list(rule.get("slices"), &format!("domain rule {domain}.slices"))?;
        let unknown = sorted_unknown(&selected, known);
        if !unknown.is_empty() {
            return fail(format!(
                "domain {domain} references unknown slices {}",
                repr_list(&unknown)
            ));
        }
        rules.insert(domain, selected);
    }
    let missing = domains
        .iter()
        .filter(|domain| !rules.contains_key(**domain))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return fail(format!(
            "slice domain rules do not cover: {}",
            missing.join(", ")
        ));
    }
    Ok(rules)
}
