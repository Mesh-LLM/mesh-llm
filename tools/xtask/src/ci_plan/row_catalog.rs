//! Matrix row catalogs (`runtime_rows`, `platform_rows`, `sdk_rows`,
//! `smoke_rows`) and the domain-to-row mappings that select from them.
//! Rows are carried verbatim, because the plan copies every row field.

use crate::ci_plan::diagnostics::{
    PlanResult, fail, nonempty_string, repr, repr_list, sorted_unknown, string_list,
};
use crate::ci_plan::document::Json;
use std::collections::BTreeSet;

/// One catalog of rows, in catalog order, plus its domain mapping.
pub(super) struct RowCatalog {
    pub(super) field: &'static str,
    pub(super) rows: Vec<(String, Json)>,
    /// Domain to row IDs, from `domain_rows`, `platform_domain_rows`, ...
    pub(super) mapping: Vec<(String, Vec<String>)>,
}

impl RowCatalog {
    pub(super) fn ids(&self) -> impl Iterator<Item = &str> {
        self.rows.iter().map(|(id, _)| id.as_str())
    }

    pub(super) fn row(&self, id: &str) -> Option<&Json> {
        self.rows
            .iter()
            .find(|(row_id, _)| row_id == id)
            .map(|(_, row)| row)
    }

    /// Row IDs mapped from `domains`, in domain order, repeats kept.
    pub(super) fn mapped(&self, domains: &[String]) -> Vec<String> {
        domains
            .iter()
            .flat_map(|domain| {
                self.mapping
                    .iter()
                    .filter(move |(name, _)| name == domain)
                    .flat_map(|(_, ids)| ids.iter().cloned())
            })
            .collect()
    }
}

pub(super) struct RowCatalogs {
    pub(super) runtime: RowCatalog,
    pub(super) platform: RowCatalog,
    pub(super) sdk: RowCatalog,
    pub(super) smoke: RowCatalog,
}

/// Validates the four row catalogs in legacy order.
pub(super) fn validate(slices: &Json) -> PlanResult<RowCatalogs> {
    Ok(RowCatalogs {
        runtime: rows(
            slices,
            "runtime_rows",
            &["platform", "architecture", "runner_role"],
        )?,
        platform: rows(slices, "platform_rows", &["platform", "architecture"])?,
        sdk: rows(
            slices,
            "sdk_rows",
            &["language", "platform", "architecture"],
        )?,
        smoke: rows(slices, "smoke_rows", &["kind"])?,
    })
}

fn rows(slices: &Json, field: &'static str, required: &[&str]) -> PlanResult<RowCatalog> {
    let Some(items) = slices.get(field).and_then(Json::as_array) else {
        return fail(format!("slices.{field} must be an array"));
    };
    let mut parsed: Vec<(String, Json)> = Vec::with_capacity(items.len());
    for (index, row) in items.iter().enumerate() {
        if row.as_object().is_none() {
            return fail(format!("slices.{field}[{index}] must be an object"));
        }
        let id = nonempty_string(row.get("id"), &format!("{field}[{index}].id"))?;
        for key in required {
            nonempty_string(row.get(key), &format!("{field}[{index}].{key}"))?;
        }
        parsed.push((id, row.clone()));
    }
    let unique = parsed.iter().map(|(id, _)| id).collect::<BTreeSet<_>>();
    if unique.len() != parsed.len() {
        return fail(format!("slices.{field} contains duplicate IDs"));
    }
    Ok(RowCatalog {
        field,
        rows: parsed,
        mapping: Vec::new(),
    })
}

/// Validates the domain-to-row mappings (after `batch_limits`, as the legacy
/// planner does) and attaches each to its catalog.
pub(super) fn validate_mappings(
    slices: &Json,
    domains: &BTreeSet<&str>,
    catalogs: &mut RowCatalogs,
) -> PlanResult<()> {
    let order: [(&str, &mut RowCatalog); 4] = [
        ("domain_rows", &mut catalogs.runtime),
        ("smoke_domain_rows", &mut catalogs.smoke),
        ("platform_domain_rows", &mut catalogs.platform),
        ("sdk_domain_rows", &mut catalogs.sdk),
    ];
    for (field, catalog) in order {
        let Some(entries) = slices.get(field).and_then(Json::as_object) else {
            return fail(format!("slices.{field} must be an object"));
        };
        let known = catalog.ids().collect::<BTreeSet<_>>();
        let mut mapping = Vec::with_capacity(entries.len());
        for (domain, row_ids) in entries {
            if !domains.contains(domain.as_str()) {
                return fail(format!(
                    "{field} references unknown domain {}",
                    repr(domain)
                ));
            }
            let mapped = string_list(Some(row_ids), &format!("{field}.{domain}"))?;
            let unknown = sorted_unknown(&mapped, &known);
            if !unknown.is_empty() {
                return fail(format!(
                    "{field}.{domain} references unknown rows {}",
                    repr_list(&unknown)
                ));
            }
            mapping.push((domain.clone(), mapped));
        }
        catalog.mapping = mapping;
    }
    Ok(())
}
