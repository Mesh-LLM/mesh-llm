//! Changed paths and crates to domains, then domains and profile to the
//! required slice closure with its reasons.

use crate::ci_plan::catalog::Ownership;
use crate::ci_plan::diagnostics::{PlanResult, fail};
use crate::ci_plan::glob_pattern;
use crate::ci_plan::request::{FORCE_ALL, Profile};
use crate::ci_plan::slice_catalog::SliceCatalog;
use std::collections::{BTreeMap, BTreeSet};

/// Owned domains in catalog order. Every changed path must be owned.
pub(super) fn matched_domains(
    ownership: &Ownership,
    changed_files: &[String],
    direct_crates: &[String],
) -> PlanResult<Vec<String>> {
    if changed_files.iter().any(|path| path == FORCE_ALL) {
        return Ok(ownership.domains.clone());
    }
    let mut matched = BTreeSet::new();
    let mut unknown_paths = Vec::new();
    for path in changed_files {
        let owners = ownership
            .path_rules
            .iter()
            .filter(|rule| {
                rule.patterns
                    .iter()
                    .any(|pattern| glob_pattern::matches(path, pattern))
            })
            .map(|rule| rule.domain.as_str())
            .collect::<Vec<_>>();
        if owners.is_empty() {
            unknown_paths.push(path.as_str());
        }
        matched.extend(owners);
    }
    for name in direct_crates {
        matched.extend(
            ownership
                .crate_rules
                .iter()
                .filter(|rule| {
                    rule.patterns
                        .iter()
                        .any(|pattern| glob_pattern::matches(name, pattern))
                })
                .map(|rule| rule.domain.as_str()),
        );
    }
    if !unknown_paths.is_empty() {
        return fail(format!(
            "ownership has no rule for changed paths: {}",
            unknown_paths.join(", ")
        ));
    }
    Ok(ownership
        .domains
        .iter()
        .filter(|domain| matched.contains(domain.as_str()))
        .cloned()
        .collect())
}

/// Documentation, optionally with CI control-plane prose, and nothing else.
pub(super) fn documentation_only(domains: &[String]) -> bool {
    !domains.is_empty()
        && domains
            .iter()
            .filter(|domain| *domain != "ci-control")
            .collect::<BTreeSet<_>>()
            == BTreeSet::from([&"docs".to_owned()])
}

pub(super) struct Selection {
    /// Required slices in catalog order.
    pub(super) required: Vec<String>,
    /// Sorted, unique reasons per required slice.
    pub(super) reasons: BTreeMap<String, Vec<String>>,
    pub(super) force_all_rows: bool,
}

/// `_select_slices`: profile base, domain rules, control-plane fail-open and
/// the dependency closure.
pub(super) fn select_slices(
    catalog: &SliceCatalog,
    profile: Profile,
    domains: &[String],
) -> PlanResult<Selection> {
    let Some(definition) = catalog.profiles.get(&profile) else {
        return fail(format!("profile {} must be an object", profile.name()));
    };
    let docs_only = documentation_only(domains);
    let has = |name: &str| domains.iter().any(|domain| domain == name);
    let mut reasons: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for slice in &definition.base_slices {
        reasons.insert(slice.clone(), vec!["profile:base".to_owned()]);
    }
    if profile != Profile::PrDraft || (has("ci-control") && docs_only) {
        for domain in domains {
            for slice in catalog.domain_rules.get(domain).into_iter().flatten() {
                reasons
                    .entry(slice.clone())
                    .or_default()
                    .push(format!("domain:{domain}"));
            }
        }
    }
    let mut force_all_rows = definition.all_rows;
    if (has("ci-control") && !docs_only) || has("runner-infra") {
        force_all_rows = true;
        for slice in &definition.control_slices {
            reasons
                .entry(slice.clone())
                .or_default()
                .push("control-plane:fail-open".to_owned());
        }
    }
    close_dependencies(catalog, &mut reasons);
    let required = catalog
        .slices
        .iter()
        .map(|slice| slice.id.clone())
        .filter(|id| reasons.contains_key(id))
        .collect::<Vec<_>>();
    for list in reasons.values_mut() {
        if list.is_empty() {
            list.push("planner".to_owned());
        }
        list.sort();
        list.dedup();
    }
    Ok(Selection {
        required,
        reasons,
        force_all_rows,
    })
}

/// Adds `dependency:<dependent>` until the selection is closed. Each
/// dependency is recorded once, from the first dependent (in slice order)
/// that pulled it in during the pass that discovered it.
fn close_dependencies(catalog: &SliceCatalog, reasons: &mut BTreeMap<String, Vec<String>>) {
    loop {
        let mut added = false;
        let selected = catalog
            .slices
            .iter()
            .filter(|slice| reasons.contains_key(&slice.id))
            .map(|slice| (slice.id.clone(), slice.depends_on.clone()))
            .collect::<Vec<_>>();
        for (id, dependencies) in selected {
            for dependency in dependencies {
                if let std::collections::btree_map::Entry::Vacant(entry) = reasons.entry(dependency)
                {
                    entry.insert(vec![format!("dependency:{id}")]);
                    added = true;
                }
            }
        }
        if !added {
            return;
        }
    }
}
