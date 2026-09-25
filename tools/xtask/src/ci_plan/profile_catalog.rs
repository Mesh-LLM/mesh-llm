//! Profile definitions in `ci/slices.yml`: base and control slices, the
//! all-rows switch and fan-out budgets whose platform ceilings must fit the
//! total worker ceiling.

use crate::ci_plan::diagnostics::{
    PlanResult, fail, positive_int, repr_list, sorted_unknown, string_list,
};
use crate::ci_plan::document::Json;
use std::collections::BTreeSet;

pub(super) const BUDGET_KEYS: [&str; 4] = [
    "linux_max_parallel",
    "macos_max_parallel",
    "windows_max_parallel",
    "total_max_workers",
];

pub(super) struct ProfileDefinition {
    pub(super) base_slices: Vec<String>,
    pub(super) control_slices: Vec<String>,
    pub(super) all_rows: bool,
    /// Budget values in `BUDGET_KEYS` order.
    pub(super) budgets: [u64; 4],
}

pub(super) fn profile_definition(
    name: &str,
    definition: &Json,
    known: &BTreeSet<&str>,
) -> PlanResult<ProfileDefinition> {
    if definition.as_object().is_none() {
        return fail(format!("profile {name} must be an object"));
    }
    let mut lists = Vec::with_capacity(2);
    for field in ["base_slices", "control_slices"] {
        let selected = string_list(definition.get(field), &format!("profile {name}.{field}"))?;
        let unknown = sorted_unknown(&selected, known);
        if !unknown.is_empty() {
            return fail(format!(
                "profile {name}.{field} references {}",
                repr_list(&unknown)
            ));
        }
        lists.push(selected);
    }
    let Some(Json::Bool(all_rows)) = definition.get("all_rows") else {
        return fail(format!("profile {name}.all_rows must be boolean"));
    };
    let budgets = budgets(name, definition.get("budgets"))?;
    let control_slices = lists.pop().unwrap_or_default();
    let base_slices = lists.pop().unwrap_or_default();
    Ok(ProfileDefinition {
        base_slices,
        control_slices,
        all_rows: *all_rows,
        budgets,
    })
}

fn budgets(name: &str, raw: Option<&Json>) -> PlanResult<[u64; 4]> {
    let Some(entries) = raw.and_then(Json::as_object) else {
        return fail(format!("profile {name}.budgets must be an object"));
    };
    let keys = entries
        .iter()
        .map(|(key, _)| key.as_str())
        .collect::<BTreeSet<_>>();
    if keys != BUDGET_KEYS.into_iter().collect() {
        return fail(format!("profile {name}.budgets has an unexpected shape"));
    }
    let mut values = [0_u64; 4];
    // The legacy check walks a Python set, whose order varies per process, so
    // with several invalid budgets it may name any of them; this names the
    // first in catalog order.
    for (key, value) in entries {
        let number = positive_int(Some(value)).and_then(|number| u64::try_from(number).ok());
        let Some(number) = number else {
            return fail(format!("profile {name}.budgets.{key} must be positive"));
        };
        if let Some(slot) = BUDGET_KEYS.iter().position(|budget| budget == key) {
            values[slot] = number;
        }
    }
    let platform: u128 = values[..3].iter().map(|value| u128::from(*value)).sum();
    if platform > u128::from(values[3]) {
        return fail(format!(
            "profile {name}.budgets platform ceilings exceed total_max_workers"
        ));
    }
    Ok(values)
}
