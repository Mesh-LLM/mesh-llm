//! The release-tag image branch rule: conditional `release_tag` selectors
//! must form exactly one complete UI artifact pair.

use crate::ci_operations::catalog_validation::is_conditional;
use crate::ci_operations::python_access::{Outcome, is_str, require};
use crate::ci_plan::document::Json;

fn all_bindings(roles: &Json) -> Vec<&Json> {
    roles
        .as_object()
        .unwrap_or_default()
        .iter()
        .flat_map(|(_, role)| {
            role.get("bindings")
                .and_then(Json::as_array)
                .unwrap_or_default()
        })
        .collect()
}

pub(crate) fn validate_conditional_pair(roles: &Json) -> Outcome<()> {
    let bindings = all_bindings(roles);
    let field = |binding: &Json, key: &str| binding.get(key).cloned().unwrap_or(Json::Null);
    let in_artifact_job = |binding: &&Json| {
        is_str(&field(binding, "workflow"), "ci-ui-artifact-slice.yml")
            && is_str(&field(binding, "job"), "ui_artifact")
    };
    let conditional: Vec<&Json> = bindings
        .iter()
        .copied()
        .filter(|binding| is_conditional(&field(binding, "matrix_selector")))
        .collect();
    if conditional.is_empty() {
        return Ok(());
    }
    let occupants = bindings
        .iter()
        .filter(|binding| in_artifact_job(binding))
        .count();
    let mut tags: Vec<String> = conditional
        .iter()
        .filter_map(|binding| {
            field(binding, "matrix_selector")
                .get("release_tag")
                .and_then(Json::as_str)
                .map(str::to_owned)
        })
        .collect();
    tags.sort_unstable();
    tags.dedup();
    let complete = occupants == 2
        && conditional.len() == 2
        && tags == ["empty", "nonempty"]
        && conditional.iter().all(|binding| {
            in_artifact_job(binding)
                && is_str(&field(binding, "image_expression"), "{image}")
                && field(binding, "epoch_field") == Json::Null
        });
    require(complete, || {
        "release-tag image branches must be one complete UI artifact pair".to_owned()
    })
}
