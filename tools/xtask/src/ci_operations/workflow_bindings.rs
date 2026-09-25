//! The image-binding census of `check`: every literal runner image in every
//! workflow must be a registered binding with the catalogued digest, and
//! matrix/conditional consumers must select the declared image.

use crate::ci_operations::identity_text::REPOSITORY;
use crate::ci_operations::python_access::{
    Outcome, eq, is_str, item, item_by, object, require, string,
};
use crate::ci_operations::workflow_census::{Workflows, job};
use crate::ci_operations::workflow_text::{
    image_values, matrix_rows, one_field, row_has_cuda_major,
};
use crate::ci_plan::document::Json;
use crate::prepared_input::python_value::display;
use std::collections::BTreeMap;

fn selector(key: &str, value: &str) -> Json {
    object(&[(key, string(value))])
}

/// The binding census; returns the registered locations per job.
pub(crate) fn check_bindings(
    catalog: &Json,
    workflows: &Workflows,
) -> Outcome<BTreeMap<(String, String), usize>> {
    let (images, roles) = (item(catalog, "images")?, item(catalog, "consumer_roles")?);
    let mut expected: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (role_id, role) in roles.as_object().unwrap_or_default() {
        let image = item_by(images, item(role, "image_id")?)?;
        for binding in item(role, "bindings")?.as_array().unwrap_or_default() {
            let workflow = display(Some(item(binding, "workflow")?));
            let job_id = display(Some(item(binding, "job")?));
            let place = format!("{workflow}:{job_id} ({role_id})");
            let body = workflows.get(&workflow).and_then(|jobs| job(jobs, &job_id));
            require(body.is_some(), || format!("{place}: missing job"))?;
            check_binding(
                &Binding {
                    catalog,
                    image,
                    binding,
                    place: &place,
                    workflow: &workflow,
                    job_id: &job_id,
                },
                body.unwrap_or_default(),
            )?;
            *expected.entry((workflow, job_id)).or_default() += 1;
        }
    }
    let mut actual: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (workflow, jobs) in workflows {
        for (job_id, body) in jobs {
            let count: usize = image_values(body)
                .iter()
                .map(|value| value.matches(REPOSITORY).count())
                .sum();
            if count > 0 {
                actual.insert((workflow.clone(), job_id.clone()), count);
            }
        }
    }
    require(actual == expected, || {
        "runner image consumer census drift; register every literal image binding".to_owned()
    })?;
    Ok(expected)
}

struct Binding<'a> {
    catalog: &'a Json,
    image: &'a Json,
    binding: &'a Json,
    place: &'a str,
    workflow: &'a str,
    job_id: &'a str,
}

fn check_binding(binding: &Binding<'_>, body: &str) -> Outcome<()> {
    let place = binding.place;
    let selected_selector = item(binding.binding, "matrix_selector")?;
    if eq(selected_selector, &selector("release_tag", "empty"))
        || eq(selected_selector, &selector("release_tag", "nonempty"))
    {
        return check_conditional(binding, body);
    }
    let mut selected = body;
    let matrix = *selected_selector != Json::Null;
    if matrix {
        require(
            one_field(body, "image", place, Some(6))? == "${{ matrix.runner_image }}",
            || format!("{place}: matrix image consumer drift"),
        )?;
        require(
            one_field(body, "pinned_epoch", place, None)? == "${{ matrix.toolchain_epoch }}",
            || format!("{place}: matrix epoch consumer drift"),
        )?;
        let major = display(Some(item(selected_selector, "cuda_major")?));
        let rows: Vec<&str> = matrix_rows(body)
            .into_iter()
            .filter(|row| row_has_cuda_major(row, &major))
            .collect();
        require(rows.len() == 1, || {
            format!(
                "{place}: missing or duplicate CUDA matrix row {}",
                crate::prepared_input::python_value::repr(Some(selected_selector))
            )
        })?;
        selected = rows[0];
    }
    let reference = display(Some(item(binding.image, "reference")?));
    let field = if matrix { "runner_image" } else { "image" };
    let actual = one_field(selected, field, place, if matrix { None } else { Some(6) })?;
    let expression =
        display(Some(item(binding.binding, "image_expression")?)).replace("{image}", &reference);
    require(actual == expression, || {
        format!("{place}: image reference drift")
    })?;
    let epoch_field = item(binding.binding, "epoch_field")?;
    if let Some(name) = epoch_field.as_str().filter(|name| !name.is_empty()) {
        let epoch = item(binding.image, "native_toolchain_epoch")?;
        let actual = one_field(selected, name, place, None)?;
        require(is_str(epoch, &actual), || {
            format!("{place}: native toolchain epoch drift")
        })?;
    }
    Ok(())
}

fn check_conditional(binding: &Binding<'_>, body: &str) -> Outcome<()> {
    let (images, roles) = (
        item(binding.catalog, "images")?,
        item(binding.catalog, "consumer_roles")?,
    );
    let mut branches: BTreeMap<String, String> = BTreeMap::new();
    for (_, owner) in roles.as_object().unwrap_or_default() {
        for other in item(owner, "bindings")?.as_array().unwrap_or_default() {
            if is_str(item(other, "workflow")?, binding.workflow)
                && is_str(item(other, "job")?, binding.job_id)
            {
                let tag = display(Some(item(item(other, "matrix_selector")?, "release_tag")?));
                let reference = display(Some(item(
                    item_by(images, item(owner, "image_id")?)?,
                    "reference",
                )?));
                branches.insert(tag, reference);
            }
        }
    }
    let branch = |tag: &str| {
        branches
            .get(tag)
            .cloned()
            .ok_or_else(|| crate::repository::python_text::repr(tag))
    };
    let expression = format!(
        "${{{{ inputs.release_tag != '' && '{}' || '{}' }}}}",
        branch("nonempty")?,
        branch("empty")?
    );
    let place = binding.place;
    require(
        one_field(body, "image", place, Some(6))? == expression,
        || format!("{place}: conditional image reference drift"),
    )
}
