//! `check` and the planner-row lookup shared with `diagnose`: every
//! registered consumer, the actual planner runtime rows, compiler seed and
//! SDK Rust consumers must agree with the catalog.

use crate::ci_operations::catalog_validation::validate;
use crate::ci_operations::python_access::{Outcome, item, item_by, object, require};
use crate::ci_operations::runner_identity::read_json;
use crate::ci_operations::sdk_census::check_sdk;
use crate::ci_operations::seed_census::check_seed;
use crate::ci_operations::workflow_bindings::check_bindings;
use crate::ci_operations::workflow_census::{Workflows, job, workflow_jobs};
use crate::ci_operations::workflow_text::one_field;
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::ci_plan::document::Json;
use crate::prepared_input::python_value::display;
use crate::repository::python_text::repr;
use serde_json::Value;
use std::path::{Path, PathBuf};

pub(crate) fn runtime_consumer(catalog: &Json) -> Outcome<(String, String, String)> {
    let runtime = item(item(catalog, "compiler_seed")?, "runtime_consumer")?;
    Ok((
        display(Some(item(runtime, "workflow")?)),
        display(Some(item(runtime, "job")?)),
        display(Some(item(runtime, "row_id")?)),
    ))
}

/// `planner_rows(root)`: the exhaustive main-profile runtime rows from the
/// checked-in planner catalogs, through the Rust planner.
pub(crate) fn planner_rows(root: &Path) -> Outcome<Vec<Value>> {
    let planner = root.join("scripts/plan-ci.py");
    if !planner.is_file() {
        let error = std::io::Error::from_raw_os_error(2);
        return Err(os_error_text(&error, &python_path_display(&planner)));
    }
    let slices = read_json(&root.join("ci/slices.yml"))?;
    let ownership = read_json(&root.join("ci/ownership.yml"))?;
    crate::ci_plan::exhaustive_runtime_rows(&ownership, &slices)
}

fn load_workflows(catalog: &Json, root: &Path) -> Outcome<Workflows> {
    let directory = root.join(".github/workflows");
    let shown = python_path_display(&directory);
    let entries = std::fs::read_dir(&directory).map_err(|error| os_error_text(&error, &shown))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext == "yml" || ext == "yaml")
        })
        .collect();
    paths.sort();
    let (runtime_workflow, runtime_job, _) = runtime_consumer(catalog)?;
    let mut workflows = Workflows::new();
    for path in paths {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let guard = (name == runtime_workflow).then_some(runtime_job.as_str());
        workflows.insert(name, workflow_jobs(&path, guard)?);
    }
    Ok(workflows)
}

pub(crate) fn check(catalog: &Json, root: &Path) -> Outcome<Json> {
    validate(catalog, root)?;
    let workflows = load_workflows(catalog, root)?;
    let locations = check_bindings(catalog, &workflows)?;
    let (runtime_workflow, runtime_job, _) = runtime_consumer(catalog)?;
    let jobs = workflows
        .get(&runtime_workflow)
        .ok_or_else(|| repr(&runtime_workflow))?;
    let body = job(jobs, &runtime_job).ok_or_else(|| repr(&runtime_job))?;
    require(
        one_field(body, "image", "runtime consumer", Some(6))?
            == "${{ matrix.runtime.container_image }}",
        || "runtime matrix image consumer drift".to_owned(),
    )?;
    require(
        one_field(body, "pinned_epoch", "runtime consumer", None)?
            == "${{ matrix.runtime.toolchain_epoch }}",
        || "runtime matrix epoch consumer drift".to_owned(),
    )?;
    let runtime_rows = check_planner_rows(catalog, root)?;
    let seed_consumers = check_seed(catalog, &workflows, root)?;
    check_sdk(catalog, &workflows, root)?;
    let count = |value: usize| Json::Number(serde_json::Number::from(value));
    let object_len =
        |key: &str| item(catalog, key).map(|value| value.as_object().unwrap_or_default().len());
    Ok(object(&[
        ("images", count(object_len("images")?)),
        ("roles", count(object_len("consumer_roles")?)),
        ("workflow_bindings", count(locations.values().sum())),
        ("runtime_rows", count(runtime_rows)),
        ("seed_consumers", count(seed_consumers)),
    ]))
}

fn check_planner_rows(catalog: &Json, root: &Path) -> Outcome<usize> {
    let rows = planner_rows(root)?;
    let with_image: Vec<&Value> = rows
        .iter()
        .filter(|row| row.get("container_image").is_some())
        .collect();
    let mut ids: Vec<String> = with_image
        .iter()
        .map(|row| row["id"].as_str().unwrap_or_default().to_owned())
        .collect();
    let total = ids.len();
    ids.sort();
    ids.dedup();
    require(ids.len() == total, || {
        "duplicate planner image row".to_owned()
    })?;
    let expected_rows = item(catalog, "runtime_rows")?
        .as_object()
        .unwrap_or_default();
    let mut expected_ids: Vec<String> = expected_rows.iter().map(|(key, _)| key.clone()).collect();
    expected_ids.sort();
    require(ids == expected_ids, || {
        "planner image row census drift".to_owned()
    })?;
    let images = item(catalog, "images")?;
    for (row_id, expected) in expected_rows {
        let actual = with_image
            .iter()
            .rev()
            .find(|row| row["id"] == row_id.as_str())
            .map_or(&Value::Null, |row| *row);
        let image = item_by(images, item(expected, "image_id")?)?;
        for field in ["platform", "architecture", "backend"] {
            let want = item(expected, field)?.to_value();
            require(actual[field] == want, || {
                format!("{row_id}: planner {field} drift")
            })?;
        }
        require(
            actual["container_image"] == item(image, "reference")?.to_value(),
            || format!("{row_id}: planner image drift"),
        )?;
        let epoch = item(image, "native_toolchain_epoch")?.to_value();
        require(actual["toolchain_epoch"] == epoch, || {
            format!("{row_id}: planner epoch drift")
        })?;
    }
    Ok(with_image.len())
}
