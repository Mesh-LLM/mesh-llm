//! `ci-ops runner-identity`: the Rust owner of
//! `scripts/runner-image-identity.py`. Inspects checked-in runner image
//! identities without changing CI execution; all inputs are local files and
//! no network, build or cache operation is performed. Output, diagnostics
//! and exit statuses match the legacy tool byte for byte.

use crate::ci_operations::catalog_validation::validate;
use crate::ci_operations::evidence_catalog::{BindRequest, bind};
use crate::ci_operations::evidence_input::{decode, read_bytes};
use crate::ci_operations::identity_text::lower_hex;
use crate::ci_operations::python_access::{
    Outcome, contains_key, item, item_by, object, require, string,
};
use crate::ci_operations::runner_identity_argv::{Args, Parsed, parse};
use crate::ci_operations::runner_identity_check::{check, planner_rows, runtime_consumer};
use crate::ci_operations::workflow_census::{job, workflow_jobs};
use crate::ci_operations::workflow_text::one_field;
use crate::ci_plan::catalog::python_path_display;
use crate::ci_plan::document::Json;
use crate::prepared_input::python_json::dumps_indented;
use crate::prepared_input::python_value::display;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::repr;
use std::path::{Path, PathBuf};

/// `default_root` is the checkout the legacy script lived in.
pub(crate) fn run(args: &[String], default_root: impl FnOnce() -> PathBuf) -> CheckReport {
    let args = match parse(args) {
        Parsed::Run(args) => args,
        Parsed::Report(report) => return report,
    };
    let root = args.root.as_ref().map_or_else(default_root, PathBuf::from);
    match execute(&args, &root) {
        Ok(Json::String(text)) => CheckReport::success(format!("{text}\n")),
        Ok(value) => CheckReport::success(dumps_indented(&value) + "\n"),
        Err(message) => {
            CheckReport::failure(String::new(), format!("runner image identity: {message}\n"))
        }
    }
}

pub(crate) fn read_json(path: &Path) -> Outcome<Json> {
    let value = decode(&read_bytes(path)?)?;
    require(value.as_object().is_some(), || {
        format!("{}: expected an object", python_path_display(path))
    })?;
    Ok(value)
}

fn execute(args: &Args, root: &Path) -> Outcome<Json> {
    let catalog_path = args
        .catalog
        .as_ref()
        .map_or_else(|| root.join("ci/runner-images.json"), PathBuf::from);
    let catalog = read_json(&catalog_path)?;
    validate(&catalog, root)?;
    match args.command.as_str() {
        "bind" => bind(
            &catalog,
            &BindRequest {
                image_id: &args.image_id,
                cohort: Path::new(&args.cohort),
                anchor: Path::new(&args.anchor),
                output: Path::new(&args.output),
                root,
            },
        ),
        "check" => check(&catalog, root),
        "diagnose" => diagnose(&catalog, root),
        "lookup" => lookup(&catalog, &args.role, args.field.as_deref()),
        "seed-key" => {
            require(lower_hex(&args.recipe_hash, 64), || {
                "recipe hash must be lowercase SHA-256".to_owned()
            })?;
            let prefix = display(Some(item(item(&catalog, "compiler_seed")?, "key_prefix")?));
            Ok(string(&format!("{prefix}{}", args.recipe_hash)))
        }
        _ => Ok(object(&[
            ("schema_version", item(&catalog, "schema_version")?.clone()),
            ("valid", Json::Bool(true)),
        ])),
    }
}

fn lookup(catalog: &Json, role_id: &str, field: Option<&str>) -> Outcome<Json> {
    let roles = item(catalog, "consumer_roles")?;
    require(contains_key(roles, &string(role_id))?, || {
        format!("unknown consumer role: {role_id}")
    })?;
    let role = item(roles, role_id)?;
    let image = item_by(item(catalog, "images")?, item(role, "image_id")?)?;
    Ok(match field {
        Some(field) => item(image, field)?.clone(),
        None => image.clone(),
    })
}

/// Eligibility concerns, reported separately; a matching image is not coverage.
fn diagnose(catalog: &Json, root: &Path) -> Outcome<Json> {
    let (workflow, job_id, row_id) = runtime_consumer(catalog)?;
    let jobs = workflow_jobs(
        &root.join(".github/workflows").join(&workflow),
        Some(&job_id),
    )?;
    let body = job(&jobs, &job_id).ok_or_else(|| repr(&job_id))?;
    require(
        one_field(body, "allow_trusted_seed", "runtime consumer", None)? == "false",
        || "runtime seed restore is deliberately disabled".to_owned(),
    )?;
    let matches = planner_rows(root)?
        .iter()
        .filter(|row| row["id"] == row_id.as_str())
        .count();
    require(matches == 1, || {
        format!("planner has no unique row {row_id}")
    })?;
    let mut messages = vec![string(
        "runtime seed restore is deliberately disabled: run 34272984200/1 observed zero reuse in all three verified warm samples",
    )];
    if *item(item(catalog, "compiler_seed")?, "workload_coverage")? == Json::Null {
        messages.push(string(
            "compiler seed workload coverage is unqualified; image identity alone does not establish warm-cache coverage",
        ));
    }
    Ok(Json::Array(messages))
}
