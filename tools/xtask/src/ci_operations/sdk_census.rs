//! SDK Rust consumer check of `check`: the SDK workflow's environment,
//! pinned toolchain action and cache key expression match the catalog.

use crate::ci_operations::catalog_validation::image_digest;
use crate::ci_operations::python_access::{Outcome, item, item_by, require};
use crate::ci_operations::seed_census::inputs_repr;
use crate::ci_operations::workflow_census::{Workflows, job, read_text};
use crate::ci_operations::workflow_text::one_field;
use crate::ci_plan::document::Json;
use crate::prepared_input::python_value::display;
use crate::repository::python_text::repr;
use std::path::Path;

fn sdk_cache_expression(catalog: &Json) -> Outcome<String> {
    let inputs = inputs_repr(item(item(catalog, "sdk_rust")?, "cache_recipe_inputs")?);
    Ok(format!(
        "${{{{ format('{{0}}-sdk-rust-cargo-v1-{{1}}-{{2}}-{{3}}-image-{{4}}-{{5}}-{{6}}-{{7}}', env.CACHE_NAMESPACE, runner.os, runner.arch, env.SDK_RUST_TARGET, env.SDK_RUST_IMAGE_DIGEST, env.SDK_RUST_TOOLCHAIN_EPOCH, env.SDK_RUST_PROFILE_LINKER, hashFiles({inputs})) }}}}"
    ))
}

pub(crate) fn check_sdk(catalog: &Json, workflows: &Workflows, root: &Path) -> Outcome<()> {
    let (images, roles, sdk) = (
        item(catalog, "images")?,
        item(catalog, "consumer_roles")?,
        item(catalog, "sdk_rust")?,
    );
    let role = item_by(roles, item(sdk, "role")?)?;
    let binding = &item(role, "bindings")?.as_array().unwrap_or_default()[0];
    let workflow = display(Some(item(binding, "workflow")?));
    let text = read_text(&root.join(".github/workflows").join(&workflow))?;
    let image = item_by(images, item(role, "image_id")?)?;
    let expected = [
        ("TARGET", display(Some(item(sdk, "target")?))),
        ("IMAGE_DIGEST", image_digest(image)?),
        (
            "TOOLCHAIN_EPOCH",
            display(Some(item(sdk, "toolchain_epoch")?)),
        ),
        (
            "PROFILE_LINKER",
            display(Some(item(sdk, "profile_linker")?)),
        ),
    ];
    for (name, value) in expected {
        require(
            one_field(&text, &format!("SDK_RUST_{name}"), "sdk_rust", Some(2))? == value,
            || format!("SDK Rust {name} drift"),
        )?;
    }
    let job_id = display(Some(item(binding, "job")?));
    let body = workflows.get(&workflow).and_then(|jobs| job(jobs, &job_id));
    let body = body.ok_or_else(|| repr(&job_id))?;
    let refs: Vec<&str> = body
        .match_indices("uses: dtolnay/rust-toolchain@")
        .map(|(at, _)| {
            let start = at + "uses: ".len();
            let rest = &body[start..];
            &rest[..rest.find(char::is_whitespace).unwrap_or(rest.len())]
        })
        .collect();
    let action = display(Some(item(sdk, "rust_action_ref")?));
    require(refs == [action.as_str()], || {
        "SDK Rust action drift".to_owned()
    })?;
    let namespace = display(Some(item(sdk, "cache_namespace")?));
    require(
        one_field(&text, "CACHE_NAMESPACE", "sdk_rust", Some(2))? == namespace,
        || "SDK Rust namespace drift".to_owned(),
    )?;
    require(
        one_field(body, "prefix-key", "sdk_rust", None)? == sdk_cache_expression(catalog)?,
        || "SDK Rust cache key expression drift".to_owned(),
    )
}
