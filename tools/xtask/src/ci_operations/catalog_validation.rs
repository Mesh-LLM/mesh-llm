//! `validate(catalog, root)` from `scripts/runner-image-identity.py`: the
//! image, consumer-role and binding contract. Runtime rows, the compiler
//! seed and the SDK Rust block live in `catalog_contracts`.

use crate::ci_operations::catalog_contracts;
use crate::ci_operations::evidence_catalog;
use crate::ci_operations::identity_text::{
    EPOCH_PREFIX, is_identifier, is_reference, is_workflow_name,
};
use crate::ci_operations::python_access::{
    Outcome, contains_key, eq, has_exact_fields, is_str, item, item_by, object, require, string,
};
use crate::ci_plan::document::Json;
use crate::prepared_input::python_value::repr;
use std::path::Path;

/// `fields(value, expected, where)`.
pub(crate) fn fields(value: &Json, expected: &str, place: &str) -> Outcome<()> {
    require(value.as_object().is_some(), || {
        format!("{place}: expected an object")
    })?;
    require(has_exact_fields(value, expected), || {
        format!("{place}: unexpected or missing fields")
    })
}

/// `named_map(value, where)`: a nonempty object keyed by identifiers.
pub(crate) fn named_map<'a>(value: &'a Json, place: &str) -> Outcome<&'a [(String, Json)]> {
    let entries = value.as_object().filter(|entries| !entries.is_empty());
    require(entries.is_some(), || {
        format!("{place}: expected a nonempty object")
    })?;
    let entries = entries.unwrap_or_default();
    require(entries.iter().all(|(key, _)| is_identifier(key)), || {
        format!("{place}: invalid identifier")
    })?;
    Ok(entries)
}

/// `digest(image)`: the hex part of a validated immutable reference.
pub(crate) fn image_digest(image: &Json) -> Outcome<String> {
    let reference = item(image, "reference")?;
    let text = reference.as_str().unwrap_or_default();
    Ok(text
        .rsplit_once(':')
        .map_or(text, |(_, hex)| hex)
        .to_owned())
}

fn one_of(value: &Json, choices: &[&str]) -> bool {
    choices.iter().any(|choice| is_str(value, choice))
}

pub(crate) fn validate(catalog: &Json, root: &Path) -> Outcome<()> {
    fields(
        catalog,
        "schema_version images consumer_roles runtime_rows compiler_seed sdk_rust",
        "catalog",
    )?;
    let schema = item(catalog, "schema_version")?;
    require(schema.as_int() == Some(1), || {
        "unsupported schema_version".to_owned()
    })?;
    let images = item(catalog, "images")?;
    let mut references: Vec<String> = Vec::new();
    for (image_id, image) in named_map(images, "images")? {
        validate_image(image_id, image)?;
        references.push(
            item(image, "reference")?
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        );
    }
    evidence_catalog::validate_catalog(catalog, root)?;
    let mut unique = references.clone();
    unique.sort_unstable();
    unique.dedup();
    require(unique.len() == references.len(), || {
        "duplicate image reference".to_owned()
    })?;
    let roles = item(catalog, "consumer_roles")?;
    let mut seen: Vec<(String, String, String)> = Vec::new();
    for (role_id, role) in named_map(roles, "consumer_roles")? {
        validate_role(role_id, role, images, &mut seen)?;
    }
    let mut unique = seen.clone();
    unique.sort_unstable();
    unique.dedup();
    require(unique.len() == seen.len(), || {
        "duplicate consumer binding".to_owned()
    })?;
    crate::ci_operations::catalog_release_pair::validate_conditional_pair(roles)?;
    catalog_contracts::validate_runtime_rows(catalog, images)?;
    catalog_contracts::validate_compiler_seed(catalog, images, roles)?;
    catalog_contracts::validate_sdk_rust(catalog, roles)
}

fn validate_image(image_id: &str, image: &Json) -> Outcome<()> {
    fields(
        image,
        "reference environment backend native_toolchain_epoch receipt provenance",
        image_id,
    )?;
    let reference = item(image, "reference")?;
    require(reference.as_str().is_some_and(is_reference), || {
        format!("{image_id}: image must be an immutable runner digest")
    })?;
    require(is_str(item(image, "environment")?, "public"), || {
        format!("{image_id}: unsupported environment")
    })?;
    let backends = [
        "cpu", "web", "cuda12", "cuda13", "rocm", "vulkan", "ui", "browser",
    ];
    require(one_of(item(image, "backend")?, &backends), || {
        format!("{image_id}: unsupported backend")
    })?;
    let epoch = item(image, "native_toolchain_epoch")?;
    let expected = string(&format!("{EPOCH_PREFIX}{}", image_digest(image)?));
    require(*epoch == Json::Null || eq(epoch, &expected), || {
        format!("{image_id}: epoch/digest mismatch")
    })
}

fn release_tag(value: &str) -> Json {
    object(&[("release_tag", string(value))])
}

fn cuda_major(value: &str) -> Json {
    object(&[("cuda_major", string(value))])
}

pub(crate) fn is_conditional(selector: &Json) -> bool {
    eq(selector, &release_tag("empty")) || eq(selector, &release_tag("nonempty"))
}

fn validate_role(
    role_id: &str,
    role: &Json,
    images: &Json,
    seen: &mut Vec<(String, String, String)>,
) -> Outcome<()> {
    fields(role, "image_id scope bindings", role_id)?;
    let image_id = item(role, "image_id")?;
    require(contains_key(images, image_id)?, || {
        format!("{role_id}: unknown image_id")
    })?;
    let scope = item(role, "scope")?;
    require(one_of(scope, &["ordinary", "release"]), || {
        format!("{role_id}: invalid scope")
    })?;
    let bindings = item(role, "bindings")?
        .as_array()
        .filter(|list| !list.is_empty());
    require(bindings.is_some(), || format!("{role_id}: no bindings"))?;
    let release = is_str(scope, "release");
    for binding in bindings.unwrap_or_default() {
        fields(
            binding,
            "workflow job matrix_selector image_expression epoch_field",
            role_id,
        )?;
        let at = |key: &str| binding.get(key).unwrap_or(&Json::Null);
        let workflow = at("workflow")
            .as_str()
            .filter(|name| is_workflow_name(name));
        require(workflow.is_some(), || {
            format!("{role_id}: invalid workflow")
        })?;
        let job = at("job").as_str().filter(|name| is_identifier(name));
        require(job.is_some(), || format!("{role_id}: invalid job"))?;
        let selector = at("matrix_selector");
        let release_binding =
            workflow == Some("release.yml") || eq(selector, &release_tag("nonempty"));
        require(release_binding == release, || {
            format!("{role_id}: release scope mismatch")
        })?;
        let allowed = *selector == Json::Null
            || eq(selector, &cuda_major("12"))
            || eq(selector, &cuda_major("13"))
            || is_conditional(selector);
        require(allowed, || {
            format!("{role_id}: unsupported matrix selector")
        })?;
        let expression = at("image_expression").as_str();
        require(
            expression
                .is_some_and(|text| text.matches("{image}").count() == 1 && !text.contains('\n')),
            || format!("{role_id}: image_expression must contain one image placeholder"),
        )?;
        let epoch_field = at("epoch_field");
        let epoch_ok =
            *epoch_field == Json::Null || one_of(epoch_field, &["pinned_epoch", "toolchain_epoch"]);
        require(epoch_ok, || format!("{role_id}: invalid epoch_field"))?;
        let epoch = item(item_by(images, image_id)?, "native_toolchain_epoch")?;
        require(*epoch_field == Json::Null || *epoch != Json::Null, || {
            format!("{role_id}: native epoch is unknown")
        })?;
        let key = (
            workflow.unwrap_or_default().to_owned(),
            job.unwrap_or_default().to_owned(),
            repr(Some(selector)),
        );
        seen.push(key);
    }
    Ok(())
}
