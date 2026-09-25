//! Runtime rows, compiler seed and SDK Rust sections of the identity
//! catalog contract (the tail of `validate` in
//! `scripts/runner-image-identity.py`).

use crate::ci_operations::catalog_validation::{fields, image_digest, named_map};
use crate::ci_operations::identity_text::{
    is_identifier, is_pinned_rust_action, is_recipe_input, is_seed_prefix, is_workflow_name,
};
use crate::ci_operations::python_access::{
    Outcome, contains_key, eq, is_str, item, item_by, require, type_name,
};
use crate::ci_plan::document::Json;
use crate::prepared_input::python_value::display;

pub(crate) fn validate_runtime_rows(catalog: &Json, images: &Json) -> Outcome<()> {
    for (row_id, row) in named_map(item(catalog, "runtime_rows")?, "runtime_rows")? {
        fields(row, "image_id platform architecture backend", row_id)?;
        let image_id = item(row, "image_id")?;
        require(contains_key(images, image_id)?, || {
            format!("{row_id}: unknown image_id")
        })?;
        let architecture = item(row, "architecture")?;
        require(
            is_str(item(row, "platform")?, "linux")
                && (is_str(architecture, "amd64") || is_str(architecture, "arm64")),
            || format!("{row_id}: unsupported runtime platform"),
        )?;
        let backend = item(row, "backend")?;
        let known = ["cpu", "cuda", "rocm", "vulkan"]
            .iter()
            .any(|name| is_str(backend, name));
        require(known, || format!("{row_id}: unsupported runtime backend"))?;
        let image = item_by(images, image_id)?;
        let image_backend = display(Some(item(image, "backend")?));
        require(
            image_backend.starts_with(backend.as_str().unwrap_or_default()),
            || format!("{row_id}: image/backend mismatch"),
        )?;
        require(
            *item(image, "native_toolchain_epoch")? != Json::Null,
            || format!("{row_id}: unknown native epoch"),
        )?;
    }
    Ok(())
}

fn string_list(value: &Json) -> Option<&[Json]> {
    value.as_array().filter(|list| !list.is_empty())
}

fn unique(list: &[Json]) -> Outcome<bool> {
    let mut seen: Vec<String> = Vec::new();
    for entry in list {
        if matches!(entry, Json::Array(_) | Json::Object(_)) {
            return Err(format!("unhashable type: '{}'", type_name(entry)));
        }
        let key = format!(
            "{}:{}",
            type_name(entry),
            crate::prepared_input::python_value::repr(Some(entry))
        );
        if seen.contains(&key) {
            return Ok(false);
        }
        seen.push(key);
    }
    Ok(true)
}

fn recipe_inputs_valid(value: &Json) -> bool {
    string_list(value).is_some_and(|list| {
        list.iter()
            .all(|entry| entry.as_str().is_some_and(is_recipe_input))
    })
}

pub(crate) fn validate_compiler_seed(catalog: &Json, images: &Json, roles: &Json) -> Outcome<()> {
    let seed = item(catalog, "compiler_seed")?;
    fields(
        seed,
        "image_id architecture key_prefix recipe_inputs recipe workload_coverage publisher_role consumer_roles runtime_consumer",
        "compiler_seed",
    )?;
    let image_id = item(seed, "image_id")?;
    require(contains_key(images, image_id)?, || {
        "compiler_seed: unknown image_id".to_owned()
    })?;
    require(is_str(item(seed, "architecture")?, "amd64"), || {
        "compiler_seed: unsupported architecture".to_owned()
    })?;
    let short: String = image_digest(item_by(images, image_id)?)?
        .chars()
        .take(8)
        .collect();
    let prefix = item(seed, "key_prefix")?;
    require(
        prefix
            .as_str()
            .is_some_and(|text| is_seed_prefix(text, &short)),
        || "compiler_seed: key prefix/image mismatch".to_owned(),
    )?;
    let inputs = item(seed, "recipe_inputs")?;
    require(recipe_inputs_valid(inputs), || {
        "compiler_seed: invalid recipe_inputs".to_owned()
    })?;
    require(unique(inputs.as_array().unwrap_or_default())?, || {
        "compiler_seed: duplicate recipe input".to_owned()
    })?;
    let recipe = item(seed, "recipe")?;
    require(recipe.as_str().is_some_and(is_identifier), || {
        "compiler_seed: invalid recipe".to_owned()
    })?;
    require(*item(seed, "workload_coverage")? == Json::Null, || {
        "compiler_seed: workload coverage has not been qualified".to_owned()
    })?;
    let consumers = item(seed, "consumer_roles")?;
    let consumers_ok = match string_list(consumers) {
        Some(list) => unique(list)?,
        None => false,
    };
    require(consumers_ok, || {
        "compiler_seed: invalid consumer roles".to_owned()
    })?;
    let publisher = item(seed, "publisher_role")?;
    for role_id in std::iter::once(publisher).chain(consumers.as_array().unwrap_or_default()) {
        let compatible = contains_key(roles, role_id)? && {
            let role = item_by(roles, role_id)?;
            eq(item(role, "image_id")?, image_id) && is_str(item(role, "scope")?, "ordinary")
        };
        require(compatible, || {
            format!(
                "compiler_seed: incompatible role {}",
                display(Some(role_id))
            )
        })?;
    }
    let runtime = item(seed, "runtime_consumer")?;
    fields(
        runtime,
        "workflow job row_id",
        "compiler_seed.runtime_consumer",
    )?;
    let publisher_bindings = item(item_by(roles, publisher)?, "bindings")?;
    require(
        publisher_bindings
            .as_array()
            .is_some_and(|list| list.len() == 1),
        || "compiler_seed: publisher role must declare exactly one binding".to_owned(),
    )?;
    let valid_consumer = item(runtime, "workflow")?
        .as_str()
        .is_some_and(is_workflow_name)
        && item(runtime, "job")?.as_str().is_some_and(is_identifier);
    require(valid_consumer, || {
        "compiler_seed: invalid runtime consumer".to_owned()
    })?;
    let rows = item(catalog, "runtime_rows")?;
    let row_id = item(runtime, "row_id")?;
    let matches =
        contains_key(rows, row_id)? && eq(item(item_by(rows, row_id)?, "image_id")?, image_id);
    require(matches, || {
        "compiler_seed: runtime image mismatch".to_owned()
    })
}

pub(crate) fn validate_sdk_rust(catalog: &Json, roles: &Json) -> Outcome<()> {
    let sdk = item(catalog, "sdk_rust")?;
    fields(
        sdk,
        "role target rust_action_ref toolchain_epoch profile_linker cache_namespace cache_recipe_inputs",
        "sdk_rust",
    )?;
    let role = item(sdk, "role")?;
    require(contains_key(roles, role)?, || {
        "sdk_rust: unknown role".to_owned()
    })?;
    let bindings = item(item_by(roles, role)?, "bindings")?;
    require(
        bindings.as_array().is_some_and(|list| list.len() == 1),
        || "sdk_rust: role must declare exactly one binding".to_owned(),
    )?;
    for key in [
        "target",
        "toolchain_epoch",
        "profile_linker",
        "cache_namespace",
    ] {
        let value = item(sdk, key)?;
        let present = value
            .as_str()
            .is_some_and(|text| !crate::repository::python_text::strip(text).is_empty());
        require(present, || format!("sdk_rust.{key}: expected a string"))?;
    }
    let action = item(sdk, "rust_action_ref")?;
    require(action.as_str().is_some_and(is_pinned_rust_action), || {
        "sdk_rust: action must be pinned".to_owned()
    })?;
    let pin = action
        .as_str()
        .unwrap_or_default()
        .split('@')
        .nth(1)
        .unwrap_or_default();
    let epoch = item(sdk, "toolchain_epoch")?.as_str().unwrap_or_default();
    require(epoch.ends_with(pin), || {
        "sdk_rust: action/epoch mismatch".to_owned()
    })?;
    require(
        recipe_inputs_valid(item(sdk, "cache_recipe_inputs")?),
        || "sdk_rust: invalid cache_recipe_inputs".to_owned(),
    )?;
    Ok(())
}
