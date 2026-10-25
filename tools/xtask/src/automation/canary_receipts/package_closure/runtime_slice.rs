//! Static realized runtime-slice admission checks, without native execution.
#[path = "runtime_slice/context.rs"]
mod context;
#[path = "runtime_slice/lexical.rs"]
mod lexical;
use super::model_boundaries::{mask, mask_with_preserved};
use crate::command::DynResult;
use lexical::{balanced, find, tokens, top_level, unbraced};
use std::{fs, path::Path};

pub(super) fn verify(native: &Path) -> DynResult<()> {
    let root = native.canonicalize()?;
    let path = root.join("src/skippy/model_loading.cpp");
    if !path.exists() {
        return Ok(());
    }
    if !fs::symlink_metadata(&path)?.is_file() || !path.canonicalize()?.starts_with(&root) {
        return Err("runtime admission source escapes prepared tree".into());
    }
    if fs::metadata(&path)?.len() > 8 * 1024 * 1024 {
        return Err("runtime admission source exceeds 8 MiB".into());
    }
    validate(&fs::read_to_string(path)?)
}

fn body<'a>(source: &'a str, guard: &str) -> DynResult<&'a str> {
    context::direct_body(source, guard)
}
fn invalid(source: &str, guard: &str, message: &str) -> DynResult<()> {
    let body = body(source, guard)?;
    let literal = format!("\"{message}\"");
    let executable = mask_with_preserved(body, &[&literal]);
    let lexed = tokens(&executable);
    if unbraced(&lexed)? {
        return Err("conditional unbraced admission failure path".into());
    }
    let top = top_level(&lexed);
    let free = find(&top, "llama_model_free(model);").ok_or("admission failure must free model")?;
    let declaration = format!("const char * message = {literal};");
    let message = find(&top, &declaration).ok_or("admission failure message missing")?;
    let error = find(
        &top,
        "skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);",
    )
    .ok_or("admission failure must report invalid argument")?;
    let returned = find(&top, "return SKIPPY_STATUS_INVALID_ARGUMENT;")
        .ok_or("admission failure must return invalid argument")?;
    if !(free < message && message < error && error < returned)
        || top.iter().position(|token| token.text == "return") != Some(returned)
    {
        return Err("admission failure path is conditional, unreachable or misordered".into());
    }
    Ok(())
}
fn boundary(source: &str, guard: &str, message: &str) -> DynResult<()> {
    let body = body(source, guard)?;
    let literal = format!("\"{message}\"");
    let executable = mask_with_preserved(body, &[&literal]);
    let lexed = tokens(&executable);
    if unbraced(&lexed)? {
        return Err("conditional unbraced frontier failure".into());
    }
    let top = top_level(&lexed);
    let returned = find(&top, &format!("return fail_boundary_load({literal});"))
        .ok_or("frontier must return realized boundary failure")?;
    if top.iter().position(|token| token.text == "return") != Some(returned) {
        return Err("frontier failure return is unreachable".into());
    }
    Ok(())
}
fn validate(source: &str) -> DynResult<()> {
    let admission = context::function_body(source)?;
    let masked = mask(admission);
    for line in masked.lines() {
        if let Some(directive) = line.trim_start().strip_prefix('#') {
            let directive = tokens(directive);
            if directive.first().is_some_and(|token| {
                ["if", "ifdef", "ifndef", "elif", "else", "endif"].contains(&token.text)
            }) {
                return Err("runtime admission contains preprocessor branches".into());
            }
        }
    }
    let admission_tokens = tokens(&masked);
    let end = find(&admission_tokens, "skippy_model * stage_model")
        .ok_or("missing runtime validation boundary")?;
    for comparison in admission_tokens[..end].windows(7) {
        if comparison[..4]
            .iter()
            .map(|token| token.text)
            .eq(["model", "-", ">", "arch"])
            && ["=", "!"].contains(&comparison[4].text)
            && comparison[5].text == "="
            && comparison[6].text.starts_with("LLM_ARCH_")
        {
            return Err("runtime admission depends on architecture allowlist".into());
        }
    }
    let stage_plan = context::direct_body(admission, "if (skippy_runtime_has_stage_plan(config))")?;
    let stage_masked = mask(stage_plan);
    let stage_tokens = tokens(&stage_masked);
    let stage_top = top_level(&stage_tokens);
    let mut previous = None;
    for (guard, message) in [
        (
            "if (config->layer_end > n_layer)",
            "layer_end exceeds model layer count",
        ),
        (
            "if (skippy_runtime_is_source_stage(config) != (config->layer_start == 0))",
            "admitted activation imports disagree with the source-stage range",
        ),
        (
            "if (skippy_runtime_is_terminal_stage(config) != (config->layer_end == n_layer))",
            "admitted activation exports disagree with the terminal-stage range",
        ),
    ] {
        invalid(stage_plan, guard, message)?;
        let position =
            find(&stage_top, guard).ok_or("stage guard is not direct capability child")?;
        if previous.is_some_and(|previous| position <= previous) {
            return Err("stage admission guards are out of contract order".into());
        }
        previous = Some(position);
    }
    for (outer, guard, message) in [
        (
            "if (skippy_runtime_has_stage_plan(config) && !skippy_runtime_is_terminal_stage(config))",
            "if (!build_boundary(false,stage_model->output_activation_boundary))",
            "stage graph output frontier does not match its admitted planner identities",
        ),
        (
            "if (skippy_runtime_has_stage_plan(config) && !skippy_runtime_is_source_stage(config))",
            "if (!build_boundary(true,stage_model->input_activation_boundary))",
            "stage graph input frontier does not match its admitted planner identities",
        ),
    ] {
        boundary(context::direct_body(admission, outer)?, guard, message)?;
    }
    Ok(())
}
#[cfg(test)]
#[path = "runtime_slice/tests.rs"]
mod tests;
