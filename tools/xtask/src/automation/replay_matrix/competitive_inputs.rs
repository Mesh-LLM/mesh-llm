//! Bounded supplied tokenizer materialization; acquisition/export capability is separate.
#[path = "competitive_inputs/projection.rs"]
mod projection;
#[path = "competitive_inputs/schema.rs"]
mod schema;
#[cfg(test)]
#[path = "competitive_inputs/tests.rs"]
mod tests;
use crate::{automation::command_interrupt::Interrupt, command::DynResult};
use schema::{Budget, Request};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write as _,
    path::Path,
    time::{Duration, Instant},
};

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    use crate::repository::{check_args::Grammar, check_report::CheckReport};
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix competitive-inputs-local --input PATH",
        values: &["--input"],
        flags: &["--help"],
    };
    let flags = match GRAMMAR.parse(args) {
        Ok(v) => v,
        Err(e) => return e.emit(),
    };
    if flags.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !flags.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let bytes = crate::automation::receipt_files::bounded(
        Path::new(flags.last("--input").ok_or("input required")?),
        1024 * 1024,
    )?;
    let request: Request = serde_json::from_slice(&bytes)?;
    request.validate()?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now()
        .checked_add(Duration::from_secs(request.timeout_seconds))
        .ok_or("deadline overflow")?;
    let mut receipt = json!({"schema_version":1,"status":"INCOMPLETE","request_sha256":schema::digest(&serde_json::to_vec(&request)?),"acquisition_performed":false,"tokenizer_export_performed":false,"families":[],"error":null,"output_owned":false});
    let result = execute(
        &request,
        &Budget {
            deadline,
            cancellation: &cancel,
        },
        &mut receipt,
    );
    let finish: DynResult<()> = interrupt.finish().map_err(|e| e.to_string().into());
    let decision = terminal(
        result,
        finish,
        &Budget {
            deadline,
            cancellation: &cancel,
        },
    );
    receipt["status"] = json!(if decision.is_ok() {
        "SUPPLIED_MATERIALIZED"
    } else {
        "FAILED"
    });
    receipt["error"] = decision
        .as_ref()
        .err()
        .map_or(Value::Null, |e| json!(e.to_string()));
    if receipt["output_owned"] == json!(true) {
        publish(
            &request.output_directory.join("materialization.json"),
            &receipt,
        )?;
    }
    decision
}
fn publish(path: &Path, value: &Value) -> DynResult<()> {
    let mut file = tempfile::NamedTempFile::new_in(path.parent().ok_or("receipt parent")?)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(path)
        .map_err(|_| "fresh receipt publication refused")?;
    Ok(())
}
fn execute(input: &Request, budget: &Budget<'_>, receipt: &mut Value) -> DynResult<()> {
    budget.check()?;
    let config_bytes = crate::automation::receipt_files::bounded(&input.config, 1024 * 1024)?;
    if schema::digest(&config_bytes) != input.config_sha256 {
        return Err("config pin mismatch".into());
    }
    let config: Value = serde_json::from_slice(&config_bytes)?;
    super::competitive_plan::config::admit(&config)?;
    let selected = schema::selected(&config, input)?;
    receipt["acquisition_plan"] = schema::acquisition_plan(&config, &selected);
    // All policy and family roster admission precedes output mutation.
    fs::create_dir(&input.output_directory)?;
    receipt["output_owned"] = json!(true);
    fs::create_dir(input.output_directory.join("tokenizers"))?;
    for model in selected {
        budget.check()?;
        let key = model["key"].as_str().ok_or("model key")?;
        let source = input
            .sources
            .iter()
            .find(|s| s.key == key)
            .ok_or("missing supplied tokenizer source")?;
        let output = input.output_directory.join("tokenizers").join(key);
        let row = projection::materialize(
            source,
            &output,
            model["tokenizer_sha256"].as_str().ok_or("tokenizer pin")?,
            input.maximum_source_bytes,
            budget,
        )?;
        receipt["families"]
            .as_array_mut()
            .ok_or("receipt rows")?
            .push(row);
    }
    budget.check()?;
    let after = crate::automation::receipt_files::bounded(&input.config, 1024 * 1024)?;
    if after != config_bytes {
        return Err("config custody changed".into());
    }
    Ok(())
}

fn terminal(result: DynResult<()>, finish: DynResult<()>, budget: &Budget<'_>) -> DynResult<()> {
    result.and(finish).and_then(|()| budget.check())
}
