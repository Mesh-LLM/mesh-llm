//! Independent numerical/output comparison; distinct from workload smoke shape checks.
mod native;
mod numeric;
mod options;
#[cfg(test)]
mod tests;
use crate::command::DynResult;
use options::{Options, Reference};
use serde_json::{Value, json};
use std::io::Write;

fn request(base: &str, path: &str, payload: &Value) -> DynResult<Vec<u8>> {
    let bytes = super::request(base, path, payload)?;
    if !serde_json::from_slice::<Value>(&bytes)?.is_object() {
        return Err("oracle HTTP response must be a JSON object".into());
    }
    Ok(bytes)
}
fn pair(
    candidate: &str,
    reference: &str,
    path: &str,
    payload: &Value,
) -> DynResult<(Vec<u8>, Vec<u8>)> {
    Ok((
        request(candidate, path, payload)?,
        request(reference, path, payload)?,
    ))
}
fn embeddings(candidate: &str, reference: &str, model: &str) -> DynResult<String> {
    let payload = json!({"model":model,"input":super::INPUTS,"encoding_format":"float"});
    let (a, b) = pair(candidate, reference, "/embeddings", &payload)?;
    let mut failures = Vec::new();
    let mut details = Vec::new();
    match numeric::embeddings(&a, &b, 3) {
        Ok(detail) => details.push(format!("batch {detail}")),
        Err(error) => failures.push(format!("batched: {error}")),
    }
    for (index, text) in super::INPUTS.iter().enumerate() {
        let (a, b) = pair(
            candidate,
            reference,
            "/embeddings",
            &json!({"model":model,"input":text,"encoding_format":"float"}),
        )?;
        match numeric::embeddings(&a, &b, 1) {
            Ok(detail) => details.push(format!("single[{index}] {detail}")),
            Err(error) => failures.push(format!("single[{index}]: {error}")),
        }
    }
    if !failures.is_empty() {
        return Err(failures.join("; ").into());
    }
    Ok(details.join("; "))
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let options = Options::parse(args)?;
    let detail = match &options.reference {
        Reference::Server(base) if options.class == "embedding" => {
            embeddings(&options.candidate, base, &options.model)?
        }
        Reference::Server(base) => {
            let (a, b) = pair(
                &options.candidate,
                base,
                "/rerank",
                &json!({"model":options.model,"query":super::RERANK_QUERY,"documents":super::RERANK_DOCUMENTS,"return_documents":true}),
            )?;
            numeric::rerank(&a, &b)?
        }
        Reference::Completion {
            executable,
            model_path,
        } => {
            let candidate = request(
                &options.candidate,
                "/completions",
                &json!({"model":options.model,"prompt":super::ENCODER_DECODER_PROMPT,"max_tokens":32,"temperature":0.0,"seed":1}),
            )?;
            native::compare(&candidate, &native::completion(executable, model_path)?)?
        }
    };
    writeln!(
        crate::cli_output::stdout(),
        "{} local-monolithic oracle passed: {detail}",
        options.class
    )?;
    Ok(())
}
