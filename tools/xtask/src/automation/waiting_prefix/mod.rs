//! Offline acceptance and input admission for the waiting-prefix A/B runner.
mod acceptance;
mod aggregation;
mod report;
mod requests;
mod telemetry;
#[cfg(test)]
mod tests;
mod workload_plan;

use crate::{automation::agentic_prompt_manifest::fixture_profile, command::DynResult};
use acceptance::{Aggregate, Contract};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{collections::BTreeMap, io::Write, path::Path};

const USAGE: &str = "cargo xtool automation waiting-prefix evaluate --comparison FILE --output FILE [--report FILE] (--contract FILE | --catalog FILE --profile NAME)\n  cargo xtool automation waiting-prefix {summarize|aggregate} --input FILE --output FILE\n  cargo xtool automation waiting-prefix execute-requests --input FILE --output FILE\n  cargo xtool automation waiting-prefix plan --catalog FILE --profile NAME --model-id ID --model-sha256 HASH [--contract FILE] [--prompt-manifest FILE] --output FILE\n  cargo xtool automation waiting-prefix validate-prompts FILE";

#[derive(Debug, Deserialize, Serialize)]
struct Prompt {
    family: String,
    prompt: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct PromptManifest {
    #[serde(default)]
    metadata: Map<String, Value>,
    prompts: Vec<Prompt>,
}

fn prompt_manifest(bytes: &[u8]) -> DynResult<PromptManifest> {
    let document: PromptManifest = serde_json::from_slice(bytes)?;
    if document.prompts.is_empty() {
        return Err("prompt manifest must contain at least one prompt".into());
    }
    if document
        .prompts
        .iter()
        .any(|item| item.family.trim().is_empty() || item.prompt.trim().is_empty())
    {
        return Err("prompt manifest requires nonempty family and prompt strings".into());
    }
    Ok(document)
}

#[derive(Deserialize)]
struct Comparison {
    aggregate: Vec<Aggregate>,
}

fn options<'a>(
    args: &'a [String],
    allowed: &[&str],
    required: &[&str],
) -> DynResult<BTreeMap<&'a str, &'a str>> {
    let mut options = BTreeMap::new();
    let mut rest = args;
    while !rest.is_empty() {
        let [flag, value, tail @ ..] = rest else {
            return Err("waiting-prefix option requires a value".into());
        };
        if !allowed.contains(&flag.as_str()) {
            return Err(format!("unknown waiting-prefix option {flag}").into());
        }
        if value.trim().is_empty()
            || value.starts_with("--")
            || options.insert(flag.as_str(), value.as_str()).is_some()
        {
            return Err(format!("empty or duplicate waiting-prefix option {flag}").into());
        }
        rest = tail;
    }
    for required in required {
        if !options.contains_key(required) {
            return Err(format!("missing {required}").into());
        }
    }
    Ok(options)
}

fn contract(options: &BTreeMap<&str, &str>) -> DynResult<Contract> {
    let input = match (
        options.get("--contract"),
        options.get("--catalog"),
        options.get("--profile"),
    ) {
        (Some(path), None, None) => {
            let document: Value = serde_json::from_slice(&std::fs::read(path)?)?;
            if document["schema_version"] != 1
                || document["name"].as_str().is_none_or(str::is_empty)
            {
                return Err("acceptance contract requires schema 1 and a nonempty name".into());
            }
            document["hardware_acceptance"].clone()
        }
        (None, Some(path), Some(name)) => {
            let document: Value = serde_json::from_slice(&std::fs::read(path)?)?;
            fixture_profile::resolve(&document, name)?["hardware_acceptance"].clone()
        }
        _ => return Err("select exactly one contract or catalog/profile pair".into()),
    };
    Ok(serde_json::from_value(input)?)
}

fn publish(path: &Path, bytes: &[u8]) -> DynResult<()> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut output = tempfile::NamedTempFile::new_in(parent)?;
    output.write_all(bytes)?;
    output.as_file().sync_all()?;
    output.persist(path)?;
    Ok(())
}

fn evaluate(args: &[String]) -> DynResult<()> {
    let options = options(
        args,
        &[
            "--comparison",
            "--output",
            "--report",
            "--contract",
            "--catalog",
            "--profile",
        ],
        &["--comparison", "--output"],
    )?;
    let contract = contract(&options)?;
    let comparison: Comparison = serde_json::from_slice(&std::fs::read(options["--comparison"])?)?;
    let result = acceptance::evaluate(&comparison.aggregate, &contract)?;
    let report = report::render(&comparison.aggregate, &result)?;
    let mut bytes = serde_json::to_vec_pretty(&result)?;
    bytes.push(b'\n');
    publish(Path::new(options["--output"]), &bytes)?;
    if let Some(path) = options.get("--report") {
        publish(Path::new(path), report.as_bytes())?;
    }
    if result.passed {
        Ok(())
    } else {
        Err("waiting-prefix hardware acceptance failed; see check results".into())
    }
}

fn measurement_command(verb: &str, args: &[String]) -> DynResult<()> {
    let options = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let bytes = std::fs::read(options["--input"])?;
    let mut output = match verb {
        "summarize" => {
            serde_json::to_vec_pretty(&telemetry::summarize(serde_json::from_slice(&bytes)?)?)?
        }
        "aggregate" => {
            #[derive(Serialize)]
            struct Document {
                aggregate: Vec<Aggregate>,
            }
            let aggregate = aggregation::aggregate(serde_json::from_slice(&bytes)?)?;
            serde_json::to_vec_pretty(&Document { aggregate })?
        }
        _ => return Err("unsupported waiting-prefix measurement command".into()),
    };
    output.push(b'\n');
    publish(Path::new(options["--output"]), &output)
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [help] if help == "--help" => {
            println!("{USAGE}");
            Ok(())
        }
        [verb, rest @ ..] if verb == "evaluate" => evaluate(rest),
        [verb, rest @ ..] if verb == "plan" => workload_plan::run(rest),
        [verb, rest @ ..] if verb == "execute-requests" => requests::run(rest),
        [verb, rest @ ..] if verb == "summarize" || verb == "aggregate" => {
            measurement_command(verb, rest)
        }
        [verb, path] if verb == "validate-prompts" => {
            let document = prompt_manifest(&std::fs::read(path)?)?;
            println!("{}", serde_json::to_string_pretty(&document)?);
            Ok(())
        }
        _ => Err(format!("usage: {USAGE}").into()),
    }
}
