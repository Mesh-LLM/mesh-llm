mod document;
mod fixture_catalog;
mod fixture_fetch;
mod fixture_fetch_command;
mod fixture_materialization;
pub(crate) mod fixture_profile;
mod options;
mod parquet_input;
mod publication;
mod selection;
mod structured_content;

#[cfg(test)]
mod fixture_fetch_tests;
#[cfg(test)]
mod fixture_tests;
#[cfg(test)]
mod tests;

use crate::command::DynResult;
use options::Options;

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!(
            "cargo xtool automation agentic-prompt-manifest --dataset-file FILE --dataset-revision REV --output FILE --source-dataset SOURCE [selection options]\n\
Selection options: --families N (8), --requests-per-family N (2), --min-isl N (8192), --max-isl N (12000, exclusive), --min-turns N (20). Repeat --source-dataset to admit multiple sources.\n\
Local fixture commands:\n\
  validate-fixtures CATALOG\n\
  show-fixture CATALOG PROFILE\n\
  check-fixture-inputs --catalog FILE --profile NAME --model-id ID --model-sha256 SHA256 [--prompt-manifest FILE]\n\
  materialize-fixture --catalog FILE --profile NAME --dataset-file FILE --output FILE\n\
Optional Hugging Face CLI commands:\n\
  fetch-fixture --catalog FILE --profile NAME --hf-bin ABSOLUTE_PATH [--cache-dir DIR] [--timeout SECONDS]\n\
  prepare-fixture --catalog FILE --profile NAME --hf-bin ABSOLUTE_PATH --output FILE [--cache-dir DIR] [--timeout SECONDS]\n\
Fetch and prepare download and strictly verify the pinned revision. The shared command timeout defaults to 600 seconds. Local commands read existing files and require no HF executable."
        );
        return Ok(());
    }
    if let [verb, rest @ ..] = args {
        if ["fetch-fixture", "prepare-fixture"].contains(&verb.as_str()) {
            return fixture_fetch_command::run(verb, rest);
        }
        if verb == "materialize-fixture" {
            return fixture_materialization::run(rest);
        }
        if verb == "check-fixture-inputs" {
            return fixture_profile::run(rest);
        }
    }
    match args {
        [verb, file] if verb == "validate-fixtures" => {
            let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(file)?)?;
            fixture_catalog::validate(&catalog)?;
            let profiles = catalog["profiles"]
                .as_object()
                .ok_or("missing fixture profiles")?
                .keys()
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::json!({"profiles":profiles,"status":"valid"})
            );
            return Ok(());
        }
        [verb, file, name] if verb == "show-fixture" => {
            let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(file)?)?;
            let profile = fixture_profile::resolve(&catalog, name)?;
            println!("{}", serde_json::to_string_pretty(profile)?);
            return Ok(());
        }
        _ => {}
    }
    let options = Options::parse(args)?;
    let rows = parquet_input::select(&options.dataset_file, &options.selection)?;
    let manifest = document::build_manifest(
        &rows,
        options.requests_per_family,
        &options.dataset_revision,
        &options.selection,
    )?;
    publication::publish(&publication::bytes(&manifest)?, &options.output)?;
    println!("{}", options.output.display());
    Ok(())
}
