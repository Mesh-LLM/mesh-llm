use std::{fs, path::Path};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{document, fixture_catalog as catalog, parquet_input, publication};
use crate::command::DynResult;

pub(super) fn materialize(
    input: &Value,
    profile_name: &str,
    dataset_file: &Path,
    output: &Path,
) -> DynResult<String> {
    catalog::validate(input)?;
    let selected = input
        .get("profiles")
        .and_then(|profiles| profiles.get(profile_name))
        .ok_or_else(|| format!("unknown scheduler fixture profile {profile_name}"))?;
    let corpus = catalog::object(selected, "corpus")?;
    if catalog::text(corpus, "kind")? != "hf" {
        return Err("selected profile does not use a Hugging Face corpus".into());
    }
    let policy = catalog::selection(catalog::object(corpus, "selection")?)?;
    let rows = parquet_input::select(dataset_file, &policy)?;
    publish_selected(input, profile_name, &rows, output)
}

pub(super) fn publish_selected(
    input: &Value,
    profile_name: &str,
    rows: &[super::selection::Trajectory],
    output: &Path,
) -> DynResult<String> {
    catalog::validate(input)?;
    let selected = super::fixture_profile::resolve(input, profile_name)?;
    let corpus = catalog::object(selected, "corpus")?;
    let dataset = &input["datasets"][catalog::text(corpus, "dataset")?];
    let policy = catalog::selection(catalog::object(corpus, "selection")?)?;
    // Identity serialization excludes message bodies and is compared before
    // parsing prompts or creating any output directory or temporary file.
    if serde_json::to_value(rows)? != corpus["rows"] {
        return Err("selected rows do not match pinned fixture provenance".into());
    }
    let requests = usize::try_from(catalog::integer(
        catalog::object(selected, "workload")?,
        "requests_per_family",
    )?)?;
    let mut manifest =
        document::build_manifest(rows, requests, catalog::text(dataset, "revision")?, &policy)?;
    manifest.metadata.dataset = catalog::text(dataset, "repo_id")?;
    let bytes = publication::bytes(&manifest)?;
    let actual = hex::encode(Sha256::digest(&bytes));
    if actual != catalog::text(corpus, "prompt_manifest_sha256")? {
        return Err(format!("prompt manifest SHA-256 mismatch: got {actual}").into());
    }
    publication::publish(&bytes, output)?;
    Ok(actual)
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let mut catalog_file = None;
    let mut profile = None;
    let mut dataset = None;
    let mut output = None;
    let mut remaining = args;
    while !remaining.is_empty() {
        let [flag, value, rest @ ..] = remaining else {
            return Err("materialize-fixture option requires a value".into());
        };
        let target = match flag.as_str() {
            "--catalog" => &mut catalog_file,
            "--profile" => &mut profile,
            "--dataset-file" => &mut dataset,
            "--output" => &mut output,
            _ => return Err(format!("unknown materialize-fixture option {flag}").into()),
        };
        if target.replace(value.as_str()).is_some() {
            return Err(format!("duplicate materialize-fixture option {flag}").into());
        }
        remaining = rest;
    }
    let input: Value =
        serde_json::from_slice(&fs::read(catalog_file.ok_or("missing --catalog")?)?)?;
    let output = output.ok_or("missing --output")?;
    let actual = materialize(
        &input,
        profile.ok_or("missing --profile")?,
        Path::new(dataset.ok_or("missing --dataset-file")?),
        Path::new(output),
    )?;
    println!("{}", json!({"output":output,"sha256":actual}));
    Ok(())
}
