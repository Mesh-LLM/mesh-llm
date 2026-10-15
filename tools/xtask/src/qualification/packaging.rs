//! `qualify packaging`: verify final composed product bytes and archive, then
//! emit packaging suite evidence.

use super::cli_input;
use super::compose;
use super::contract::{self, Json};
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

const USAGE: &str = "qualify packaging --product-dir <dir> --product-archive <file> --row-id <row> --source-sha <sha> --evidence <file>";

/// The archive member census must equal the composed product tree exactly:
/// no duplicates, no absolute or `..` members, files and directories only,
/// and byte-identical file payloads.
fn verify_archive(archive: &Path, product_dir: &Path) -> Result<String, String> {
    let mut expected: BTreeMap<String, PathBuf> = BTreeMap::new();
    collect_product_files(product_dir, product_dir, &mut expected)?;
    let bytes =
        std::fs::read(archive).map_err(|error| format!("{}: {error}", archive.display()))?;
    let decoder = flate2::read::MultiGzDecoder::new(&bytes[..]);
    let mut package = tar::Archive::new(decoder);
    let mut actual = Vec::new();
    let mut entries = package
        .entries()
        .map_err(|error| format!("invalid product archive: {error}"))?;
    while let Some(entry) = entries.next().transpose().map_err(|e| e.to_string())? {
        let kind = entry.header().entry_type();
        contract::require(
            kind == tar::EntryType::Regular || kind == tar::EntryType::Directory,
            "standalone archive contains an unsafe member",
        )?;
        let name = String::from_utf8(entry.path_bytes().into_owned())
            .map_err(|_| "archive member name is not UTF-8".to_owned())?;
        let name = name.trim_end_matches('/').to_owned();
        contract::require(
            !name.starts_with('/') && !name.split('/').any(|part| part == "..") && !name.is_empty(),
            "standalone archive contains an unsafe member",
        )?;
        if kind == tar::EntryType::Regular {
            let mut payload = Vec::new();
            entry
                .take(1024 * 1024 * 1024)
                .read_to_end(&mut payload)
                .map_err(|error| error.to_string())?;
            actual.push((name, payload));
        }
    }
    contract::require(
        actual
            .iter()
            .map(|(name, _)| name)
            .collect::<BTreeSet<_>>()
            .len()
            == actual.len(),
        "standalone archive file census differs from composed product",
    )?;
    let expected_names: BTreeSet<&String> = expected.keys().collect();
    let actual_names: BTreeSet<&String> = actual.iter().map(|(name, _)| name).collect();
    contract::require(
        expected_names == actual_names,
        "standalone archive file census differs from composed product",
    )?;
    for (name, payload) in &actual {
        let source = expected
            .get(name)
            .ok_or_else(|| format!("standalone archive member is unreadable: {name}"))?;
        contract::require(
            sha256_bytes(payload) == contract::digest(source)?,
            &format!("standalone archive member differs from product: {name}"),
        )?;
    }
    contract::digest(archive)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    hex::encode(sha2::Sha256::digest(bytes))
}

fn collect_product_files(
    root: &Path,
    directory: &Path,
    files: &mut BTreeMap<String, PathBuf>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        contract::require(!kind.is_symlink(), "standalone product contains a symlink")?;
        let relative = path
            .strip_prefix(root)
            .map_err(|_| "product path escapes its root".to_owned())?
            .iter()
            .map(|part| part.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        if kind.is_dir() {
            collect_product_files(root, &path, files)?;
        } else if kind.is_file() {
            files.insert(relative, path);
        }
    }
    Ok(())
}

fn verify(
    root: &Path,
    product_dir: &Path,
    archive: &Path,
    source_sha: &str,
    row_id: &str,
) -> Result<Value, String> {
    let row = contract::catalog_row(root, row_id)?;
    let manifest_path = product_dir.join("product-manifest.json");
    let product = contract::load_json(&manifest_path)?;
    contract::require(
        product.get("schema_version").and_then(Json::as_int) == Some(1)
            && product.get("contract").and_then(Json::as_str) == Some("skippy-product-v1"),
        "invalid standalone product manifest",
    )?;
    for (key, expected) in [
        ("source_sha", source_sha.to_owned()),
        ("target", contract::row_field(&row, "target")?.to_owned()),
        ("backend", contract::row_field(&row, "backend")?.to_owned()),
    ] {
        contract::require(
            product.get(key).and_then(Json::as_str) == Some(expected.as_str()),
            &format!("product {key} differs from selected row"),
        )?;
    }
    contract::validate_product_bytes(&product, product_dir)?;
    let archive_sha256 = verify_archive(archive, product_dir)?;

    let cli_path = product
        .get("cli")
        .and_then(|cli| cli.get("path"))
        .and_then(Json::as_str)
        .ok_or("incomplete standalone product manifest")?;
    let binary = product_dir.join(cli_path);
    let cli_name = cli_path.rsplit('/').next().unwrap_or(cli_path);
    let temporary = tempfile::Builder::new()
        .prefix("skippy-ci-cli-verify-")
        .tempdir()
        .map_err(|error| error.to_string())?;
    let cli_dir = temporary.path();
    for name in [cli_name, "build-contract.json", "host-imports.json"] {
        std::fs::copy(product_dir.join(name), cli_dir.join(name))
            .map_err(|error| format!("{name}: {error}"))?;
    }
    std::fs::write(
        cli_dir.join(format!("{cli_name}.sha256")),
        format!("{}  {cli_name}\n", contract::digest(&binary)?),
    )
    .map_err(|error| error.to_string())?;
    let cli_build = cli_input::verify(cli_dir, source_sha)?;

    let runtime_path = product
        .get("runtime")
        .and_then(|runtime| runtime.get("path"))
        .and_then(Json::as_str)
        .ok_or("incomplete standalone product manifest")?;
    let runtime_dir = product_dir.join(runtime_path);
    let runtime_manifest = contract::load_json(&runtime_dir.join("manifest.json"))?;
    let runtime = compose::validate_pair(
        &cli_build,
        &runtime_manifest,
        source_sha,
        contract::row_field(&row, "target")?,
        contract::row_field(&row, "backend")?,
    )?;
    compose::verify_discovery(&binary, &runtime_dir, &runtime)?;
    let mut executed_cases: Vec<&str> = contract::suite_cases("packaging-runtime")
        .unwrap_or(&[])
        .to_vec();
    executed_cases.sort_unstable();
    Ok(json!({
        "schema_version": 1,
        "status": "passed",
        "source_sha": source_sha,
        "row_id": row_id,
        "suite": "packaging-runtime",
        "product_manifest_sha256": contract::digest(&manifest_path)?,
        "executed_cases": executed_cases,
        "models": {},
        "observations": {
            "archive_sha256": archive_sha256,
            "cli_sha256": contract::digest(&binary)?,
            "runtime_sha256": contract::tree_digest(&runtime_dir)?,
            "runtime_id": runtime.get("id").and_then(Json::as_str).unwrap_or_default(),
            "release_version": runtime.get("release_version").and_then(Json::as_str).unwrap_or_default(),
            "skippy_abi": runtime.get("skippy_abi").and_then(Json::as_str).unwrap_or_default(),
        },
    }))
}

pub(super) fn run(root: &Path, args: &[String]) -> CheckReport {
    const GRAMMAR: Grammar = Grammar {
        usage: USAGE,
        values: &[
            "--product-dir",
            "--product-archive",
            "--row-id",
            "--source-sha",
            "--evidence",
        ],
        flags: &[],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let missing: Vec<&str> = GRAMMAR
        .values
        .iter()
        .filter(|name| parsed.last(name).is_none())
        .copied()
        .collect();
    if !missing.is_empty() || !parsed.positionals.is_empty() {
        return GRAMMAR.error(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    let value = |name: &str| parsed.last(name).unwrap_or_default();
    let result = verify(
        root,
        Path::new(value("--product-dir")),
        Path::new(value("--product-archive")),
        value("--source-sha"),
        value("--row-id"),
    )
    .and_then(|evidence| {
        let evidence_path = Path::new(value("--evidence"));
        std::fs::write(evidence_path, contract::render(&evidence)?)
            .map_err(|error| format!("{}: {error}", evidence_path.display()))
    });
    match result {
        Ok(()) => CheckReport::success(String::new()),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("standalone packaging qualification failed: {error}\n"),
        ),
    }
}
