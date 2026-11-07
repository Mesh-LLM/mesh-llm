//! The DFlash draft conversion manifest, `data/dflash-drafts/manifest.json`.
//!
//! CI converts each listed draft at the pinned llama.cpp, checks the output
//! against the SHA-256 the entry states, and uploads it to the entry's
//! `meshllm/` repo (`.github/workflows/dflash-drafts-publish.yml`). A pairing
//! may only name a `meshllm/` draft that this manifest produces, so every draft
//! we host can be rebuilt from pinned sources.

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::{Map, Value, json};

use super::dflash_catalog::bootstrap_pairings;
use super::{DFlashDraft, DFlashPairing, load_dflash_pairing_index};

const MANIFEST_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/dflash-drafts/manifest.json"
);
const INDEX_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/dflash-pairings/index.json"
);

/// What `convert_hf_to_gguf.py --outtype` writes without a separate quantize step.
const OUTTYPES: [&str; 4] = ["q8_0", "bf16", "f16", "f32"];

/// A published draft: what a pairing pins and the manifest promises.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Output {
    repo: String,
    file: String,
    sha256: String,
    bytes: u64,
}

fn is_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A file or repo-name segment: no separators, no leading dot or dash.
fn is_name(value: &str) -> bool {
    value
        .bytes()
        .next()
        .is_some_and(|b| b.is_ascii_alphanumeric())
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn is_repo(value: &str) -> bool {
    value
        .split_once('/')
        .is_some_and(|(owner, name)| is_name(owner) && is_name(name))
}

fn object<'a>(
    value: &'a Value,
    what: &str,
    keys: &[&str],
) -> Result<&'a Map<String, Value>, String> {
    let map = value
        .as_object()
        .ok_or_else(|| format!("{what} is not an object"))?;
    if let Some(key) = map.keys().find(|key| !keys.contains(&key.as_str())) {
        return Err(format!("{what} has unknown field `{key}`"));
    }
    Ok(map)
}

fn string<'a>(map: &'a Map<String, Value>, what: &str, key: &str) -> Result<&'a str, String> {
    map.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{what} needs a string `{key}`"))
}

/// A Hugging Face source pinned by revision, with every file it needs.
fn check_source(value: &Value, what: &str) -> Result<(), String> {
    let source = object(value, what, &["repo", "revision", "files"])?;
    if !is_repo(string(source, what, "repo")?) {
        return Err(format!("{what} repo is not `owner/name`"));
    }
    if !is_hex(string(source, what, "revision")?, 40) {
        return Err(format!("{what} revision is not a 40-character commit SHA"));
    }
    let files = source
        .get("files")
        .and_then(Value::as_array)
        .filter(|files| !files.is_empty())
        .ok_or_else(|| format!("{what} lists no files"))?;
    let mut paths = BTreeSet::new();
    for file in files {
        let file = object(file, what, &["path", "sha256"])?;
        let path = string(file, what, "path")?;
        if !is_name(path) || !paths.insert(path) {
            return Err(format!(
                "{what} file `{path}` is not a unique top-level file name"
            ));
        }
        if !is_hex(string(file, what, "sha256")?, 64) {
            return Err(format!("{what} file `{path}` has no SHA-256"));
        }
    }
    Ok(())
}

/// Checks the manifest and returns the drafts it publishes.
fn manifest_outputs(manifest: &Value) -> Result<Vec<Output>, String> {
    let manifest = object(manifest, "the manifest", &["schema_version", "drafts"])?;
    if manifest.get("schema_version") != Some(&json!(1)) {
        return Err("the manifest is not schema_version 1".to_string());
    }
    let drafts = manifest
        .get("drafts")
        .and_then(Value::as_array)
        .ok_or("the manifest has no `drafts` array")?;
    let mut names = BTreeSet::new();
    let mut targets = BTreeSet::new();
    let mut outputs = Vec::new();
    for draft in drafts {
        let entry = object(
            draft,
            "a draft",
            &["name", "source", "target_model", "convert", "output"],
        )?;
        let name = string(entry, "a draft", "name")?;
        if !is_name(name) || !names.insert(name) {
            return Err(format!("draft name `{name}` is not a unique name"));
        }
        check_source(
            entry.get("source").unwrap_or(&Value::Null),
            &format!("{name} source"),
        )?;
        if let Some(target) = entry.get("target_model") {
            check_source(target, &format!("{name} target_model"))?;
        }

        let what = format!("{name} convert");
        let convert = object(
            entry.get("convert").unwrap_or(&Value::Null),
            &what,
            &["llama_cpp_upstream", "outtype"],
        )?;
        if !is_hex(string(convert, &what, "llama_cpp_upstream")?, 40) {
            return Err(format!("{what} llama_cpp_upstream is not a commit SHA"));
        }
        let outtype = string(convert, &what, "outtype")?;
        if !OUTTYPES.contains(&outtype) {
            return Err(format!(
                "{what} outtype `{outtype}` is not one of {OUTTYPES:?}"
            ));
        }

        let what = format!("{name} output");
        let output = object(
            entry.get("output").unwrap_or(&Value::Null),
            &what,
            &["repo", "file", "sha256", "bytes"],
        )?;
        let repo = string(output, &what, "repo")?;
        if !is_repo(repo) || !repo.starts_with("meshllm/") {
            return Err(format!("{what} repo `{repo}` is not a meshllm/ repo"));
        }
        let file = string(output, &what, "file")?;
        if !is_name(file) || !file.ends_with(".gguf") {
            return Err(format!("{what} file `{file}` is not a top-level .gguf"));
        }
        let sha256 = string(output, &what, "sha256")?;
        if !is_hex(sha256, 64) {
            return Err(format!("{what} has no SHA-256"));
        }
        let bytes = output
            .get("bytes")
            .and_then(Value::as_u64)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| format!("{what} needs a positive `bytes`"))?;
        if !targets.insert((repo, file)) {
            return Err(format!("{what} {repo}/{file} is produced twice"));
        }
        outputs.push(Output {
            repo: repo.to_string(),
            file: file.to_string(),
            sha256: sha256.to_string(),
            bytes,
        });
    }
    Ok(outputs)
}

/// Every pairing whose draft we host must name a draft the manifest produces,
/// byte for byte.
fn check_hosted_drafts(pairings: &[DFlashPairing], outputs: &[Output]) -> Result<(), String> {
    for pairing in pairings {
        let draft = &pairing.draft;
        if !draft.repo.starts_with("meshllm/") {
            continue;
        }
        let pinned = Output {
            repo: draft.repo.clone(),
            file: draft.file.clone(),
            sha256: draft.sha256.clone(),
            bytes: draft.bytes,
        };
        if !outputs.contains(&pinned) {
            return Err(format!(
                "the pairing draft {}/{} is not a conversion data/dflash-drafts/manifest.json produces; \
                 drafts under meshllm/ are published only by that pipeline",
                draft.repo, draft.file
            ));
        }
    }
    Ok(())
}

fn checked_in_manifest() -> Value {
    serde_json::from_slice(&std::fs::read(MANIFEST_PATH).unwrap()).unwrap()
}

fn laguna_entry() -> Value {
    json!({
        "name": "Laguna-XS-2.1-DFlash-Q8_0",
        "source": {
            "repo": "poolside/Laguna-XS-2.1-DFlash",
            "revision": "5c36361aab23c8ed3afbd079c10c426b677bc607",
            "files": [
                { "path": "config.json", "sha256": "22bdc5fd9fd41dc6945523fd504f1b70809af0f5ab26c09716e33616033ecf9a" },
                { "path": "model.safetensors", "sha256": "0b51e20d76200a80e636414f45fb51a5c0e13b0852d977ba2db788214c68f6b8" }
            ]
        },
        "target_model": {
            "repo": "poolside/Laguna-XS-2.1",
            "revision": "c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db",
            "files": [
                { "path": "config.json", "sha256": "de84031daa00903361001b0ed684aba7e574952f51de84fdc3da6d6239b451ea" },
                { "path": "tokenizer.json", "sha256": "807c53a95141e77c14e45f68c51db3f84d2ea6b555a6ea832bc99c88dae6a279" },
                { "path": "tokenizer_config.json", "sha256": "faf352639727d5f21d017ed014b28f2951557da280819b681c404fdc131101ee" }
            ]
        },
        "convert": {
            "llama_cpp_upstream": "620d2225cc436ab0b001e3200e590e3b286ef9ab",
            "outtype": "q8_0"
        },
        "output": {
            "repo": "meshllm/Laguna-XS-2.1-DFlash-Q8_0-GGUF",
            "file": "Laguna-XS-2.1-DFlash-Q8_0.gguf",
            "sha256": "015e033295b5b4fa6bf04230092cafecd7f84e05d01c190785723c436c3a8b3d",
            "bytes": 494735328
        }
    })
}

fn manifest_with(entry: Value) -> Value {
    json!({ "schema_version": 1, "drafts": [entry] })
}

fn error_with(edit: impl FnOnce(&mut Value)) -> String {
    let mut entry = laguna_entry();
    edit(&mut entry);
    manifest_outputs(&manifest_with(entry)).unwrap_err()
}

#[test]
fn the_checked_in_manifest_is_valid() {
    manifest_outputs(&checked_in_manifest()).unwrap();
}

/// A default pairing must not depend on a draft someone uploaded by hand.
#[test]
fn every_hosted_pairing_draft_comes_from_the_manifest() {
    let outputs = manifest_outputs(&checked_in_manifest()).unwrap();
    check_hosted_drafts(&bootstrap_pairings(), &outputs).unwrap();
    let index = load_dflash_pairing_index(Path::new(INDEX_PATH)).unwrap();
    check_hosted_drafts(&index.pairings, &outputs).unwrap();
}

#[test]
fn a_hosted_draft_must_match_its_manifest_entry_byte_for_byte() {
    let outputs = manifest_outputs(&manifest_with(laguna_entry())).unwrap();
    let mut pairing = bootstrap_pairings().remove(0);
    pairing.draft.repo = "meshllm/Laguna-XS-2.1-DFlash-Q8_0-GGUF".to_string();
    pairing.draft.file = "Laguna-XS-2.1-DFlash-Q8_0.gguf".to_string();
    pairing.draft.sha256 =
        "015e033295b5b4fa6bf04230092cafecd7f84e05d01c190785723c436c3a8b3d".to_string();
    pairing.draft.bytes = 494_735_328;
    check_hosted_drafts(std::slice::from_ref(&pairing), &outputs).unwrap();

    // Each pinned field must match on its own.
    for edit in [
        |draft: &mut DFlashDraft| draft.repo = "meshllm/other-draft".to_string(),
        |draft: &mut DFlashDraft| draft.file = "other.gguf".to_string(),
        |draft: &mut DFlashDraft| draft.bytes += 1,
    ] {
        let mut changed = pairing.clone();
        edit(&mut changed.draft);
        assert!(check_hosted_drafts(std::slice::from_ref(&changed), &outputs).is_err());
    }

    // A hand-published file in the same repo is refused.
    pairing.draft.sha256 =
        "40eb6bf48f42ebc1990a0ba8c74d2070a256e7f1842a58d8aa9b195e907b04e8".to_string();
    let error = check_hosted_drafts(std::slice::from_ref(&pairing), &outputs).unwrap_err();
    assert!(
        error.contains("meshllm/Laguna-XS-2.1-DFlash-Q8_0-GGUF"),
        "{error}"
    );

    // Drafts hosted by their publishers need no manifest entry.
    assert!(check_hosted_drafts(&bootstrap_pairings(), &[]).is_ok());
}

#[test]
fn a_draft_with_unpinned_sources_is_refused() {
    let error = error_with(|entry| entry["source"]["revision"] = json!("main"));
    assert!(error.contains("revision"), "{error}");
    let error = error_with(|entry| entry["source"]["files"][1]["sha256"] = json!("abc"));
    assert!(error.contains("model.safetensors"), "{error}");
    let error = error_with(|entry| entry["source"]["files"] = json!([]));
    assert!(error.contains("no files"), "{error}");
    let error =
        error_with(|entry| entry["target_model"]["files"][0]["path"] = json!("../config.json"));
    assert!(error.contains("top-level"), "{error}");
}

#[test]
fn a_draft_published_outside_meshllm_or_without_its_hash_is_refused() {
    let error = error_with(|entry| entry["output"]["repo"] = json!("someone/Laguna-XS"));
    assert!(error.contains("meshllm/"), "{error}");
    let error = error_with(|entry| entry["output"]["file"] = json!("q8/draft.gguf"));
    assert!(error.contains("top-level .gguf"), "{error}");
    let error = error_with(|entry| {
        entry["output"].as_object_mut().unwrap().remove("sha256");
    });
    assert!(error.contains("sha256"), "{error}");
    let error = error_with(|entry| entry["output"]["bytes"] = json!(0));
    assert!(error.contains("bytes"), "{error}");
}

#[test]
fn a_draft_conversion_must_be_one_the_workflow_runs() {
    let error = error_with(|entry| entry["convert"]["outtype"] = json!("q4_k_m"));
    assert!(error.contains("outtype"), "{error}");
    let error = error_with(|entry| entry["convert"]["llama_cpp_upstream"] = json!("b1234"));
    assert!(error.contains("llama_cpp_upstream"), "{error}");
    let error = error_with(|entry| entry["convert"]["extra_args"] = json!(["--vocab-only"]));
    assert!(error.contains("unknown field `extra_args`"), "{error}");
}

#[test]
fn two_drafts_cannot_share_a_name_or_an_output() {
    let mut second = laguna_entry();
    second["name"] = json!("Laguna-XS-2.1-DFlash-Q8_0-again");
    let manifest = json!({ "schema_version": 1, "drafts": [laguna_entry(), second] });
    let error = manifest_outputs(&manifest).unwrap_err();
    assert!(error.contains("produced twice"), "{error}");
    let manifest = json!({ "schema_version": 1, "drafts": [laguna_entry(), laguna_entry()] });
    let error = manifest_outputs(&manifest).unwrap_err();
    assert!(error.contains("unique name"), "{error}");
}
