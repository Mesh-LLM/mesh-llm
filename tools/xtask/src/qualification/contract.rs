//! Shared Skippy qualification contract: the closed nine-row catalog, suite
//! case sets, identity digest formats, strict JSON loading, and product-byte
//! re-hashing. Every qualification verb fails closed through these helpers.
//!
//! This is the Rust owner of the contract module the legacy
//! `skippy/scripts/validate-ci-qualification.py` carried; nothing here
//! emulates Python behavior beyond the documented receipt semantics.

pub(crate) use crate::ci_plan::document::Json;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Number;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The nine core qualification rows. `ci/slices.yml` `runtime_rows` must name
/// exactly this set; any drift fails closed.
pub(crate) const CORE_ROWS: [&str; 9] = [
    "linux-cpu",
    "linux-cuda",
    "linux-rocm",
    "linux-vulkan",
    "macos-metal",
    "windows-cpu",
    "windows-cuda",
    "windows-rocm",
    "windows-vulkan",
];

/// Rows the protected hosted graph executes unconditionally; they can never be
/// recorded `hardware-unavailable`.
pub(crate) const REQUIRED_AVAILABLE_ROWS: [&str; 3] = ["linux-cpu", "macos-metal", "windows-cpu"];

/// Cases each suite must execute on a composed standalone product.
pub(crate) const SUITE_CASES: [(&str, &[&str]); 7] = [
    (
        "packaging-runtime",
        &[
            "archive-integrity",
            "imports",
            "abi",
            "discovery",
            "version",
            "no-driver",
        ],
    ),
    (
        "dense",
        &[
            "load",
            "prefill-decode",
            "stream",
            "continuation",
            "restart",
        ],
    ),
    (
        "recurrent",
        &["prefill-decode", "state-preservation", "restart"],
    ),
    (
        "moe",
        &[
            "expert-execution",
            "repeated-restore",
            "suffix-continuation",
        ],
    ),
    (
        "kv-cache",
        &[
            "dense-prefix-hit",
            "recurrent-prefix-hit",
            "suffix-continuation",
            "divergent-prefix",
            "isolation",
        ],
    ),
    (
        "system-one",
        &[
            "laya-goldens",
            "reader-contract",
            "negative-cases",
            "lifecycle",
        ],
    ),
    (
        "decisions",
        &[
            "endpoint-equivalence",
            "probability-contract",
            "negative-cases",
            "lifecycle",
        ],
    ),
];

/// The runtime-issued device token a qualified row must select, from the
/// runtime's own `backend_device_selected` event.
pub(crate) const EXPECTED_DEVICE: [(&str, &str); 5] = [
    ("cpu", "CPU"),
    ("cuda", "CUDA0"),
    ("rocm", "ROCm0"),
    ("vulkan", "Vulkan0"),
    ("metal", "MTL0"),
];

/// Capability tags each suite's model set must carry in
/// `ci/model-artifacts/registry.json`.
const MODEL_TAGS: [(&str, &[&str]); 6] = [
    ("dense", &["dense"]),
    ("recurrent", &["hybrid", "recurrent"]),
    ("moe", &["moe"]),
    ("kv-cache", &["dense", "hybrid"]),
    ("system-one", &["system-one"]),
    ("decisions", &["decision"]),
];

pub(crate) fn suite_cases(name: &str) -> Option<&'static [&'static str]> {
    SUITE_CASES
        .iter()
        .find(|(suite, _)| *suite == name)
        .map(|(_, cases)| *cases)
}

pub(crate) fn suite_names() -> BTreeSet<&'static str> {
    SUITE_CASES.iter().map(|(name, _)| *name).collect()
}

pub(crate) fn expected_device(backend: &str) -> Option<&'static str> {
    EXPECTED_DEVICE
        .iter()
        .find(|(kind, _)| *kind == backend)
        .map(|(_, device)| *device)
}

pub(crate) fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

pub(crate) fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn is_git_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// SHA-256 of one file's bytes.
pub(crate) fn digest(path: &Path) -> Result<String, String> {
    crate::product::digest::file_sha256(path).map_err(|failure| {
        crate::artifact::os_error_line(&failure.error, &failure.path.to_string_lossy())
    })
}

/// Tree digest over every regular file below `root` as
/// `len(path) (8 bytes, big-endian) || path || sha256(bytes)` in sorted
/// relative-POSIX-path order. Any symlink fails closed.
pub(crate) fn tree_digest(root: &Path) -> Result<String, String> {
    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort();
    let mut digest = Sha256::new();
    for relative in files {
        let bytes = relative.as_bytes();
        digest.update((bytes.len() as u64).to_be_bytes());
        digest.update(bytes);
        digest.update(hex::decode(file_digest(&root.join(&relative))?).map_err(|e| e.to_string())?);
    }
    Ok(hex::encode(digest.finalize()))
}

fn collect_files(root: &Path, directory: &Path, files: &mut Vec<String>) -> Result<(), String> {
    let entries = std::fs::read_dir(directory)
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let kind = entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", path.display()))?;
        require(!kind.is_symlink(), "standalone product contains a symlink")?;
        let relative = path
            .strip_prefix(root)
            .map_err(|_| "product path escapes its root".to_owned())?
            .iter()
            .map(|part| part.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        if kind.is_dir() {
            collect_files(root, &path, files)?;
        } else if kind.is_file() {
            files.push(relative);
        }
    }
    Ok(())
}

fn file_digest(path: &Path) -> Result<String, String> {
    let mut handle = std::fs::File::open(path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = handle
            .read(&mut buffer)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex::encode(digest.finalize()))
}

/// JSON object parsing that rejects duplicate keys, so a repeated key can
/// never hide an earlier value from the qualification contract.
struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Json;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Json, E> {
        Ok(Json::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Json, E> {
        Ok(Json::Number(value.into()))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Json, E> {
        Ok(Json::Number(value.into()))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Json, E> {
        Number::from_f64(value)
            .map(Json::Number)
            .ok_or_else(|| E::custom("non-finite number"))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Json, E> {
        Ok(Json::String(value.to_owned()))
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Json, E> {
        Ok(Json::String(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element::<StrictJson>()? {
            items.push(item.0);
        }
        Ok(Json::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        let mut entries: Vec<(String, Json)> = Vec::new();
        while let Some((key, value)) = map.next_entry::<String, StrictJson>()? {
            if entries.iter().any(|(name, _)| *name == key) {
                return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
            }
            entries.push((key, value.0));
        }
        Ok(Json::Object(entries))
    }
}

struct StrictJson(Json);

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StrictVisitor).map(StrictJson)
    }
}

/// Parse JSON bytes rejecting duplicate object keys at every depth.
pub(crate) fn parse_strict(bytes: &[u8]) -> Result<Json, String> {
    serde_json::from_slice::<StrictJson>(bytes)
        .map(|value| value.0)
        .map_err(|error| format!("invalid JSON: {error}"))
}

/// Load one JSON object file, rejecting duplicate keys and non-object roots.
pub(crate) fn load_json(path: &Path) -> Result<Json, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let value = parse_strict(&bytes)?;
    require(
        value.as_object().is_some(),
        &format!("{} must contain a JSON object", path.display()),
    )?;
    Ok(value)
}

/// Load the `ci/slices.yml` runtime catalog row for `row_id`, failing closed
/// when the catalog drifts from the nine core rows.
pub(crate) fn catalog_row(root: &Path, row_id: &str) -> Result<Json, String> {
    let catalog = load_json(&root.join("ci/slices.yml"))?;
    let rows = catalog
        .get("runtime_rows")
        .and_then(Json::as_array)
        .ok_or("core row catalog lacks runtime_rows")?;
    let mut by_id: Vec<(&str, &Json)> = Vec::new();
    for row in rows {
        let id = row
            .get("id")
            .and_then(Json::as_str)
            .ok_or("core row catalog row lacks an id")?;
        by_id.push((id, row));
    }
    let ids: BTreeSet<&str> = by_id.iter().map(|(id, _)| *id).collect();
    let expected: BTreeSet<&str> = CORE_ROWS.into_iter().collect();
    require(
        ids == expected && by_id.len() == CORE_ROWS.len(),
        "core row catalog drift",
    )?;
    by_id
        .into_iter()
        .find(|(id, _)| *id == row_id)
        .map(|(_, row)| row.clone())
        .ok_or_else(|| format!("unknown core row: {row_id}"))
}

pub(crate) fn row_field<'a>(row: &'a Json, field: &str) -> Result<&'a str, String> {
    row.get(field)
        .and_then(Json::as_str)
        .ok_or_else(|| format!("core row lacks {field}"))
}

/// Validate an evidence `models` map against the pinned registry: every model
/// must be registered with exactly one file pinned to the recorded digest, and
/// the suite's capability groups must be covered.
pub(crate) fn validate_models(root: &Path, name: &str, models: &Json) -> Result<(), String> {
    let registry = load_json(&root.join("ci/model-artifacts/registry.json"))?;
    let artifacts = registry
        .get("artifacts")
        .and_then(Json::as_array)
        .ok_or("model registry lacks artifacts")?;
    let entries = models
        .as_object()
        .ok_or_else(|| format!("{name} lacks model identities"))?;
    if name == "packaging-runtime" {
        return require(
            entries.is_empty(),
            "packaging-only suite must not claim a model",
        );
    }
    require(
        !entries.is_empty(),
        &format!("{name} lacks model identities"),
    )?;
    let mut tags_found: BTreeSet<String> = BTreeSet::new();
    for (artifact_id, model_hash) in entries {
        let artifact = artifacts
            .iter()
            .find(|artifact| artifact.get("id").and_then(Json::as_str) == Some(artifact_id))
            .ok_or_else(|| format!("{name} names an unregistered model"))?;
        let files = artifact
            .get("artifact")
            .and_then(|artifact| artifact.get("files"))
            .and_then(Json::as_array)
            .ok_or_else(|| format!("{name} model lacks pinned files"))?;
        require(
            files.len() == 1
                && files[0].get("sha256").and_then(Json::as_str) == model_hash.as_str(),
            &format!("{name} model digest differs from pinned registry"),
        )?;
        if let Some(tags) = artifact.get("capability_tags").and_then(Json::as_array) {
            tags_found.extend(
                tags.iter()
                    .filter_map(|tag| tag.as_str().map(str::to_owned)),
            );
        }
    }
    let has_any = |group: &[&str]| group.iter().any(|tag| tags_found.contains(*tag));
    if name == "kv-cache" {
        require(
            tags_found.contains("dense") && has_any(&["hybrid", "recurrent"]),
            "KV suite requires dense and recurrent models",
        )
    } else if name == "recurrent" {
        require(
            has_any(&["hybrid", "recurrent"]),
            "recurrent suite lacks a recurrent model",
        )
    } else {
        let required = MODEL_TAGS
            .iter()
            .find(|(suite, _)| *suite == name)
            .map(|(_, tags)| *tags)
            .ok_or_else(|| format!("unknown qualification suite: {name}"))?;
        require(
            required.iter().all(|tag| tags_found.contains(*tag)),
            &format!("{name} lacks required model capability"),
        )
    }
}

/// Resolve a manifest-recorded product path, rejecting absolute paths,
/// traversal, symlinked parents, and non-regular or missing files.
pub(crate) fn product_file(product_dir: &Path, name: &Json) -> Result<PathBuf, String> {
    let name = match name {
        Json::String(text) if !text.is_empty() => text,
        _ => return Err("invalid standalone product path".to_owned()),
    };
    let relative = Path::new(name);
    require(
        !relative.is_absolute()
            && !relative
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir)),
        "standalone product path escapes bundle",
    )?;
    let parts: Vec<String> = relative
        .components()
        .filter_map(|part| match part {
            std::path::Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    let mut prefix = product_dir.to_path_buf();
    for part in &parts {
        prefix.push(part);
        require(
            !prefix.is_symlink(),
            "standalone product path contains a symlink",
        )?;
    }
    let path = product_dir.join(relative);
    require(
        path.is_file() && !path.is_symlink(),
        &format!("missing standalone product file: {name}"),
    )?;
    Ok(path)
}

/// Resolve a manifest-recorded runtime directory with the same containment
/// rules as [`product_file`], requiring a real directory.
fn product_directory(product_dir: &Path, name: &Json) -> Result<PathBuf, String> {
    let name = match name {
        Json::String(text) if !text.is_empty() => text,
        _ => return Err("invalid standalone runtime path".to_owned()),
    };
    let relative = Path::new(name);
    require(
        !relative.is_absolute()
            && !relative
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir)),
        "standalone runtime path escapes bundle",
    )?;
    let mut prefix = product_dir.to_path_buf();
    for part in relative.components() {
        let std::path::Component::Normal(part) = part else {
            continue;
        };
        prefix.push(part);
        require(
            !prefix.is_symlink(),
            "standalone runtime path contains a symlink",
        )?;
    }
    let path = product_dir.join(relative);
    require(
        path.is_dir() && !path.is_symlink(),
        "standalone runtime directory is missing",
    )?;
    Ok(path)
}

/// Re-hash every byte the product manifest binds: CLI binary, build contract,
/// import report, runtime manifest, and the runtime tree digest.
pub(crate) fn validate_product_bytes(product: &Json, product_dir: &Path) -> Result<(), String> {
    let cli = product.get("cli");
    let runtime = product.get("runtime");
    let (Some(cli), Some(runtime)) = (cli, runtime) else {
        return Err("incomplete standalone product manifest".to_owned());
    };
    require(
        cli.as_object().is_some() && runtime.as_object().is_some(),
        "incomplete standalone product manifest",
    )?;
    let binary = product_file(product_dir, cli.get("path").unwrap_or(&Json::Null))?;
    require(
        digest(&binary)? == cli.get("sha256").and_then(Json::as_str).unwrap_or(""),
        "standalone CLI bytes differ from manifest",
    )?;
    let contract = product_file(product_dir, &Json::String("build-contract.json".to_owned()))?;
    require(
        digest(&contract)?
            == cli
                .get("build_contract_sha256")
                .and_then(Json::as_str)
                .unwrap_or(""),
        "standalone CLI contract bytes differ from manifest",
    )?;
    let imports = product_file(product_dir, &Json::String("host-imports.json".to_owned()))?;
    require(
        digest(&imports)?
            == cli
                .get("host_imports_sha256")
                .and_then(Json::as_str)
                .unwrap_or(""),
        "standalone CLI import report bytes differ from manifest",
    )?;
    let runtime_dir = product_directory(product_dir, runtime.get("path").unwrap_or(&Json::Null))?;
    let manifest = product_file(&runtime_dir, &Json::String("manifest.json".to_owned()))?;
    require(
        digest(&manifest)?
            == runtime
                .get("manifest_sha256")
                .and_then(Json::as_str)
                .unwrap_or(""),
        "standalone runtime manifest bytes differ",
    )?;
    require(
        tree_digest(&runtime_dir)? == runtime.get("sha256").and_then(Json::as_str).unwrap_or(""),
        "standalone runtime bytes differ from manifest",
    )?;
    Ok(())
}

/// Serialize a qualification JSON artifact with sorted keys, two-space
/// indent, and a trailing newline.
pub(crate) fn render(value: &serde_json::Value) -> Result<String, String> {
    let mut text = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    text.push('\n');
    Ok(text)
}

/// Atomically replace `path` with `bytes` via a sibling temporary file.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent", path.display()))?;
    std::fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .map_err(|error| format!("{}: {error}", parent.display()))?;
    use std::io::Write as _;
    temporary
        .write_all(bytes)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| format!("{}: {error}", path.display()))?;
    temporary
        .persist(path)
        .map(|_| ())
        .map_err(|error| format!("{}: {}", path.display(), error.error))
}

#[cfg(test)]
pub(crate) mod tests;
