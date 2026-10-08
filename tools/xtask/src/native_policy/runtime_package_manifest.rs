use crate::repository::text::repr;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub(super) struct Package {
    pub(super) root: PathBuf,
    pub(super) os: String,
    pub(super) arch: String,
    pub(super) backend: String,
    pub(super) target: String,
    pub(super) libraries: Vec<String>,
    pub(super) tools: Vec<String>,
    pub(super) primary: String,
    pub(super) min_glibc: Option<String>,
    pub(super) relocatable: Vec<String>,
}

fn object<'a>(value: &'a Value, error: &str) -> Result<&'a Map<String, Value>, String> {
    value.as_object().ok_or_else(|| error.to_owned())
}

fn string<'a>(value: Option<&'a Value>, error: &str) -> Result<&'a str, String> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| error.to_owned())
}

fn file(root: &Path, label: &str, raw: &str) -> Result<PathBuf, String> {
    if raw.is_empty() || raw.contains('\0') {
        return Err(format!("{label} path must be a non-empty string"));
    }
    if raw.contains('\\') {
        return Err(format!(
            "{label} path must use forward slashes inside the artifact: {raw}"
        ));
    }
    if raw.starts_with('/')
        || (raw
            .as_bytes()
            .first()
            .is_some_and(|byte| byte.is_ascii_alphabetic())
            && raw.as_bytes().get(1) == Some(&b':'))
        || raw.split('/').any(|part| part == "..")
    {
        return Err(format!(
            "{label} path must be relative inside the artifact: {raw}"
        ));
    }
    let candidate = root.join(raw);
    let resolved = candidate
        .canonicalize()
        .map_err(|_| format!("missing {label}: {}", candidate.display()))?;
    let artifact_root = root.canonicalize().map_err(|error| error.to_string())?;
    if !resolved.starts_with(artifact_root) {
        return Err(format!("{label} path resolves outside the artifact: {raw}"));
    }
    if !resolved.is_file() {
        return Err(format!("missing {label}: {}", candidate.display()));
    }
    Ok(resolved)
}

fn digest(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

fn canonical_sha(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl Package {
    pub(super) fn read(root: &Path) -> Result<Self, String> {
        let path = root.join("manifest.json");
        if !path.is_file() {
            return Err(format!("missing manifest: {}", path.display()));
        }
        let raw = fs::read(&path).map_err(|error| error.to_string())?;
        let document: Value = serde_json::from_slice(&raw).map_err(|error| error.to_string())?;
        if document.get("schema_version").and_then(Value::as_u64) != Some(2) {
            return Err("native runtime manifest requires schema_version 2; import legacy caches explicitly".to_owned());
        }
        let runtime = object(
            document
                .get("runtime")
                .ok_or("missing manifest field: runtime")?,
            "runtime must be an object",
        )?;
        let missing: Vec<&str> = [
            "id",
            "release_version",
            "skippy_abi",
            "platform",
            "backend",
            "libraries",
            "files",
        ]
        .into_iter()
        .filter(|field| !runtime.contains_key(*field))
        .collect();
        if !missing.is_empty() {
            return Err(format!(
                "missing runtime manifest field(s): {}",
                missing.join(", ")
            ));
        }
        let id = string(runtime.get("id"), "runtime id must be a non-empty string")?;
        for field in ["release_version", "skippy_abi"] {
            string(
                runtime.get(field),
                &format!("runtime {field} must be a non-empty string"),
            )?;
        }
        if root.file_name().and_then(|name| name.to_str()) != Some(id) {
            return Err("artifact directory name must match runtime id".to_owned());
        }
        let libraries = runtime
            .get("libraries")
            .and_then(Value::as_array)
            .filter(|entries| !entries.is_empty())
            .ok_or("runtime libraries must be a non-empty list")?;
        let platform = object(&runtime["platform"], "runtime platform must be an object")?;
        let os = string(
            platform.get("os"),
            "runtime platform must declare os and arch",
        )?;
        let arch = string(
            platform.get("arch"),
            "runtime platform must declare os and arch",
        )?;
        if !["linux", "macos", "windows"].contains(&os) {
            return Err(format!("unsupported runtime platform os: {}", repr(os)));
        }
        if !["x86_64", "aarch64", "arm"].contains(&arch) {
            return Err(format!("unsupported runtime platform arch: {}", repr(arch)));
        }
        let target = string(
            platform.get("target"),
            "runtime platform target must be a non-empty string",
        )?;
        let pair = match target {
            "aarch64-apple-darwin" => ("macos", "aarch64"),
            "x86_64-apple-darwin" => ("macos", "x86_64"),
            "x86_64-unknown-linux-gnu" | "x86_64-linux-android" => ("linux", "x86_64"),
            "aarch64-unknown-linux-gnu" | "aarch64-linux-android" => ("linux", "aarch64"),
            "armv7-unknown-linux-gnueabihf" | "armv7-linux-androideabi" => ("linux", "arm"),
            "x86_64-pc-windows-msvc" => ("windows", "x86_64"),
            _ => return Err(format!("unsupported runtime platform target: {target}")),
        };
        if (os, arch) != pair {
            return Err(format!(
                "runtime os/arch do not match target: {os}/{arch} != {}/{}",
                pair.0, pair.1
            ));
        }
        let min_glibc = platform.get("min_glibc").filter(|value| !value.is_null()).map(|value| {
            let text = value.as_str().unwrap_or_default();
            let valid = text.split_once('.').is_some_and(|(major, minor)| !major.is_empty() && !minor.is_empty() && major.bytes().all(|byte| byte.is_ascii_digit()) && minor.bytes().all(|byte| byte.is_ascii_digit()));
            if valid { Ok(text.to_owned()) } else { Err(format!("runtime platform min_glibc must be a major.minor version like '2.35', got {}", repr(text))) }
        }).transpose()?;
        if min_glibc.is_some() && os != "linux" {
            return Err(format!(
                "runtime platform min_glibc is only supported on linux, got os {}",
                repr(os)
            ));
        }
        let backend = object(&runtime["backend"], "runtime backend must be an object")?;
        let kind = string(backend.get("kind"), "runtime backend must declare kind")?;
        if !["cpu", "metal", "cuda", "rocm", "vulkan"].contains(&kind) {
            return Err(format!("unsupported runtime backend kind: {}", repr(kind)));
        }
        if (kind == "metal" && os != "macos")
            || (["cuda", "rocm", "vulkan"].contains(&kind) && os == "macos")
        {
            return Err(format!("runtime backend {kind} is unsupported on {os}"));
        }
        let library_paths: Vec<String> = libraries
            .iter()
            .map(|value| {
                string(Some(value), "library path must be a non-empty string").map(str::to_owned)
            })
            .collect::<Result<_, _>>()?;
        for relative in &library_paths {
            file(root, "library", relative)?;
        }
        let files = object(
            &runtime["files"],
            "runtime files and tools must be checksum maps",
        )?;
        let tool_checksums = match runtime.get("tools") {
            Some(value) => object(value, "runtime files and tools must be checksum maps")?,
            None => {
                static EMPTY: std::sync::LazyLock<Map<String, Value>> =
                    std::sync::LazyLock::new(Map::new);
                &EMPTY
            }
        };
        if files.is_empty() {
            return Err("runtime files must be a non-empty checksum map".to_owned());
        }
        let missing: Vec<&str> = library_paths
            .iter()
            .map(String::as_str)
            .filter(|relative| !files.contains_key(*relative))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "runtime libraries are missing file checksums: {}",
                missing.join(", ")
            ));
        }
        for (label, checksums) in [("file", files), ("tool", tool_checksums)] {
            for (relative, expected) in checksums {
                let path = file(root, label, relative)?;
                let expected = expected.as_str().unwrap_or_default();
                if !canonical_sha(expected) {
                    return Err(format!(
                        "{label} checksum must be a canonical SHA-256 for {relative}"
                    ));
                }
                if digest(&path)? != expected {
                    return Err(format!("{label} checksum mismatch for {relative}"));
                }
                #[cfg(unix)]
                if label == "tool" && os != "windows" {
                    use std::os::unix::fs::PermissionsExt;
                    if path
                        .metadata()
                        .map_err(|error| error.to_string())?
                        .permissions()
                        .mode()
                        & 0o111
                        == 0
                    {
                        return Err(format!("runtime tool is not executable: {relative}"));
                    }
                }
            }
        }
        let build = object(
            document.get("build").unwrap_or(&Value::Null),
            "runtime build metadata must be an object",
        )?;
        let primary = string(
            build.get("primary_library"),
            "build.primary_library must be a non-empty string",
        )?;
        if !library_paths.iter().any(|relative| relative == primary) {
            return Err("build.primary_library must be declared in runtime.libraries".to_owned());
        }
        let expected = string(
            build.get("library_sha256"),
            "build.library_sha256 must be a canonical SHA-256",
        )?;
        if !canonical_sha(expected) {
            return Err("build.library_sha256 must be a canonical SHA-256".to_owned());
        }
        if files.get(primary).and_then(Value::as_str) != Some(expected) {
            return Err(format!(
                "build.library_sha256 must match runtime.files for {primary}"
            ));
        }
        let actual = digest(&file(root, "primary library", primary)?)?;
        if actual != expected {
            return Err(format!(
                "library_sha256 mismatch for {primary}: {actual} != {expected}"
            ));
        }
        let relocatable = match build.get("relocatable_libraries") {
            Some(Value::Array(list)) if !list.is_empty() => list
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
            Some(Value::String(text)) if !text.is_empty() => {
                text.chars().map(|ch| ch.to_string()).collect()
            }
            _ => library_paths.clone(),
        };
        Ok(Self {
            root: root.to_path_buf(),
            os: os.to_owned(),
            arch: arch.to_owned(),
            backend: kind.to_owned(),
            target: target.to_owned(),
            libraries: library_paths,
            tools: tool_checksums.keys().cloned().collect(),
            primary: primary.to_owned(),
            min_glibc,
            relocatable,
        })
    }

    pub(super) fn entries(&self) -> impl Iterator<Item = &str> {
        self.libraries.iter().chain(&self.tools).map(String::as_str)
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;
    #[test]
    fn legacy_runtime_requires_explicit_import() {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("manifest.json"),
            br#"{"schema_version":1,"runtime":{"mesh_version":"1.0.0"}}"#,
        )
        .unwrap();
        let error = Package::read(root.path()).err().unwrap();
        assert!(error.contains("schema_version 2"));
    }
}
