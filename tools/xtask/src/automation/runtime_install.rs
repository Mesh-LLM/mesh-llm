use crate::command::DynResult;
use serde::Deserialize;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    runtime: Runtime,
}

#[derive(Deserialize)]
struct Runtime {
    id: String,
    #[serde(default)]
    release_version: Option<String>,
    libraries: Vec<PathBuf>,
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [source, cache] => {
            let target = install(Path::new(source), Path::new(cache))?;
            writeln!(
                crate::cli_output::stderr(),
                "Installed CI native runtime: {}",
                target.display()
            )?;
            Ok(())
        }
        _ => Err("usage: automation runtime-cache-install SOURCE CACHE".into()),
    }
}

fn segment(value: &str) -> DynResult<()> {
    if value.trim().is_empty()
        || Path::new(value).components().count() != 1
        || !matches!(
            Path::new(value).components().next(),
            Some(Component::Normal(_))
        )
        || value.contains(['/', '\\'])
    {
        return Err("runtime cache identity must be a nonempty single path segment".into());
    }
    Ok(())
}

fn install(source: &Path, cache: &Path) -> DynResult<PathBuf> {
    let source = source.canonicalize()?;
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(source.join("manifest.json"))?)?;
    if manifest.schema_version != 2 {
        return Err(
            "native runtime manifest requires schema_version 2; import legacy caches explicitly"
                .into(),
        );
    }
    segment(&manifest.runtime.id)?;
    let version = manifest
        .runtime
        .release_version
        .as_deref()
        .filter(|value| !value.is_empty())
        .unwrap_or("unknown");
    segment(version)?;
    if manifest.runtime.libraries.is_empty() {
        return Err("native runtime libraries are empty".into());
    }
    for library in &manifest.runtime.libraries {
        if library.is_absolute()
            || library
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err("runtime library must be a contained relative path".into());
        }
        let path = source.join(library).canonicalize()?;
        if !path.starts_with(&source) || !path.is_file() {
            return Err("native runtime library is missing or escapes its artifact".into());
        }
    }
    std::fs::create_dir_all(cache)?;
    let cache = cache.canonicalize()?;
    if cache.starts_with(&source) || source.starts_with(&cache) {
        return Err("runtime source and cache must not overlap".into());
    }
    let version_root = cache.join(version);
    if std::fs::symlink_metadata(&version_root)
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        return Err("runtime cache version must not be a symlink".into());
    }
    std::fs::create_dir_all(&version_root)?;
    let target = version_root.join(&manifest.runtime.id);
    match std::fs::symlink_metadata(&target) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err("runtime cache target must not be a symlink".into());
        }
        Ok(_) => std::fs::remove_dir_all(&target)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.into()),
    }
    copy_tree(&source, &target)?;
    Ok(target)
}

fn copy_tree(source: &Path, target: &Path) -> DynResult<()> {
    std::fs::create_dir(target)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err("runtime artifact must not contain symlinks".into());
        }
        let destination = target.join(entry.file_name());
        if kind.is_dir() {
            copy_tree(&entry.path(), &destination)?;
        } else if kind.is_file() {
            std::fs::copy(entry.path(), destination)?;
        } else {
            return Err("runtime artifact contains a special file".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(root: &Path, id: &str) -> PathBuf {
        let source = root.join("source");
        std::fs::create_dir_all(source.join("lib")).unwrap();
        std::fs::write(source.join("lib/runtime.so"), b"fixture-runtime").unwrap();
        std::fs::write(
            source.join("manifest.json"),
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 2,
                "runtime": {"id":id,"release_version":"1.2.3","mesh_version":"legacy-ignored","libraries":["lib/runtime.so"]}
            }))
            .unwrap(),
        )
        .unwrap();
        source
    }

    #[test]
    fn installs_exact_artifact_and_replaces_old_cache() {
        let root = tempfile::tempdir().unwrap();
        let source = fixture(root.path(), "cpu-fixture");
        let cache = root.path().join("cache");
        std::fs::create_dir_all(cache.join("1.2.3/cpu-fixture")).unwrap();
        std::fs::write(cache.join("1.2.3/cpu-fixture/stale"), b"old").unwrap();
        let target = install(&source, &cache).unwrap();
        assert_eq!(
            std::fs::read(target.join("lib/runtime.so")).unwrap(),
            b"fixture-runtime"
        );
        assert_eq!(
            std::fs::read(target.join("manifest.json")).unwrap(),
            std::fs::read(source.join("manifest.json")).unwrap()
        );
        assert_eq!(target, cache.join("1.2.3/cpu-fixture"));
        assert!(!cache.join("legacy-ignored").exists());
        assert!(!target.join("stale").exists());
    }

    #[test]
    fn rejects_identity_path_escape_without_deleting_cache() {
        let root = tempfile::tempdir().unwrap();
        let source = fixture(root.path(), "../escape");
        let cache = root.path().join("cache");
        assert!(install(&source, &cache).is_err());
        assert!(!cache.exists());
    }

    #[test]
    fn rejects_missing_library_before_replacing_cache() {
        let root = tempfile::tempdir().unwrap();
        let source = fixture(root.path(), "cpu");
        std::fs::remove_file(source.join("lib/runtime.so")).unwrap();
        let cache = root.path().join("cache");
        assert!(install(&source, &cache).is_err());
        assert!(!cache.exists());
    }

    #[test]
    fn rejects_legacy_and_noninteger_schemas_before_replacing_cache() {
        for schema in [
            None,
            Some("1"),
            Some("2.0"),
            Some("\"2\""),
            Some("true"),
            Some("null"),
        ] {
            let root = tempfile::tempdir().unwrap();
            let source = fixture(root.path(), "cpu");
            let prefix = schema.map_or_else(String::new, |schema| {
                format!("\"schema_version\":{schema},")
            });
            std::fs::write(source.join("manifest.json"), format!("{{{prefix}\"runtime\":{{\"id\":\"cpu\",\"release_version\":\"1.2.3\",\"libraries\":[\"lib/runtime.so\"]}}}}" )).unwrap();
            let cache = root.path().join("cache");
            let previous = cache.join("1.2.3/cpu");
            std::fs::create_dir_all(&previous).unwrap();
            std::fs::write(previous.join("preserved"), b"previous").unwrap();
            assert!(install(&source, &cache).is_err(), "schema {schema:?}");
            assert_eq!(
                std::fs::read(previous.join("preserved")).unwrap(),
                b"previous"
            );
        }
    }

    #[test]
    fn schema_two_missing_release_uses_unknown_without_legacy_alias() {
        let root = tempfile::tempdir().unwrap();
        let source = fixture(root.path(), "cpu");
        let path = source.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        manifest["runtime"]
            .as_object_mut()
            .unwrap()
            .remove("release_version");
        std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let cache = root.path().join("cache");
        assert_eq!(install(&source, &cache).unwrap(), cache.join("unknown/cpu"));
        assert!(!cache.join("legacy-ignored").exists());
    }
}
