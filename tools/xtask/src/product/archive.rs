use super::{archive_tar, archive_zip, compose};
use crate::repository::check_report::CheckReport;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum Kind {
    #[serde(rename = "tar.gz")]
    TarGz,
    Zip,
}

#[derive(Serialize)]
pub(super) struct Entry {
    pub(super) name: String,
    pub(super) path: PathBuf,
    pub(super) directory: bool,
    pub(super) size: u64,
    pub(super) mode: u32,
    pub(super) sha256: Option<String>,
}

#[derive(Serialize)]
struct Plan {
    kind: Kind,
    source: PathBuf,
    archive: PathBuf,
    entries: Vec<Entry>,
}

#[derive(Deserialize)]
struct ProductManifest {
    contract: String,
    mesh_version: String,
    backend: String,
    host: ProductPath,
    runtime: ProductPath,
}

#[derive(Deserialize)]
struct ProductPath {
    path: String,
}

fn member_path(source: &Path, relative: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe product manifest path: {relative}"));
    }
    Ok(source.join(path))
}

fn verify_product(source: &Path) -> Result<(), String> {
    let manifest: ProductManifest = serde_json::from_slice(
        &fs::read(source.join("product-manifest.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("malformed product manifest: {error}"))?;
    if manifest.contract != "mesh-llm-product-v2" {
        return Err("unsupported product manifest contract".into());
    }
    let host = member_path(source, &manifest.host.path)?;
    let runtime = member_path(source, &manifest.runtime.path)?;
    if !matches!(manifest.host.path.as_str(), "mesh-llm" | "mesh-llm.exe")
        || runtime.parent() != Some(source.join("native-runtimes").as_path())
    {
        return Err("product manifest does not describe one host and native runtime".into());
    }
    let args = [
        "--bundle".to_owned(),
        source.to_string_lossy().into_owned(),
        "--host".to_owned(),
        host.to_string_lossy().into_owned(),
        "--runtime".to_owned(),
        runtime.to_string_lossy().into_owned(),
        "--version".to_owned(),
        manifest.mesh_version,
        "--backend".to_owned(),
        manifest.backend,
        "--check".to_owned(),
    ];
    let report = compose::run(&args);
    if report.code != 0 {
        return Err(report.stderr.trim_end().to_owned());
    }
    Ok(())
}

fn plan(args: &[String]) -> Result<Plan, String> {
    let [source, archive, kind] = args else {
        return Err(
            "usage: product archive-plan|archive-write SOURCE_DIR ARCHIVE_PATH tar.gz|zip".into(),
        );
    };
    let kind = match kind.as_str() {
        "tar.gz" => Kind::TarGz,
        "zip" => Kind::Zip,
        other => return Err(format!("unsupported archive kind: {other}")),
    };
    let source = PathBuf::from(source);
    let archive = PathBuf::from(archive);
    let root = source
        .file_name()
        .ok_or("source must have a bundle directory name")?;
    if root != "mesh-bundle" {
        return Err("source must be named mesh-bundle".into());
    }
    if !source.is_dir()
        || fs::symlink_metadata(&source)
            .map_err(|error| error.to_string())?
            .file_type()
            .is_symlink()
    {
        return Err("source must be a real directory".into());
    }
    let source_absolute = source.canonicalize().map_err(|error| error.to_string())?;
    if archive
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err("archive path may not traverse parent directories".into());
    }
    let archive_absolute = std::env::current_dir()
        .map_err(|error| error.to_string())?
        .join(&archive);
    if archive_absolute.starts_with(&source_absolute)
        || archive.starts_with(&source)
        || archive
            .parent()
            .filter(|parent| parent.exists())
            .and_then(|parent| parent.canonicalize().ok())
            .is_some_and(|parent| parent.starts_with(&source_absolute))
    {
        return Err("archive output may not be inside the source directory".into());
    }
    let sidecar = archive.with_file_name(format!(
        "{}.sha256",
        archive
            .file_name()
            .ok_or("archive path has no filename")?
            .to_string_lossy()
    ));
    if sidecar.starts_with(&source_absolute)
        || sidecar
            .parent()
            .filter(|parent| parent.exists())
            .and_then(|parent| parent.canonicalize().ok())
            .is_some_and(|parent| parent.starts_with(&source_absolute))
    {
        return Err("archive sidecar may not be inside the source directory".into());
    }
    let mut entries = Vec::new();
    collect(&source, "mesh-bundle", &mut entries)?;
    if entries.len() > 4096 {
        return Err("archive exceeds 4096 entries".into());
    }
    Ok(Plan {
        kind,
        source,
        archive,
        entries,
    })
}

fn collect(path: &Path, name: &str, entries: &mut Vec<Entry>) -> Result<(), String> {
    if entries.len() >= 4096 {
        return Err("archive exceeds 4096 entries".into());
    }
    let metadata = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if metadata.file_type().is_symlink() {
        return Err(format!("archive input is a symlink: {}", path.display()));
    }
    if !(metadata.is_file() || metadata.is_dir()) {
        return Err(format!(
            "archive input is not a regular file or directory: {}",
            path.display()
        ));
    }
    entries.push(Entry {
        name: format!("{name}{}", if metadata.is_dir() { "/" } else { "" }),
        path: path.to_owned(),
        directory: metadata.is_dir(),
        size: metadata.len(),
        mode: mode(&metadata),
        sha256: if metadata.is_file() {
            Some(super::digest::file_sha256(path).map_err(|failure| failure.error.to_string())?)
        } else {
            None
        },
    });
    if metadata.is_dir() {
        let mut children = fs::read_dir(path)
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        children.sort_by_key(|child| child.file_name());
        for child in children {
            let filename = child.file_name();
            let filename = filename
                .to_str()
                .ok_or("archive member name is not UTF-8")?;
            if filename.contains(['\\', '\n', '\r', '\0']) {
                return Err("archive member name is not portable".into());
            }
            collect(&child.path(), &format!("{name}/{filename}"), entries)?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn mode(metadata: &fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o777
}

#[cfg(not(unix))]
fn mode(_metadata: &fs::Metadata) -> u32 {
    0o644
}

pub(super) fn run(args: &[String], write: bool) -> CheckReport {
    let result = (|| -> Result<String, String> {
        let plan = plan(args)?;
        if write {
            verify_product(&plan.source)?;
            if let Some(parent) = plan.archive.parent() {
                fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
            let temporary = plan.archive.with_extension("archive-partial");
            let result = (|| {
                let file = fs::File::create(&temporary).map_err(|error| error.to_string())?;
                match plan.kind {
                    Kind::TarGz => archive_tar::write(file, &plan.entries),
                    Kind::Zip => archive_zip::write(file, &plan.entries),
                }
            })();
            if let Err(error) = result.and_then(|()| verify_product(&plan.source)) {
                let _cleanup = fs::remove_file(&temporary);
                return Err(error);
            }
            fs::rename(&temporary, &plan.archive).map_err(|error| error.to_string())?;
            let mut input = fs::File::open(&plan.archive).map_err(|error| error.to_string())?;
            let mut digest = Sha256::new();
            let mut buffer = [0_u8; 65536];
            loop {
                let count = input.read(&mut buffer).map_err(|error| error.to_string())?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
            }
            let name = plan
                .archive
                .file_name()
                .ok_or("archive path has no filename")?
                .to_string_lossy();
            fs::write(
                plan.archive.with_file_name(format!("{name}.sha256")),
                format!("{}  {name}\n", hex::encode(digest.finalize())),
            )
            .map_err(|error| error.to_string())?;
            Ok(String::new())
        } else {
            serde_json::to_string(&plan)
                .map(|json| format!("{json}\n"))
                .map_err(|error| error.to_string())
        }
    })();
    match result {
        Ok(stdout) => CheckReport::success(stdout),
        Err(message) => CheckReport::failure(String::new(), format!("{message}\n")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn migration_product_archive_rejects_same_size_change_after_planning()
    -> Result<(), Box<dyn std::error::Error>> {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "xtask-archive-drift-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let source = root.join("mesh-llm");
        fs::write(&source, b"host")?;
        let entry = Entry {
            name: "mesh-bundle/mesh-llm".into(),
            path: source.clone(),
            directory: false,
            size: 4,
            mode: 0o755,
            sha256: Some(hex::encode(Sha256::digest(b"host"))),
        };
        fs::write(&source, b"HOST")?;
        let tar = archive_tar::write(
            fs::File::create(root.join("drift.tar.gz"))?,
            &[Entry {
                path: source.clone(),
                ..entry
            }],
        );
        let zip = archive_zip::write(
            fs::File::create(root.join("drift.zip"))?,
            &[Entry {
                name: "mesh-bundle/mesh-llm".into(),
                path: source.clone(),
                directory: false,
                size: 4,
                mode: 0o755,
                sha256: Some(hex::encode(Sha256::digest(b"host"))),
            }],
        );
        let original = fs::read(&source)?;
        fs::remove_dir_all(root)?;
        assert!(tar.is_err_and(|error| error.contains("archive input changed")));
        assert!(zip.is_err_and(|error| error.contains("archive input changed")));
        assert_eq!(original, b"HOST");
        Ok(())
    }
}
