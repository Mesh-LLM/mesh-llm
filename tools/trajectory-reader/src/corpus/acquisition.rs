use super::config::Source;
use crate::DynResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Debug, Serialize)]
pub(super) struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub format: Format,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Format {
    Parquet,
    Jsonl,
}
#[derive(Deserialize)]
struct ConversionManifest {
    schema_version: u32,
    sources: Vec<ConvertedSource>,
}
#[derive(Deserialize)]
struct ConvertedSource {
    dataset: String,
    revision: String,
    config: String,
    split: String,
    conversion_provenance: serde_json::Value,
    artifacts: Vec<DeclaredArtifact>,
}
#[derive(Deserialize)]
struct DeclaredArtifact {
    path: PathBuf,
    sha256: String,
    bytes: u64,
    format: Format,
}

pub(super) struct Acquired {
    pub artifacts: Vec<Artifact>,
    pub provenance: serde_json::Value,
}
pub(super) fn acquire(
    source: &Source,
    cache: &Path,
    manifest: Option<&Path>,
) -> DynResult<Acquired> {
    if let Some(manifest) = manifest {
        return converted(source, manifest);
    }
    let (owner, name) = source
        .dataset
        .split_once('/')
        .ok_or("dataset namespace missing")?;
    let client = hf_hub::HFClientBuilder::new()
        .cache_dir(cache)
        .request_timeout(Duration::from_secs(120))
        .build_sync()?;
    let repo = client.dataset(owner, name);
    let info = repo.info().revision(source.revision.clone()).send()?;
    if info.id != source.dataset || info.sha.as_deref() != Some(source.revision.as_str()) {
        return Err("dataset resolved revision mismatch".into());
    }
    let mut names = info
        .siblings
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.rfilename)
        .collect::<Vec<_>>();
    names.sort();
    let requested = requested_files(source, &names)?;
    let mut artifacts = Vec::new();
    for (name, format) in requested {
        if Path::new(&name)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err("unsafe repository artifact path".into());
        }
        let path = repo
            .download_file()
            .filename(name)
            .revision(source.revision.clone())
            .send()?;
        artifacts.push(inspect(&path, format)?);
    }
    Ok(Acquired {
        artifacts,
        provenance: serde_json::json!({"kind":"immutable_hf_repository","dataset":source.dataset,"revision":source.revision}),
    })
}
pub(super) fn requested_files(
    source: &Source,
    names: &[String],
) -> DynResult<Vec<(String, Format)>> {
    if let Some(path) = raw_path(source) {
        if !names.iter().any(|name| name == &path) {
            return Err("pinned dataset raw file missing".into());
        }
        return Ok(vec![(path, Format::Jsonl)]);
    }
    let matches = names
        .iter()
        .filter(|name| parquet_match(name, source))
        .cloned()
        .map(|path| (path, Format::Parquet))
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return Err(
            "no pinned repository Parquet; supply a revision-bound conversion manifest".into(),
        );
    }
    Ok(matches)
}
fn raw_path(source: &Source) -> Option<String> {
    match (
        source.dataset.as_str(),
        source.config.as_str(),
        source.split.as_str(),
    ) {
        ("bigcode/commitpackft", "python" | "rust", "train") => {
            Some(format!("data/{}/data.jsonl", source.config))
        }
        ("codeparrot/apps", "all", "train" | "test") => Some(format!("{}.jsonl", source.split)),
        _ => None,
    }
}
fn parquet_match(name: &str, source: &Source) -> bool {
    let path = Path::new(name);
    let Some(file) = path.file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    let split = file.starts_with(&format!("{}-", source.split))
        || path
            .parent()
            .and_then(Path::file_name)
            .and_then(|s| s.to_str())
            == Some(&source.split);
    name.ends_with(".parquet")
        && split
        && (source.config == "default"
            || path
                .components()
                .any(|p| p.as_os_str() == source.config.as_str()))
}
fn converted(source: &Source, path: &Path) -> DynResult<Acquired> {
    let mut bytes = Vec::new();
    regular(path)?
        .take(8 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err("conversion manifest exceeds 8MiB".into());
    }
    let manifest: ConversionManifest = serde_json::from_slice(&bytes)?;
    if manifest.schema_version != 1 {
        return Err("conversion manifest requires schema 1".into());
    }
    let matches = manifest
        .sources
        .iter()
        .filter(|s| {
            s.dataset == source.dataset
                && s.revision == source.revision
                && s.config == source.config
                && s.split == source.split
        })
        .collect::<Vec<_>>();
    let [selected] = matches.as_slice() else {
        return Err("conversion requires exactly one matching immutable dataset source".into());
    };
    if !selected.conversion_provenance.is_object()
        || selected
            .conversion_provenance
            .as_object()
            .is_none_or(|p| p.is_empty())
        || selected.artifacts.is_empty()
    {
        return Err("conversion provenance and artifacts required".into());
    }
    let mut names = BTreeSet::new();
    let mut artifacts = Vec::new();
    for item in &selected.artifacts {
        if item.path.is_absolute()
            || item
                .path
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            || !names.insert(&item.path)
        {
            return Err("conversion artifact requires unique safe relative path".into());
        }
        let artifact = inspect(
            &path
                .parent()
                .ok_or("manifest parent missing")?
                .join(&item.path),
            item.format,
        )?;
        if artifact.sha256 != item.sha256 || artifact.bytes != item.bytes {
            return Err("conversion artifact digest/size mismatch".into());
        }
        artifacts.push(artifact);
    }
    artifacts.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Acquired {
        artifacts,
        provenance: serde_json::json!({"kind":"declared_conversion","manifest_sha256":super::digest(&bytes),"conversion":selected.conversion_provenance}),
    })
}
pub(super) fn regular(path: &Path) -> DynResult<File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("dataset input must be a regular file".into());
    }
    Ok(file)
}
pub(super) fn inspect(path: &Path, format: Format) -> DynResult<Artifact> {
    let mut file = regular(path)?;
    let bytes = file.metadata()?.len();
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    let sha256 = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    Ok(Artifact {
        path: path.to_owned(),
        sha256,
        bytes,
        format,
    })
}
