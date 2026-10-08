#[path = "workload_production.rs"]
pub(super) mod production;

use super::{archive, executable, process, source};
use crate::{automation::canary_receipts::Digest, command::DynResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    pub(super) head: String,
    pub(super) worktree_sha256: Digest,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Record {
    path: String,
    sha256: Digest,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Producer {
    schema_version: u8,
    source: Source,
    files: BTreeMap<String, Record>,
}

/// Only verify_producer can construct this pre-snapshot proof.
pub(super) struct VerifiedProducer {
    digest: Digest,
    closure: PathBuf,
    native_head: String,
}
impl VerifiedProducer {
    pub(super) fn digest(&self) -> &Digest {
        &self.digest
    }
}

const FIXED: [(&str, &str); 8] = [
    ("candidate", "cargo/debug/skippy"),
    ("model_package", "cargo/debug/skippy-package-builder"),
    ("correctness", "cargo/debug/skippy-correctness"),
    ("topology_plan", "cargo/debug/skippy-topology-plan"),
    ("native_stamp", "native/.mesh-llm-build-stamp"),
    ("llama-server", "native/bin/llama-server"),
    ("llama-completion", "native/bin/llama-completion"),
    ("llama-tts", "native/bin/llama-tts"),
];

fn producer(bytes: &[u8]) -> DynResult<Producer> {
    let producer: Producer = serde_json::from_slice(bytes)?;
    source::revision(&producer.source.head)?;
    if producer.schema_version != 1 {
        return Err("unsupported workload producer schema".into());
    }
    let mut paths = BTreeSet::new();
    for record in producer.files.values() {
        archive::path(&record.path)?;
        if record.path == "producer.json" || !paths.insert(&record.path) {
            return Err("duplicate or self-referential workload member".into());
        }
    }
    for (key, path) in FIXED {
        if producer
            .files
            .get(key)
            .is_none_or(|record| record.path != path)
        {
            return Err("missing or relocated canonical workload artifact".into());
        }
    }
    let test = producer
        .files
        .get("test_binary")
        .ok_or("missing workload test executable")?;
    let relative = Path::new(&test.path);
    if relative.parent() != Some(Path::new("cargo/debug/deps"))
        || !relative
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("skippy_serving-"))
    {
        return Err("workload test artifact is outside its candidate target".into());
    }
    Ok(producer)
}

fn stamp(bytes: &[u8], native_head: &str) -> DynResult<()> {
    let mut fields = BTreeMap::new();
    for line in std::str::from_utf8(bytes)?.lines() {
        let (key, value) = line.split_once('=').ok_or("malformed native build stamp")?;
        if ["stamp-version", "patched-sha", "backend", "link-mode"].contains(&key)
            && fields.insert(key, value).is_some()
        {
            return Err("duplicate native build identity field".into());
        }
    }
    if fields.get("stamp-version") != Some(&"3")
        || fields.get("patched-sha") != Some(&native_head)
        || fields.get("backend") != Some(&"cpu")
        || fields.get("link-mode") != Some(&"static")
    {
        return Err("workload stamp differs from prepared static CPU native identity".into());
    }
    Ok(())
}

pub(super) fn archive(path: &Path, candidate: &str, native_head: &str) -> DynResult<()> {
    let mut file = File::open(path)?;
    let members = archive::scan(&mut file)?;
    let manifest = members
        .get("producer.json")
        .ok_or("workload archive has no producer manifest")?;
    let producer = producer(&archive::bytes(&mut file, manifest, 2 * 1024 * 1024)?)?;
    if producer.source.head != candidate || producer.source.worktree_sha256 != Digest::of_bytes(b"")
    {
        return Err("workload archive is not bound to the clean candidate snapshot".into());
    }
    if members.len() != producer.files.len() + 1
        || producer.files.values().any(|record| {
            members
                .get(&record.path)
                .is_none_or(|member| member.sha256 != record.sha256)
        })
    {
        return Err("workload archive differs from exact producer closure".into());
    }
    let native_stamp = &members[&producer.files["native_stamp"].path];
    stamp(
        &archive::bytes(&mut file, native_stamp, 64 * 1024)?,
        native_head,
    )?;
    for (key, record) in &producer.files {
        let member = &members[&record.path];
        if key != "native_stamp" {
            if !member.executable {
                return Err("workload executable has no execute bits".into());
            }
            executable::inspect(&mut file, member.start, member.size)?;
        }
    }
    for key in ["candidate", "test_binary"] {
        if members[&producer.files[key].path].modified <= native_stamp.modified {
            return Err("workload executable predates stamped native ABI".into());
        }
    }
    Ok(())
}

fn contained(root: &Path, relative: &str) -> DynResult<PathBuf> {
    archive::path(relative)?;
    let path = root.join(relative);
    let actual = path.canonicalize()?;
    if !actual.starts_with(root) || !fs::symlink_metadata(&path)?.is_file() {
        return Err("workload member escapes closure or is not a regular file".into());
    }
    Ok(actual)
}

fn files(root: &Path, producer: &Producer, native_head: &str) -> DynResult<()> {
    verify_files(root, producer, native_head, true)
}

fn verify_files(
    root: &Path,
    producer: &Producer,
    native_head: &str,
    package: bool,
) -> DynResult<()> {
    let root = root.canonicalize()?;
    for (key, record) in &producer.files {
        let path = contained(&root, &record.path)?;
        if Digest::of_file(&path)? != record.sha256 {
            return Err("workload artifact changed since producer admission".into());
        }
        if key != "native_stamp" {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if fs::metadata(&path)?.permissions().mode() & 0o111 == 0 {
                    return Err("workload artifact is not executable".into());
                }
            }
            if package {
                let mut reader = File::open(&path)?;
                let length = reader.metadata()?.len();
                executable::inspect(&mut reader, 0, length)?;
            }
        }
    }
    let native_stamp = contained(&root, &producer.files["native_stamp"].path)?;
    if fs::metadata(&native_stamp)?.len() > 64 * 1024 {
        return Err("native build stamp exceeds metadata bound".into());
    }
    stamp(&fs::read(&native_stamp)?, native_head)?;
    let modified = fs::metadata(native_stamp)?.modified()?;
    for key in ["candidate", "test_binary"] {
        if fs::metadata(contained(&root, &producer.files[key].path)?)?.modified()? <= modified {
            return Err("workload executable predates stamped native ABI".into());
        }
    }
    Ok(())
}

pub(super) fn seal_verified_receipt(
    closure: &Path,
    expected: &Digest,
    native_head: &str,
    candidate: &str,
) -> DynResult<Vec<u8>> {
    source::revision(candidate)?;
    let bytes = fs::read(closure.join("producer.json"))?;
    if Digest::of_bytes(&bytes) != *expected {
        return Err("pre-snapshot producer differs from trusted controller receipt".into());
    }
    let mut producer = producer(&bytes)?;
    files(closure, &producer, native_head)?;
    producer.source = Source {
        head: candidate.to_owned(),
        worktree_sha256: Digest::of_bytes(b""),
    };
    Ok(serde_json::to_vec_pretty(&producer)?)
}

pub(super) fn members(bytes: &[u8]) -> DynResult<Vec<(String, Digest)>> {
    let producer = producer(bytes)?;
    Ok(producer
        .files
        .values()
        .map(|record| (record.path.clone(), record.sha256.clone()))
        .collect())
}

/// Source snapshot matches legacy binary diff plus ordered nonignored new files.
/// capture must be a new, protected file outside the checkout; only its hash is returned.
pub(super) fn source_identity(root: &Path, capture: &Path) -> DynResult<Source> {
    let root = root.canonicalize()?;
    if capture.exists()
        || !capture.is_absolute()
        || capture
            .parent()
            .ok_or("source capture has no parent")?
            .canonicalize()?
            .starts_with(&root)
    {
        return Err("source diff capture must be new and outside checkout".into());
    }
    process::git(
        &root,
        &["diff".into(), "--binary".into(), "HEAD".into(), "--".into()],
        Some(capture),
    )?;
    let mut digest = Sha256::new();
    let mut file = File::open(capture)?;
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let names = process::git(
        &root,
        &[
            "ls-files".into(),
            "--others".into(),
            "--exclude-standard".into(),
            "-z".into(),
        ],
        None,
    )?;
    let mut names = names
        .split(|byte| *byte == 0)
        .filter(|name| !name.is_empty())
        .collect::<Vec<_>>();
    names.sort();
    for name in names {
        let relative = std::str::from_utf8(name)?;
        let path = root.join(relative);
        if !fs::symlink_metadata(&path)?.is_file() || !path.canonicalize()?.starts_with(&root) {
            return Err("new source member is not a contained regular file".into());
        }
        digest.update(name);
        digest.update(b"\0");
        digest.update(Digest::of_file(&path)?.as_str().as_bytes());
    }
    Ok(Source {
        head: process::text(&root, &["rev-parse", "HEAD"])?,
        worktree_sha256: Digest::try_from(hex::encode(digest.finalize()))?,
    })
}

/// Rebind only a previously admitted dirty-tree closure to its identical snapshot.
/// This returns bytes for packaging; it never changes the producer manifest.
pub(super) fn seal(
    root: &Path,
    closure: &Path,
    verified: &VerifiedProducer,
    candidate: &str,
) -> DynResult<Vec<u8>> {
    source::revision(candidate)?;
    let bytes = fs::read(closure.join("producer.json"))?;
    if Digest::of_bytes(&bytes) != verified.digest || closure.canonicalize()? != verified.closure {
        return Err("workload producer changed after pre-snapshot admission".into());
    }
    let mut producer = producer(&bytes)?;
    process::text(root, &["diff", "--cached", "--exit-code", candidate, "--"])?;
    process::text(root, &["diff", "--exit-code", candidate, "--"])?;
    if !process::text(root, &["ls-files", "--others", "--exclude-standard"])?.is_empty() {
        return Err("untracked source after workload snapshot".into());
    }
    files(closure, &producer, &verified.native_head)?;
    producer.source = Source {
        head: candidate.to_owned(),
        worktree_sha256: Digest::of_bytes(b""),
    };
    Ok(serde_json::to_vec_pretty(&producer)?)
}

pub(super) fn verify_producer(
    root: &Path,
    closure: &Path,
    capture: &Path,
    native_head: &str,
) -> DynResult<VerifiedProducer> {
    let bytes = fs::read(closure.join("producer.json"))?;
    let producer = producer(&bytes)?;
    if producer.source != source_identity(root, capture)? {
        return Err("workload producer source changed during build".into());
    }
    files(closure, &producer, native_head)?;
    Ok(VerifiedProducer {
        digest: Digest::of_bytes(&bytes),
        closure: closure.canonicalize()?,
        native_head: native_head.to_owned(),
    })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{
        fs::FileTimes,
        os::unix::fs::PermissionsExt,
        time::{Duration, UNIX_EPOCH},
    };

    #[test]
    fn actual_producer_files_require_strict_mtime_digest_and_executable_identity() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("closure");
        fs::create_dir(&root).unwrap();
        let native = "b".repeat(40);
        let mut bytes = super::super::tests::workload_tar(&"a".repeat(40), &native, 200, |_| {});
        bytes.extend_from_slice(&[0; 1024]);
        let archive_path = directory.path().join("fixture.tar");
        fs::write(&archive_path, bytes).unwrap();
        let mut file = File::open(archive_path).unwrap();
        let members = archive::scan(&mut file).unwrap();
        for (name, member) in &members {
            let path = root.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(
                &path,
                archive::bytes(&mut file, member, 2 * 1024 * 1024).unwrap(),
            )
            .unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
            let seconds = if name == "native/.mesh-llm-build-stamp" {
                100
            } else {
                200
            };
            File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(seconds)))
                .unwrap();
        }
        let producer = producer(&fs::read(root.join("producer.json")).unwrap()).unwrap();
        assert!(files(&root, &producer, &native).is_ok());
        let candidate = root.join("cargo/debug/skippy");
        File::options()
            .write(true)
            .open(&candidate)
            .unwrap()
            .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(100)))
            .unwrap();
        assert!(files(&root, &producer, &native).is_err());
        File::options()
            .write(true)
            .open(&candidate)
            .unwrap()
            .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(200)))
            .unwrap();
        fs::set_permissions(&candidate, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(files(&root, &producer, &native).is_err());
        fs::set_permissions(&candidate, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(candidate, b"replaced executable").unwrap();
        assert!(files(&root, &producer, &native).is_err());
    }
}
