//! Local source-bound producer admission; package-only Mach-O gates stay separate.
use super::super::{process, producer_receipt, source};
use super::{
    FIXED, Producer, Record, Source, contained, producer, source_identity, stamp, verify_files,
};
use crate::{automation::canary_receipts::Digest, command::DynResult};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    process::operation(|| {
        match args {
        [mode, root, output] if mode == "snapshot" => snapshot(Path::new(root), Path::new(output)),
        [mode, root, closure, test, snapshot] if mode == "produce" => produce(Path::new(root), Path::new(closure), Path::new(test), Path::new(snapshot)),
        [mode, root, binary, native] if mode == "fresh" => fresh(Path::new(root), Path::new(binary), Path::new(native)),
        [mode, root, binary, native, manifest] if mode == "verify" => verified_test(Path::new(root), Path::new(binary), Path::new(native), Path::new(manifest)).map(|_| ()),
        _ => Err("usage: workload-manifest {snapshot ROOT OUTPUT | produce ROOT CLOSURE TEST_BINARY SNAPSHOT | fresh ROOT BINARY NATIVE_DIR | verify ROOT BINARY NATIVE_DIR MANIFEST}".into()),
    }
    })
}

struct Capture(PathBuf);
impl Capture {
    fn new() -> DynResult<Self> {
        let mut random = [0; 16];
        getrandom::fill(&mut random)
            .map_err(|error| format!("source capture randomness unavailable: {error}"))?;
        let directory =
            std::env::temp_dir().join(format!("mesh-workload-source-{}", hex::encode(random)));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&directory)?;
        Ok(Self(directory.canonicalize()?))
    }
    fn identity(&self, root: &Path) -> DynResult<Source> {
        source_identity(root, &self.0.join("diff"))
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.0.join("diff"));
        let _ = fs::remove_dir(&self.0);
    }
}
fn identity(root: &Path) -> DynResult<Source> {
    if !root.is_absolute() {
        return Err("workload source root must be absolute".into());
    }
    Capture::new()?.identity(root)
}
fn document(path: &Path) -> DynResult<Vec<u8>> {
    if !fs::symlink_metadata(path)?.is_file() || fs::metadata(path)?.len() > 2 * 1024 * 1024 {
        return Err("workload metadata must be a bounded regular file".into());
    }
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(2 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 2 * 1024 * 1024 {
        return Err("workload metadata exceeds bound".into());
    }
    Ok(bytes)
}
fn publish(path: &Path, bytes: &[u8]) -> DynResult<()> {
    if !path.is_absolute() {
        return Err("workload output must be absolute".into());
    }
    if fs::symlink_metadata(path).is_ok_and(|metadata| !metadata.is_file()) {
        return Err("workload output is not a regular file".into());
    }
    let mut random = [0; 16];
    getrandom::fill(&mut random)
        .map_err(|error| format!("workload output randomness unavailable: {error}"))?;
    let parent = path
        .parent()
        .ok_or("workload output has no parent")?
        .canonicalize()?;
    let pending = parent.join(format!(".mesh-workload-{}", hex::encode(random)));
    producer_receipt::create_file(&pending, bytes)?;
    let result = process::check().and_then(|()| Ok(fs::rename(&pending, path)?));
    if result.is_err() {
        let _ = fs::remove_file(pending);
    }
    result
}
fn snapshot(root: &Path, output: &Path) -> DynResult<()> {
    let source = identity(root)?;
    source::revision(&source.head)?;
    publish(output, &serde_json::to_vec_pretty(&source)?)
}
fn freshness(binary: &Path, native: &Path, native_head: &str) -> DynResult<()> {
    if !binary.is_absolute() || !native.is_absolute() || !fs::symlink_metadata(binary)?.is_file() {
        return Err(
            "workload executable and native directory must be absolute regular inputs".into(),
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(binary)?.permissions().mode() & 0o111 == 0 {
            return Err("workload candidate is not executable".into());
        }
    }
    let stamp_path = native.join(".mesh-llm-build-stamp");
    if fs::metadata(&stamp_path)?.len() > 64 * 1024 {
        return Err("native build stamp exceeds metadata bound".into());
    }
    let bytes = document(&stamp_path)?;
    stamp(&bytes, native_head)?;
    if fs::metadata(binary)?.modified()? <= fs::metadata(stamp_path)?.modified()? {
        return Err("workload executable predates stamped native ABI".into());
    }
    Ok(())
}
fn fresh(root: &Path, binary: &Path, native: &Path) -> DynResult<()> {
    if !root.is_absolute() {
        return Err("workload root must be absolute".into());
    }
    let prepared = source::prepared(root)?;
    freshness(binary, native, &prepared.head)
}
fn records(closure: &Path, test: &Path) -> DynResult<BTreeMap<String, Record>> {
    let closure = closure.canonicalize()?;
    let test = test.canonicalize()?;
    let relative = test
        .strip_prefix(&closure)?
        .to_str()
        .ok_or("non-UTF8 workload test path")?;
    let mut records = BTreeMap::new();
    for (key, path) in FIXED.into_iter().chain([("test_binary", relative)]) {
        process::check()?;
        records.insert(
            key.to_owned(),
            Record {
                path: path.to_owned(),
                sha256: Digest::of_file(&contained(&closure, path)?)?,
            },
        );
    }
    Ok(records)
}
fn produce(root: &Path, closure: &Path, test: &Path, snapshot: &Path) -> DynResult<()> {
    if !closure.is_absolute() || !test.is_absolute() {
        return Err("workload producer paths must be absolute".into());
    }
    let expected: Source = serde_json::from_slice(&document(snapshot)?)?;
    let current = identity(root)?;
    if current != expected {
        return Err("repository source changed while building workload producers".into());
    }
    let prepared = source::prepared(root)?;
    let document = Producer {
        schema_version: 1,
        source: current.clone(),
        files: records(closure, test)?,
    };
    let bytes = serde_json::to_vec_pretty(&document)?;
    let parsed = producer(&bytes)?;
    verify_files(closure, &parsed, &prepared.head, false)?;
    if identity(root)? != current || source::prepared(root)?.head != prepared.head {
        return Err("repository source changed during workload admission".into());
    }
    publish(&closure.join("producer.json"), &bytes)
}
pub(crate) fn verified_test(
    root: &Path,
    binary: &Path,
    native: &Path,
    manifest: &Path,
) -> DynResult<PathBuf> {
    let manifest = manifest.canonicalize()?;
    let closure = manifest
        .parent()
        .ok_or("workload manifest has no parent")?
        .canonicalize()?;
    if binary.canonicalize()? != closure.join("cargo/debug/skippy").canonicalize()?
        || native.canonicalize()? != closure.join("native").canonicalize()?
    {
        return Err("workload caller paths differ from admitted closure".into());
    }
    let bytes = document(&manifest)?;
    let producer = producer(&bytes)?;
    let current = identity(root)?;
    if producer.source != current {
        return Err("workload producer does not match current head and worktree".into());
    }
    let prepared = source::prepared(root)?;
    verify_files(&closure, &producer, &prepared.head, false)?;
    if identity(root)? != current
        || document(&manifest)? != bytes
        || source::prepared(root)?.head != prepared.head
    {
        return Err("workload source or producer changed during verification".into());
    }
    contained(&closure, &producer.files["test_binary"].path)
}
