//! Input trees for the packaging-snippet parity cases. Runtime archives are
//! deterministic ustar+gzip streams built in-process (mtime 0), so their
//! digests are stable goldens.

use crate::support::TestResult;
use flate2::Compression;
use flate2::write::GzEncoder;
use std::fs;
use std::io::Write;
use std::path::Path;

/// One parity case: a tree builder and the port's arguments, where
/// `{scratch}` is the scratch directory.
pub(crate) struct Case {
    pub(crate) name: &'static str,
    pub(crate) setup: fn(&Path) -> TestResult,
    pub(crate) args: &'static [&'static str],
}

fn octal(field: &mut [u8], value: u64) {
    let digits = format!("{value:0width$o}", width = field.len() - 1);
    field[..digits.len()].copy_from_slice(digits.as_bytes());
}

fn header(name: &str, size: u64, kind: u8) -> [u8; 512] {
    let mut block = [0_u8; 512];
    block[..name.len()].copy_from_slice(name.as_bytes());
    octal(
        &mut block[100..108],
        if kind == b'5' { 0o755 } else { 0o644 },
    );
    octal(&mut block[108..116], 0);
    octal(&mut block[116..124], 0);
    octal(&mut block[124..136], size);
    octal(&mut block[136..148], 0);
    block[148..156].fill(b' ');
    block[156] = kind;
    block[257..263].copy_from_slice(b"ustar\0");
    block[263..265].copy_from_slice(b"00");
    let sum: u32 = block.iter().map(|byte| u32::from(*byte)).sum();
    let digits = format!("{sum:06o}\0 ");
    block[148..156].copy_from_slice(digits.as_bytes());
    block
}

/// A gzip tar holding `members`; a name ending in `/` is a directory.
pub(crate) fn tar_gz(path: &Path, members: &[(&str, &str)]) -> TestResult {
    let mut raw = Vec::new();
    for (name, body) in members {
        if name.ends_with('/') {
            raw.extend_from_slice(&header(name, 0, b'5'));
            continue;
        }
        raw.extend_from_slice(&header(name, body.len() as u64, b'0'));
        raw.extend_from_slice(body.as_bytes());
        raw.resize(raw.len().div_ceil(512) * 512, 0);
    }
    raw.resize(raw.len() + 1024, 0);
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, encoder.finish()?)?;
    Ok(())
}

pub(crate) fn write(root: &Path, relative: &str, body: &str) -> TestResult {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, body)?;
    Ok(())
}

/// A runtime manifest with every required field.
pub(crate) fn manifest(id: &str, version: &str, abi: &str, platform: &str) -> String {
    format!(
        "{{\"runtime\": {{\"id\": \"{id}\", \"mesh_version\": \"{version}\", \
         \"skippy_abi\": {abi}, \"platform\": {platform}, \
         \"backend\": {{\"kind\": \"cpu\"}}, \"libraries\": [\"libskippy.so\"], \
         \"files\": [\"lib/libskippy.so\"]}}}}"
    )
}

/// `dist/<id>.tar.gz` holding `<id>/manifest.json` and one library.
pub(crate) fn runtime(root: &Path, id: &str, manifest_text: &str, library: &str) -> TestResult {
    let dir = format!("{id}/");
    let manifest_name = format!("{id}/manifest.json");
    let library_name = format!("{id}/lib/libskippy.so");
    tar_gz(
        &root.join(format!("dist/{id}.tar.gz")),
        &[
            (dir.as_str(), ""),
            (manifest_name.as_str(), manifest_text),
            (format!("{id}/lib/").as_str(), ""),
            (library_name.as_str(), library),
        ],
    )
}

pub(crate) const LINUX: &str = "{\"os\": \"linux\", \"arch\": \"x86_64\"}";
pub(crate) const MAC: &str = "{\"os\": \"macos\", \"arch\": \"aarch64\"}";
