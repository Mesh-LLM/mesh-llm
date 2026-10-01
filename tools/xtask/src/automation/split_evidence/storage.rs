use super::{
    Error,
    args::NAMES,
    boundary, encoding,
    types::{Snapshot, Text},
};
use crate::automation::codepoint_json::strings::JsonString;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub(super) fn load(paths: &[PathBuf; 6]) -> Result<[Snapshot; 6], Error> {
    let mut snapshots = Vec::new();
    for (path, label) in paths.iter().zip(NAMES) {
        let raw = fs::read(path).map_err(|source| Error::Read {
            label,
            path: path.clone(),
            source,
        })?;
        let sha256 = hex::encode(Sha256::digest(&raw));
        let payload = encoding::snapshot(&raw).map_err(|reason| Error::Json {
            label,
            path: path.clone(),
            reason,
        })?;
        boundary::object(&payload, label)?;
        let basename = path.file_name().unwrap_or_default().to_string_lossy();
        snapshots.push(Snapshot {
            payload,
            sha256,
            basename: Text(JsonString::from(basename.as_ref())),
        });
    }
    snapshots
        .try_into()
        .map_err(|_| Error::Contract("six snapshots required".into()))
}

struct Temporary(PathBuf);

impl Drop for Temporary {
    fn drop(&mut self) {
        match fs::remove_file(&self.0) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing_cleanup(&self.0, &error),
        }
    }
}

fn tracing_cleanup(path: &Path, error: &std::io::Error) {
    if let Err(output_error) = writeln!(
        crate::cli_output::stderr(),
        "cannot remove split evidence temporary {}: {error}",
        path.display()
    ) {
        std::mem::drop(output_error);
    }
}

pub(super) fn write(path: &Path, text: &str) -> Result<(), Error> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let basename = path.file_name().unwrap_or_default().to_string_lossy();
    let mut random = [0_u8; 16];
    getrandom::fill(&mut random).map_err(|error| {
        Error::Contract(format!("cannot allocate split evidence temporary: {error}"))
    })?;
    let temporary = parent.join(format!(".{basename}.{}.tmp", hex::encode(random)));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let guard = Temporary(temporary);
    file.write_all(text.as_bytes())?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    fs::rename(&guard.0, path)?;
    Ok(())
}
