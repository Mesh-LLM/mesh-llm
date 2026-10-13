use crate::command::DynResult;
use std::{
    io::Write,
    path::{Path, PathBuf},
};

struct Temporary(PathBuf);

impl Drop for Temporary {
    fn drop(&mut self) {
        match std::fs::remove_file(&self.0) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => {
                eprintln!("cannot remove replay snapshot temporary: {error}");
            }
        }
    }
}

pub(super) fn write(path: &Path, document: &serde_json::Value) -> DynResult<()> {
    let parent = path.parent().ok_or("missing replay snapshot directory")?;
    std::fs::create_dir_all(parent)?;
    let mut random = [0_u8; 16];
    getrandom::fill(&mut random).map_err(|_| "replay snapshot entropy unavailable")?;
    let temporary = Temporary(parent.join(format!(".replay-snapshot-{}.tmp", hex::encode(random))));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary.0)?;
    serde_json::to_writer_pretty(&mut file, document)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary.0, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snapshot_replacement_preserves_complete_json_without_temporary_residue() {
        let state = tempfile::tempdir().unwrap();
        let path = state.path().join("run.json");
        std::fs::write(&path, b"prior snapshot").unwrap();
        let document = serde_json::json!({"results":[{"pass":1,"label":"candidate"}]});
        write(&path, &document).unwrap();
        let retained: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(retained, document);
        assert_eq!(std::fs::read_dir(state.path()).unwrap().count(), 1);
    }
}
