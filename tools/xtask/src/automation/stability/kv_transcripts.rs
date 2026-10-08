//! fresh, manifest-bound transcripts without deleting foreign evidence.
use super::{evidence, kv_conversation::Record};
use crate::command::DynResult;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Serialize)]
pub(super) struct Entry {
    pub model: String,
    pub attempt: u32,
    pub cohort: String,
    pub path: PathBuf,
}
pub(super) struct Transcripts {
    directory: PathBuf,
    relative: PathBuf,
    pub entries: Vec<Entry>,
}
impl Transcripts {
    pub fn create(output: &Path) -> DynResult<Self> {
        evidence::directory(output)?;
        let parent = output.join("transcripts");
        match fs::symlink_metadata(&parent) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => (),
            Ok(_) => return Err("KV transcript destination must be a real directory".into()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => fs::create_dir(&parent)?,
            Err(error) => return Err(error.into()),
        }
        // A visible, exclusive run directory makes uploaded paths fresh and unambiguous.
        // Keep partial files on failure; this process never recursively removes old evidence.
        let directory = tempfile::Builder::new()
            .prefix("run-")
            .tempdir_in(parent)?
            .keep();
        let name = directory
            .file_name()
            .ok_or("KV transcript run has no directory name")?;
        let relative = PathBuf::from("transcripts").join(name);
        Ok(Self {
            directory,
            relative,
            entries: vec![],
        })
    }
    pub fn write(
        &mut self,
        model: &str,
        attempt: u32,
        cohort: &str,
        records: &[Record],
    ) -> DynResult<()> {
        if self.entries.len() >= 100000 {
            return Err("KV transcript cohort exceeds bounded size".into());
        }
        let filename = format!("{:06}.jsonl", self.entries.len() + 1);
        let path = self.directory.join(&filename);
        // Model names and cohort labels are metadata, never filesystem selectors.
        evidence::jsonl(&path, records)?;
        self.entries.push(Entry {
            model: model.into(),
            attempt,
            cohort: cohort.into(),
            path: self.relative.join(filename),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kv_transcripts_select_a_fresh_run_and_preserve_preexisting_evidence() {
        let output = tempfile::tempdir().unwrap();
        let old = output.path().join("transcripts");
        fs::create_dir(&old).unwrap();
        fs::write(old.join("foreign.jsonl"), b"old evidence").unwrap();
        let mut run = Transcripts::create(output.path()).unwrap();
        let records = [Record {
            phase: "first_tool_call".into(),
            status_code: Some(200),
            call_id: Some("call-1".into()),
            error: None,
        }];
        run.write("../model/alias", 1, "../../cohort", &records)
            .unwrap();
        run.write("different/model", 1, "tool_loop", &records)
            .unwrap();
        assert_eq!(
            fs::read(old.join("foreign.jsonl")).unwrap(),
            b"old evidence"
        );
        assert_ne!(run.entries[0].path, run.entries[1].path);
        for entry in &run.entries {
            assert!(entry.path.starts_with(&run.relative));
            assert!(output.path().join(&entry.path).is_file());
        }
        let second = Transcripts::create(output.path()).unwrap();
        assert_ne!(run.relative, second.relative);
        assert!(second.entries.is_empty());
    }
    #[test]
    fn kv_transcript_leaf_file_is_refused_without_truncation() {
        let output = tempfile::tempdir().unwrap();
        fs::write(output.path().join("transcripts"), b"foreign").unwrap();
        assert!(Transcripts::create(output.path()).is_err());
        assert_eq!(
            fs::read(output.path().join("transcripts")).unwrap(),
            b"foreign"
        );
    }
    #[cfg(unix)]
    #[test]
    fn kv_transcript_leaf_symlink_is_refused_without_foreign_writes() {
        let output = tempfile::tempdir().unwrap();
        let foreign = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(foreign.path(), output.path().join("transcripts")).unwrap();
        assert!(Transcripts::create(output.path()).is_err());
        assert_eq!(fs::read_dir(foreign.path()).unwrap().count(), 0);
    }
}
