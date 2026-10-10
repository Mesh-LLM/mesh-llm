use super::super::rewriter_patch::shards::FamilyPatches;
use super::{Error, parent};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
pub(super) fn read_json(path: &Path) -> Result<serde_json::Value, Error> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err(Error::JsonLimit);
    }
    Ok(serde_json::from_slice(&bytes)?)
}
struct Stage(PathBuf);
impl Drop for Stage {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!("native generator staging cleanup failed: {error}");
        }
    }
}
pub(super) fn publish(output: &Path, patches: &FamilyPatches) -> Result<(), Error> {
    let parent = parent(output);
    fs::create_dir_all(parent)?;
    let mut random = [0; 16];
    getrandom::fill(&mut random).map_err(|_| Error::Entropy)?;
    let staging_path = parent.join(format!(".native-generator-{}", hex::encode(random)));
    fs::create_dir(&staging_path)?;
    let stage = Stage(staging_path);
    for shard in &patches.shards {
        fs::write(stage.0.join(&shard.file), &shard.bytes)?;
    }
    fs::write(stage.0.join("series.json"), &patches.series_json)?;
    fs::write(stage.0.join("series"), &patches.series)?;
    match fs::symlink_metadata(output) {
        Ok(_) => fs::remove_dir_all(output)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.into()),
    }
    fs::rename(&stage.0, output)?;
    Ok(())
}
