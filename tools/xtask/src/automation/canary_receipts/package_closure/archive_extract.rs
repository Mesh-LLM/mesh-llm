use super::{archive, process};
use crate::{automation::canary_receipts::Digest, command::DynResult};
use sha2::{Digest as _, Sha256};
use std::{
    fs::{self, File, FileTimes},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    time::{Duration, SystemTime},
};

pub(super) fn extract(path: &Path, target: &Path) -> DynResult<()> {
    fs::create_dir(target)?;
    let mut file = File::open(path)?;
    let members = archive::scan(&mut file)?;
    for (name, member) in members {
        process::check()?;
        let destination = target.join(name);
        fs::create_dir_all(destination.parent().ok_or("member has no parent")?)?;
        let mut options = fs::OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(if member.executable { 0o755 } else { 0o644 });
        }
        let mut output = options.open(&destination)?;
        file.seek(SeekFrom::Start(member.start))?;
        let mut remaining = member.size;
        let mut hash = Sha256::new();
        let mut buffer = [0; 65536];
        while remaining != 0 {
            process::check()?;
            let count = usize::try_from(remaining.min(u64::try_from(buffer.len())?))?;
            file.read_exact(&mut buffer[..count])?;
            output.write_all(&buffer[..count])?;
            hash.update(&buffer[..count]);
            remaining -= u64::try_from(count)?;
        }
        if Digest::try_from(hex::encode(hash.finalize()))? != member.sha256 {
            return Err("archive member changed during staged restore".into());
        }
        output.sync_all()?;
    }
    Ok(())
}

pub(super) fn normalize_workload(path: &Path) -> DynResult<()> {
    let moment = SystemTime::now();
    let stamp = moment
        .checked_sub(Duration::from_secs(120))
        .ok_or("restore clock is too early")?;
    let mut directories = vec![path.to_owned()];
    while let Some(directory) = directories.pop() {
        for entry in fs::read_dir(directory)? {
            process::check()?;
            let entry = entry?;
            let metadata = entry.file_type()?;
            if metadata.is_dir() {
                directories.push(entry.path());
            } else if metadata.is_file() {
                let modified = if entry.path() == path.join("native/.mesh-llm-build-stamp") {
                    stamp
                } else {
                    moment
                };
                File::options()
                    .write(true)
                    .open(entry.path())?
                    .set_times(FileTimes::new().set_modified(modified))?;
            } else {
                return Err("unexpected staged workload member".into());
            }
        }
    }
    Ok(())
}
