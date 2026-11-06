//! Bounded snapshots of run-owned command logs, including output hidden by tool pipes.
use super::compact;
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Default)]
pub(super) struct Commands {
    finished: BTreeSet<PathBuf>,
}
impl Commands {
    pub(super) fn report(&mut self, root: &Path, output: &mut impl Write) -> io::Result<()> {
        let logs = recent_logs(root)?;
        self.finished
            .retain(|path| logs.iter().any(|(_, log)| log == path));
        for (_, log) in logs {
            if self.finished.contains(&log) {
                continue;
            }
            let mut file = File::open(&log)?;
            let metadata = file.metadata()?;
            let mut header = Vec::new();
            Read::by_ref(&mut file).take(256).read_to_end(&mut header)?;
            let elapsed = elapsed(&header);
            file.seek(SeekFrom::End(-(metadata.len().min(4096) as i64)))?;
            let mut bytes = Vec::new();
            file.take(4096).read_to_end(&mut bytes)?;
            let tail = String::from_utf8_lossy(&bytes);
            let idle = metadata.modified()?.elapsed().unwrap_or_default().as_secs();
            let status = tail
                .lines()
                .last()
                .filter(|line| line.starts_with("exit_status: "));
            let label = log
                .parent()
                .and_then(Path::file_name)
                .unwrap_or_default()
                .to_string_lossy();
            writeln!(
                output,
                "canary command: {} — {}; {elapsed}; last output {idle}s ago; log: {}",
                compact(&label, 100),
                status.unwrap_or("running"),
                compact(&log.display().to_string(), 600)
            )?;
            for line in tail
                .lines()
                .rev()
                .take(3)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                writeln!(output, "canary output: {}", compact(line, 300))?;
            }
            if status.is_some() {
                self.finished.insert(log);
            }
        }
        Ok(())
    }
}

fn elapsed(header: &[u8]) -> String {
    let header = String::from_utf8_lossy(header);
    let started = header
        .lines()
        .find_map(|line| line.strip_prefix("started_epoch: ")?.parse::<u64>().ok());
    match started
        .and_then(|seconds| UNIX_EPOCH.checked_add(std::time::Duration::from_secs(seconds)))
    {
        Some(started) => format!(
            "{}s elapsed",
            started.elapsed().unwrap_or_default().as_secs()
        ),
        None => "elapsed unknown".into(),
    }
}

fn recent_logs(root: &Path) -> io::Result<Vec<(SystemTime, PathBuf)>> {
    if fs::symlink_metadata(root).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return Ok(Vec::new());
    }
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    // Retain only the most recently written logs rather than pinning the first
    // commands forever. A tool may run several foreground checks in one turn.
    let mut logs = Vec::new();
    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path().join("output.log");
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_file() => metadata,
            _ => continue,
        };
        logs.push((metadata.modified()?, path));
        logs.sort();
        if logs.len() > 16 {
            logs.remove(0);
        }
    }
    Ok(logs)
}
