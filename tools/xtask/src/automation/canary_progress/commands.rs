//! Bounded snapshots of run-owned command logs, including output hidden by tool pipes.
use super::compact;
use std::{
    collections::BTreeSet,
    fs::File,
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
            .retain(|path| logs.iter().any(|(_, _, log, _, _)| log == path));
        for (_, _, log, mut file, completion) in logs {
            if self.finished.contains(&log) {
                continue;
            }
            let metadata = file.metadata()?;
            let mut header = Vec::new();
            Read::by_ref(&mut file).take(256).read_to_end(&mut header)?;
            let elapsed = elapsed(&header);
            file.seek(SeekFrom::End(-(metadata.len().min(4096) as i64)))?;
            let mut bytes = Vec::new();
            file.take(4096).read_to_end(&mut bytes)?;
            let tail = String::from_utf8_lossy(&bytes);
            let idle = metadata.modified()?.elapsed().unwrap_or_default().as_secs();
            let status = completion.map(|code| format!("exit_status: {code}"));
            let label = log
                .parent()
                .and_then(Path::file_name)
                .unwrap_or_default()
                .to_string_lossy();
            writeln!(
                output,
                "canary command: {} — {}; {elapsed}; last output {idle}s ago; log: {}",
                compact(&label, 100),
                status.as_deref().unwrap_or("running"),
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

// Hold every opened directory and log by descriptor so replacements cannot
// redirect a later read outside the run-owned tree.
type Log = (bool, SystemTime, PathBuf, File, Option<i32>);

#[cfg(unix)]
fn recent_logs(root: &Path) -> io::Result<Vec<Log>> {
    use rustix::fs::{Dir, Mode, OFlags, open, openat};
    use std::os::unix::ffi::OsStrExt;
    let directory_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let root_fd = match open(root, directory_flags, Mode::empty()) {
        Ok(fd) => fd,
        Err(rustix::io::Errno::NOENT | rustix::io::Errno::LOOP) => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut logs = Vec::new();
    for entry in Dir::read_from(&root_fd)? {
        let entry = entry?;
        let name = std::ffi::OsStr::from_bytes(entry.file_name().to_bytes());
        if name == "." || name == ".." {
            continue;
        }
        let directory = match openat(&root_fd, name, directory_flags, Mode::empty()) {
            Ok(fd) => fd,
            Err(_) => continue,
        };
        let Some(file) = open_log(&directory, "output.log")? else {
            continue;
        };
        let metadata = file.metadata()?;
        let completion = match open_log(&directory, "completion")? {
            Some(file) => {
                let mut text = String::new();
                file.take(32).read_to_string(&mut text)?;
                text.trim()
                    .parse::<i32>()
                    .ok()
                    .filter(|code| (0..=255).contains(code))
            }
            None => None,
        };
        logs.push((
            completion.is_none(),
            metadata.modified()?,
            root.join(name).join("output.log"),
            file,
            completion,
        ));
        // Quiet active commands outrank even the newest completed commands.
        logs.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));
        if logs.len() > 16 {
            logs.remove(0);
        }
    }
    Ok(logs)
}

#[cfg(unix)]
fn open_log(directory: &impl std::os::fd::AsFd, name: &str) -> io::Result<Option<File>> {
    use rustix::fs::{Mode, OFlags, openat};
    let fd = match openat(
        directory,
        name,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(fd) => fd,
        Err(rustix::io::Errno::NOENT | rustix::io::Errno::LOOP | rustix::io::Errno::NOTDIR) => {
            return Ok(None);
        }
        Err(error) => return Err(error.into()),
    };
    let file = File::from(fd);
    Ok(file.metadata()?.is_file().then_some(file))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::fs;
    #[test]
    fn opened_logs_remain_bound_when_file_and_directory_are_replaced() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let command = root.path().join("command");
        fs::create_dir(&command).unwrap();
        fs::write(command.join("output.log"), "owned output").unwrap();
        let mut logs = recent_logs(root.path()).unwrap();
        let (_, _, _, mut file, _) = logs.pop().unwrap();
        let foreign = tempfile::tempdir().unwrap();
        fs::write(foreign.path().join("output.log"), "foreign secret").unwrap();
        fs::rename(&command, root.path().join("old-command")).unwrap();
        symlink(foreign.path(), &command).unwrap();
        let mut text = String::new();
        file.read_to_string(&mut text).unwrap();
        assert_eq!(text, "owned output");
        fs::remove_file(root.path().join("old-command/output.log")).unwrap();
        symlink(
            foreign.path().join("output.log"),
            root.path().join("old-command/output.log"),
        )
        .unwrap();
        assert!(recent_logs(root.path()).unwrap().is_empty());
    }
}
