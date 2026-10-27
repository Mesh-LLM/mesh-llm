//! Retain native KV extent observations before the server removes its runtime directory.
use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, ErrorKind, Seek, SeekFrom, Write},
    path::Path,
};

const MAX_READ_BYTES: u64 = 32 * 1024 * 1024;

pub(super) fn capture(runtime_root: &Path, server_pid: u32, output: &Path) -> io::Result<usize> {
    let source = runtime_root
        .join(server_pid.to_string())
        .join("logs/skippy-native.log");
    let mut native_log = File::open(&source)?;
    let offset = native_log.metadata()?.len().saturating_sub(MAX_READ_BYTES);
    native_log.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::new(native_log);
    if offset > 0 {
        let mut partial = String::new();
        reader.read_line(&mut partial)?;
    }

    let mut retained = BufWriter::new(File::create(output.join("native-kv-span.log"))?);
    let mut count = 0usize;
    for line in reader.lines() {
        let line = line?;
        if line.contains("find_slot: stream[") {
            retained.write_all(line.as_bytes())?;
            retained.write_all(b"\n")?;
            count += 1;
        }
    }
    retained.flush()?;
    if count == 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "native KV span log had no find_slot observations",
        ));
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_only_native_kv_extents_before_runtime_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path().join("runtime/42/logs");
        std::fs::create_dir_all(&runtime).unwrap();
        std::fs::write(
            runtime.join("skippy-native.log"),
            "model metadata\nfind_slot: stream[0], n = 256, used = 128, head = 256, size = 4096, n_swa = 0\nother debug\nfind_slot: stream[0], n = 512, used = 256, head = 512, size = 4096, n_swa = 0\n",
        )
        .unwrap();
        assert_eq!(
            capture(&dir.path().join("runtime"), 42, dir.path()).unwrap(),
            2
        );
        let saved = std::fs::read_to_string(dir.path().join("native-kv-span.log")).unwrap();
        assert_eq!(saved.lines().count(), 2);
        assert!(!saved.contains("model metadata"));
    }
}
