//! bounded native log evidence captured before KV requests.
use crate::process::Cancellation;
use serde::Serialize;
use std::{
    fs::File,
    io::{self, BufRead, BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    time::Instant,
};
const TAIL: usize = 4096;
const RETAINED_FINDINGS: usize = 64;
const PATTERNS: [&str; 5] = [
    "failed to find a memory slot",
    "RuntimeError: llama_decode failed",
    "llama_decode failed",
    "proactive_eviction",
    "status=error",
];

#[derive(Debug)]
pub(super) struct Checkpoint {
    path: PathBuf,
    offset: u64,
    identity: Option<(u64, u64)>,
    tail: Vec<u8>,
    next_line: u64,
}
#[derive(Debug, Serialize)]
pub(super) struct Finding {
    pub path: PathBuf,
    pub line_number: u64,
    pub pattern: &'static str,
    pub text: String,
}
#[derive(Debug, Serialize)]
pub(super) struct Scan {
    pub path: PathBuf,
    pub start_offset: u64,
    pub rescanned: bool,
    pub total_findings: u64,
    pub findings: Vec<Finding>,
}

impl Checkpoint {
    pub fn capture(path: &Path, budget: &Budget) -> io::Result<Self> {
        budget.check()?;
        let mut file = match open_regular(path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(Self {
                    path: path.to_owned(),
                    offset: 0,
                    identity: None,
                    tail: vec![],
                    next_line: 1,
                });
            }
            Err(error) => return Err(error),
        };
        let identity = Some(identity(&file)?);
        let offset = file.metadata()?.len();
        let next_line = count_lines(&mut file, offset, budget)?;
        let tail = tail(&mut file, offset)?;
        Ok(Self {
            path: path.to_owned(),
            offset,
            identity,
            tail,
            next_line,
        })
    }
    pub fn scan(&self, budget: &Budget) -> io::Result<Scan> {
        budget.check()?;
        let mut file = open_regular(&self.path)?;
        let metadata = file.metadata()?;
        let unchanged = self.identity == Some(identity(&file)?)
            && metadata.len() >= self.offset
            && tail(&mut file, self.offset)? == self.tail;
        let offset = if unchanged { self.offset } else { 0 };
        file.seek(SeekFrom::Start(offset))?;
        scan_lines(
            &self.path,
            file,
            offset,
            !unchanged,
            if unchanged { self.next_line } else { 1 },
            budget,
        )
    }
}

fn open_regular(path: &Path) -> io::Result<File> {
    if !std::fs::metadata(path)?.is_file() {
        return Err(io::Error::other("native log must be a regular file"));
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(io::Error::other("native log must be a regular file"));
    }
    Ok(file)
}
#[cfg(unix)]
fn identity(file: &File) -> io::Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file.metadata()?;
    Ok((metadata.dev(), metadata.ino()))
}
#[cfg(windows)]
fn identity(file: &File) -> io::Result<(u64, u64)> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
    };
    let mut output = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::uninit();
    // SAFETY: the live borrowed file handle and output storage remain valid for this call.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle().cast(), output.as_mut_ptr()) } == 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a successful call initialized the complete structure.
    let output = unsafe { output.assume_init() };
    Ok((
        u64::from(output.dwVolumeSerialNumber),
        (u64::from(output.nFileIndexHigh) << 32) | u64::from(output.nFileIndexLow),
    ))
}
pub(super) struct Budget {
    pub deadline: Instant,
    pub cancellation: Cancellation,
}
impl Budget {
    fn check(&self) -> io::Result<()> {
        if self.cancellation.is_cancelled() {
            Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "native log evidence cancelled",
            ))
        } else if Instant::now() >= self.deadline {
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "native log evidence deadline expired",
            ))
        } else {
            Ok(())
        }
    }
}

fn tail(file: &mut File, offset: u64) -> io::Result<Vec<u8>> {
    let count = offset.min(TAIL as u64) as usize;
    file.seek(SeekFrom::Start(offset - count as u64))?;
    let mut bytes = vec![0; count];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn count_lines(file: &mut File, offset: u64, budget: &Budget) -> io::Result<u64> {
    file.seek(SeekFrom::Start(0))?;
    let mut remaining = offset;
    let mut next = 1u64;
    let mut bytes = [0u8; 65536];
    while remaining > 0 {
        budget.check()?;
        let limit = remaining.min(bytes.len() as u64) as usize;
        let count = file.read(&mut bytes[..limit])?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "native log changed during initial checkpoint",
            ));
        }
        next = next
            .saturating_add(bytes[..count].iter().filter(|byte| **byte == b'\n').count() as u64);
        remaining -= count as u64;
    }
    Ok(next)
}

#[derive(Default)]
struct Line {
    prefix: Vec<u8>,
    window: Vec<u8>,
    matched: [bool; 5],
    has_bytes: bool,
}
impl Line {
    fn observe(&mut self, bytes: &[u8]) {
        self.has_bytes |= !bytes.is_empty();
        self.prefix
            .extend_from_slice(&bytes[..bytes.len().min(500 - self.prefix.len())]);
        self.window.extend_from_slice(bytes);
        for (index, pattern) in PATTERNS.iter().enumerate() {
            self.matched[index] |= self
                .window
                .windows(pattern.len())
                .any(|window| window == pattern.as_bytes());
        }
        if self.window.len() > 64 {
            let start = self.window.len() - 64;
            self.window.copy_within(start.., 0);
            self.window.truncate(64);
        }
    }
    fn finding(&self, path: &Path, line_number: u64) -> Option<Finding> {
        let pattern = self.matched[..3]
            .iter()
            .position(|matched| *matched)
            .map(|index| PATTERNS[index])
            .or_else(|| {
                (self.matched[3] && self.matched[4]).then_some("proactive_eviction status=error")
            })?;
        Some(Finding {
            path: path.to_owned(),
            line_number,
            pattern,
            text: String::from_utf8_lossy(&self.prefix).trim().to_owned(),
        })
    }
}
fn record(scan: &mut Scan, line: &Line, line_number: u64) {
    if let Some(finding) = line.finding(&scan.path, line_number) {
        scan.total_findings = scan.total_findings.saturating_add(1);
        if scan.findings.len() < RETAINED_FINDINGS {
            scan.findings.push(finding);
        }
    }
}
fn scan_lines(
    path: &Path,
    file: File,
    offset: u64,
    rescanned: bool,
    mut line_number: u64,
    budget: &Budget,
) -> io::Result<Scan> {
    let mut reader = BufReader::new(file);
    let mut line = Line::default();
    let mut scan = Scan {
        path: path.to_owned(),
        start_offset: offset,
        rescanned,
        total_findings: 0,
        findings: vec![],
    };
    loop {
        budget.check()?;
        let bytes = reader.fill_buf()?;
        if bytes.is_empty() {
            break;
        }
        let newline = bytes.iter().position(|byte| *byte == b'\n');
        let count = newline.unwrap_or(bytes.len());
        line.observe(&bytes[..count]);
        reader.consume(count + usize::from(newline.is_some()));
        if newline.is_some() {
            record(&mut scan, &line, line_number);
            line = Line::default();
            line_number = line_number.saturating_add(1);
        }
    }
    if line.has_bytes {
        record(&mut scan, &line, line_number);
    }
    Ok(scan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, io::Write, time::Duration};
    fn available() -> Budget {
        Budget {
            deadline: Instant::now() + Duration::from_secs(5),
            cancellation: Cancellation::default(),
        }
    }
    #[test]
    fn kv_log_checkpoint_ignores_old_failures_and_retains_appended_file_line_numbers() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("native.log");
        fs::write(
            &path,
            "RuntimeError: llama_decode failed\nold benign line\n",
        )
        .unwrap();
        let budget = available();
        let checkpoint = Checkpoint::capture(&path, &budget).unwrap();
        let mut writer = fs::OpenOptions::new().append(true).open(&path).unwrap();
        writer
            .write_all(b"new benign line\nproactive_eviction phase=cleanup status=error\n")
            .unwrap();
        drop(writer);
        let scan = checkpoint.scan(&budget).unwrap();
        assert!(!scan.rescanned);
        assert_eq!(scan.total_findings, 1);
        assert_eq!(scan.findings[0].line_number, 4);
        assert_eq!(scan.findings[0].pattern, "proactive_eviction status=error");
        assert_eq!(scan.start_offset, checkpoint.offset);
    }
    #[test]
    fn kv_log_checkpoint_reports_new_missing_rotated_truncated_and_rewritten_logs() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("native.log");
        let budget = available();
        let missing = Checkpoint::capture(&path, &budget).unwrap();
        assert_eq!(
            missing.scan(&budget).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        fs::write(&path, "llama_decode failed\n").unwrap();
        let created = missing.scan(&budget).unwrap();
        assert!(created.rescanned);
        assert_eq!(created.findings[0].line_number, 1);
        for rotation in [false, true] {
            fs::write(
                &path,
                "old benign line repeated to make initial file longer than failure\n",
            )
            .unwrap();
            let checkpoint = Checkpoint::capture(&path, &budget).unwrap();
            if rotation {
                fs::rename(&path, directory.path().join("rotated.log")).unwrap();
            }
            fs::write(&path, "failed to find a memory slot\n").unwrap();
            let scan = checkpoint.scan(&budget).unwrap();
            assert!(scan.rescanned);
            assert_eq!(scan.start_offset, 0);
            assert_eq!(scan.findings[0].line_number, 1);
        }
        fs::write(&path, "old benign line\n").unwrap();
        let checkpoint = Checkpoint::capture(&path, &budget).unwrap();
        fs::write(
            &path,
            "new replacement log with llama_decode failed and a longer payload\n",
        )
        .unwrap();
        assert!(checkpoint.scan(&budget).unwrap().rescanned);
    }
    #[test]
    fn kv_log_long_lines_and_many_findings_remain_bounded_without_hiding_failures() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("native.log");
        fs::write(&path, "").unwrap();
        let budget = available();
        let checkpoint = Checkpoint::capture(&path, &budget).unwrap();
        let mut writer = fs::OpenOptions::new().append(true).open(&path).unwrap();
        writer.write_all(b"proactive_eviction ").unwrap();
        writer.write_all(&vec![b'x'; 200_000]).unwrap();
        writer.write_all(b" status=error\n").unwrap();
        for _ in 0..1000 {
            writer
                .write_all(b"RuntimeError: llama_decode failed\n")
                .unwrap();
        }
        drop(writer);
        let scan = checkpoint.scan(&budget).unwrap();
        assert_eq!(scan.total_findings, 1001);
        assert_eq!(scan.findings.len(), RETAINED_FINDINGS);
        assert_eq!(scan.findings[0].pattern, "proactive_eviction status=error");
        assert!(scan.findings[0].text.len() <= 500);
        assert_eq!(
            scan.findings[1].pattern,
            "RuntimeError: llama_decode failed"
        );
    }
    #[test]
    fn kv_log_evidence_rejects_nonregular_paths_expired_budgets_and_cancellation() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(
            Checkpoint::capture(directory.path(), &available())
                .unwrap_err()
                .kind(),
            io::ErrorKind::Other
        );
        let path = directory.path().join("native.log");
        fs::write(&path, "benign\n").unwrap();
        let expired = Budget {
            deadline: Instant::now(),
            cancellation: Cancellation::default(),
        };
        assert_eq!(
            Checkpoint::capture(&path, &expired).unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
        let budget = available();
        let checkpoint = Checkpoint::capture(&path, &budget).unwrap();
        budget.cancellation.cancel();
        assert_eq!(
            Checkpoint::capture(&path, &budget).unwrap_err().kind(),
            io::ErrorKind::Interrupted
        );
        assert_eq!(
            checkpoint.scan(&budget).unwrap_err().kind(),
            io::ErrorKind::Interrupted
        );
        assert!(path.is_file());
    }
}
