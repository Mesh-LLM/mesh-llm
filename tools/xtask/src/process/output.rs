use super::capture::Capture;
use super::{Failure, Readiness, Stream};
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::process::{ChildStderr, ChildStdout};

pub(super) fn output_file(path: Option<&Path>) -> Result<Option<File>, Failure> {
    path.map(|path| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options
            .open(path)
            .map_err(|error| Failure::io("create output", error))
    })
    .transpose()
}

pub(super) struct Output {
    pub(super) stdout: Capture<ChildStdout>,
    pub(super) stderr: Capture<ChildStderr>,
    pub(super) failure: Option<Failure>,
}

impl Output {
    pub(super) fn poll(&mut self, readiness: &Readiness) -> bool {
        self.poll_candidate(readiness).is_some()
    }

    fn poll_candidate(&mut self, readiness: &Readiness) -> Option<Stream> {
        let mut candidate = None;
        for (stream, result) in [
            (Stream::Stdout, self.stdout.poll(readiness)),
            (Stream::Stderr, self.stderr.poll(readiness)),
        ] {
            match result {
                Ok(true) => {
                    candidate.get_or_insert(stream);
                }
                Ok(false) => (),
                Err(error) => {
                    if self.failure.is_none() {
                        self.failure = Some(error);
                    }
                }
            }
        }
        candidate
    }

    pub(super) fn snapshot(&self) -> Result<(usize, usize), Failure> {
        let bytes = (self.stdout.pending_bytes()?, self.stderr.pending_bytes()?);
        if bytes.0 > 16 * 1024 * 1024 || bytes.1 > 16 * 1024 * 1024 {
            return Err(Failure::EnumerationLimit);
        }
        Ok(bytes)
    }

    pub(super) fn buffered_readiness(
        &mut self,
        readiness: &Readiness,
        bytes: (usize, usize),
    ) -> Result<bool, Failure> {
        let stdout = self.stdout.poll_snapshot(readiness, bytes.0)?;
        let stderr = self.stderr.poll_snapshot(readiness, bytes.1)?;
        Ok(stdout || stderr)
    }
}

impl super::observed::Lines for Output {
    fn poll_lines(&mut self, readiness: &Readiness) -> Result<Option<Stream>, Failure> {
        let candidate = self.poll_candidate(readiness);
        match self.failure.take() {
            Some(error) => Err(error),
            None => Ok(candidate),
        }
    }
}

impl super::probe::ProbeLines for Output {
    fn poll_probe(
        &mut self,
        callback: &mut dyn FnMut(super::ObservedLine<'_>),
    ) -> Result<(), Failure> {
        let stdout = self.stdout.poll_probe(callback);
        let stderr = self.stderr.poll_probe(&mut |line| {
            if stdout.is_ok() {
                callback(line);
            }
        });
        stdout.and(stderr)
    }
}
