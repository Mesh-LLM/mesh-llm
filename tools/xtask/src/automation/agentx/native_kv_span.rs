//! Retain only numeric native KV observations from the standalone Skippy CLI.
use std::{
    fs::File,
    io::{self, BufWriter, ErrorKind, Write},
    path::Path,
};

use crate::process::{ObservedLine, Stream, retained::MemberId};

const MAX_OBSERVATIONS: usize = 100_000;

#[derive(Clone, Copy)]
struct Observation {
    stream: u32,
    n: u32,
    used: u32,
    head: u32,
    size: u32,
    n_swa: u32,
}

#[derive(Default)]
pub(super) struct NativeKvSpan {
    observations: Vec<Observation>,
    overflowed: bool,
}

impl NativeKvSpan {
    pub(super) fn observe(&mut self, member: MemberId, line: &ObservedLine<'_>) {
        if member != MemberId::Seed {
            return;
        }
        let observation = match line.stream {
            Stream::Stdout => parse_jsonl_observation(line.bytes),
            Stream::Stderr => parse_observation(line.bytes),
        };
        let Some(observation) = observation else {
            return;
        };
        if self.observations.len() == MAX_OBSERVATIONS {
            self.overflowed = true;
        } else {
            self.observations.push(observation);
        }
    }

    pub(super) fn save(&self, output: &Path) -> io::Result<usize> {
        if self.observations.is_empty() || self.overflowed {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "native KV span observations absent or exceeded the bounded capture",
            ));
        }
        let mut retained = BufWriter::new(File::create(output.join("native-kv-span.log"))?);
        for observation in &self.observations {
            writeln!(
                retained,
                "find_slot: stream[{}], n = {}, used = {}, head = {}, size = {}, n_swa = {}",
                observation.stream,
                observation.n,
                observation.used,
                observation.head,
                observation.size,
                observation.n_swa,
            )?;
        }
        retained.flush()?;
        Ok(self.observations.len())
    }
}

fn parse_jsonl_observation(bytes: &[u8]) -> Option<Observation> {
    if !std::str::from_utf8(bytes)
        .ok()?
        .contains("find_slot: stream[")
    {
        return None;
    }
    let event: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    if event.get("type")?.as_str()? != "native_log" {
        return None;
    }
    parse_observation(event.pointer("/data/message")?.as_str()?.as_bytes())
}

fn parse_observation(bytes: &[u8]) -> Option<Observation> {
    let line = std::str::from_utf8(bytes).ok()?;
    let (_, values) = line.split_once("find_slot: stream[")?;
    let (stream, values) = values.split_once("], n =")?;
    let (n, values) = values.split_once(", used =")?;
    let (used, values) = values.split_once(", head =")?;
    let (head, values) = values.split_once(", size =")?;
    let (size, values) = values.split_once(", n_swa =")?;
    Some(Observation {
        stream: stream.trim().parse().ok()?,
        n: n.trim().parse().ok()?,
        used: used.trim().parse().ok()?,
        head: head.trim().parse().ok()?,
        size: size.trim().parse().ok()?,
        n_swa: values.trim().parse().ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::LineEnding;

    #[test]
    fn saves_only_numeric_standalone_native_observations() {
        let dir = tempfile::tempdir().unwrap();
        let mut span = NativeKvSpan::default();
        let native =
            b"find_slot: stream[0], n = 256, used = 128, head = 256, size = 4096, n_swa = 0";
        span.observe(
            MemberId::Seed,
            &ObservedLine {
                stream: Stream::Stderr,
                bytes: native,
                ending: LineEnding::Lf,
            },
        );
        span.observe(
            MemberId::WorkerOne,
            &ObservedLine {
                stream: Stream::Stderr,
                bytes: native,
                ending: LineEnding::Lf,
            },
        );
        span.observe(
            MemberId::Seed,
            &ObservedLine {
                stream: Stream::Stderr,
                bytes: b"find_slot: stream[secret], n = prompt",
                ending: LineEnding::Lf,
            },
        );
        assert_eq!(span.save(dir.path()).unwrap(), 1);
        let saved = std::fs::read_to_string(dir.path().join("native-kv-span.log")).unwrap();
        assert_eq!(saved.trim(), std::str::from_utf8(native).unwrap());
    }

    #[test]
    fn captures_native_kv_spans_from_standalone_jsonl_stdout() {
        let dir = tempfile::tempdir().unwrap();
        let mut span = NativeKvSpan::default();
        let native = br#"{"data":{"message":"find_slot: stream[0], n = 256, used = 128, head = 256, size = 4096, n_swa = 0\n"},"schema_version":1,"sequence":17,"type":"native_log"}"#;
        for (member, bytes) in [
            (MemberId::Seed, native.as_slice()),
            (MemberId::WorkerOne, native.as_slice()),
            (MemberId::Seed, br#"{"data":{"message":"find_slot: stream[1], n = 1, used = 1, head = 1, size = 1, n_swa = 0"},"type":"status"}"#.as_slice()),
        ] {
            span.observe(
                member,
                &ObservedLine {
                    stream: Stream::Stdout,
                    bytes,
                    ending: LineEnding::Lf,
                },
            );
        }
        assert_eq!(span.save(dir.path()).unwrap(), 1);
        let saved = std::fs::read_to_string(dir.path().join("native-kv-span.log")).unwrap();
        assert_eq!(
            saved.trim(),
            "find_slot: stream[0], n = 256, used = 128, head = 256, size = 4096, n_swa = 0"
        );
    }
}
