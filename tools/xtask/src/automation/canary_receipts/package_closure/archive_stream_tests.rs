use super::super::{fixture_scope, process};
use super::data;
use std::io::{self, Read};

struct CancelAfterChunk {
    token: crate::process::Cancellation,
    read: usize,
}
impl Read for CancelAfterChunk {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        bytes.fill(7);
        self.read += bytes.len();
        self.token.cancel();
        Ok(bytes.len())
    }
}
#[test]
fn archive_hash_stops_before_reading_second_chunk_after_cancellation() {
    if fixture_scope::isolated(
        module_path!(),
        "archive_hash_stops_before_reading_second_chunk_after_cancellation",
    ) {
        return;
    }
    let mut consumed = 0;
    let result = process::operation(|| {
        let mut reader = CancelAfterChunk {
            token: process::cancellation(),
            read: 0,
        };
        let result = data(&mut reader, 3 * 65536);
        consumed = reader.read;
        assert!(result.is_err());
        result.map(|_| ())
    });
    assert!(result.is_err());
    assert_eq!(consumed, 65536);
}
