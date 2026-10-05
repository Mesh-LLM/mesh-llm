use super::*;
use serde_json::json;

fn append(events: &mut Events, event: &str, count: u64) {
    retain(
        events,
        &serde_json::to_vec(&json!({"event":event,"attributes":{"count":count}})).unwrap(),
    )
    .unwrap();
}

#[test]
fn seed_cursor_excludes_every_seed_event_stream_from_measured_telemetry() {
    let mut events = Events::default();
    append(&mut events, "stage.openai_generation_summary", 1);
    append(&mut events, "stage.openai_kv_capacity_decision", 2);
    append(&mut events, "stage.openai_kv_record_decision", 3);
    let cursor = events.cursor();
    append(&mut events, "stage.openai_generation_summary", 4);
    append(&mut events, "stage.openai_kv_capacity_decision", 5);
    append(&mut events, "stage.openai_kv_record_decision", 6);
    let measured = events.after(&cursor, 1).unwrap();
    assert_eq!(measured.events[0].attributes["count"], 4);
    assert_eq!(measured.capacity_events[0].attributes["count"], 5);
    assert_eq!(measured.record_events[0].attributes["count"], 6);
}

#[test]
fn invalid_cursors_missing_or_extra_generation_summaries_fail_closed() {
    for (cursor, expected) in [
        (
            Cursor {
                schema_version: 2,
                ..Cursor::default()
            },
            0,
        ),
        (
            Cursor {
                schema_version: 1,
                generations: 1,
                ..Cursor::default()
            },
            0,
        ),
        (
            Cursor {
                schema_version: 1,
                ..Cursor::default()
            },
            1,
        ),
    ] {
        assert!(Events::default().after(&cursor, expected).is_err());
    }
    let mut events = Events::default();
    append(&mut events, "stage.openai_generation_summary", 1);
    assert!(
        events
            .after(
                &Cursor {
                    schema_version: 1,
                    ..Cursor::default()
                },
                0
            )
            .is_err()
    );
}

#[test]
fn ordinary_noise_is_ignored_but_recognized_invalid_attributes_are_rejected() {
    let mut events = Events::default();
    retain(&mut events, b"native startup noise").unwrap();
    retain(&mut events, b"{unfinished").unwrap();
    retain(&mut events, br#"{"event":"unrelated"}"#).unwrap();
    assert!(events.events.is_empty());
    assert!(
        retain(
            &mut events,
            br#"{"event":"stage.openai_generation_summary","attributes":[]}"#
        )
        .is_err()
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("server.log");
    std::fs::write(
        &path,
        vec![b'x'; usize::try_from(MAX_LINE_BYTES + 1).unwrap()],
    )
    .unwrap();
    assert!(read(&path, None).is_err());
}

#[test]
fn log_identity_permits_append_but_rejects_cross_log_rewrite_and_truncation() {
    use std::io::Write;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("server.log");
    let seed = b"{\"event\":\"stage.openai_generation_summary\",\"attributes\":{\"count\":1}}\n";
    std::fs::write(&path, seed).unwrap();
    let cursor = read(&path, None).unwrap().cursor();
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(seed)
        .unwrap();
    let snapshot = read(&path, Some(&cursor)).unwrap();
    assert_eq!(snapshot.events.after(&cursor, 1).unwrap().events.len(), 1);
    let mut altered = read(&path, None).unwrap().cursor();
    altered.generations = 0;
    assert!(read(&path, Some(&altered)).is_err());
    let other = directory.path().join("other.log");
    std::fs::write(&other, seed).unwrap();
    assert!(read(&other, Some(&cursor)).is_err());
    let rewritten = std::str::from_utf8(seed)
        .unwrap()
        .replace("count\":1", "count\":9");
    std::fs::write(&path, rewritten).unwrap();
    assert!(read(&path, Some(&cursor)).is_err());
    std::fs::write(&path, b"").unwrap();
    assert!(read(&path, Some(&cursor)).is_err());
}

#[test]
fn live_partial_record_cannot_advance_a_cursor_until_its_complete_line_arrives() {
    use std::io::Write;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("server.log");
    let event = b"{\"event\":\"stage.openai_generation_summary\",\"attributes\":{\"count\":1}}\n";
    std::fs::write(&path, event).unwrap();
    let mut writer = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    let middle = event.len() / 2;
    writer.write_all(&event[..middle]).unwrap();
    let cursor = snapshot(&path).unwrap();
    assert_eq!(cursor.bytes, event.len() as u64);
    assert_eq!(cursor.generations, 1);
    assert!(collect(&path, &cursor, 1).unwrap().is_none());
    writer.write_all(&event[middle..event.len() - 1]).unwrap();
    assert!(collect(&path, &cursor, 1).unwrap().is_none());
    writer.write_all(b"\n").unwrap();
    let measured = collect(&path, &cursor, 1).unwrap().unwrap();
    assert_eq!(measured.events.events.len(), 1);
    assert_eq!(measured.cursor.bytes, 2 * event.len() as u64);
    assert_eq!(measured.cursor.generations, 2);
    assert!(collect(&path, &measured.cursor, 1).unwrap().is_none());
}

#[test]
fn a_cursor_digest_for_a_partial_line_is_rejected_even_when_event_counts_match() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("server.log");
    std::fs::write(&path, b"ordinary startup noise\n").unwrap();
    let cursor = Cursor {
        schema_version: 1,
        source_log: path.canonicalize().unwrap(),
        bytes: 4,
        sha256: hex::encode(Sha256::digest(b"ordi")),
        ..Cursor::default()
    };
    assert!(
        read(&path, Some(&cursor))
            .err()
            .unwrap()
            .to_string()
            .contains("complete log line")
    );
}
