use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use super::*;

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

/// A GGUF with one f32 matrix `w` of `rows` rows of four values, row `r`
/// holding `r` in every value.
fn gguf(rows: u64) -> Vec<u8> {
    let string = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    };
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    string(&mut out, "general.architecture");
    out.extend(8u32.to_le_bytes());
    string(&mut out, "test");
    string(&mut out, "w");
    out.extend(2u32.to_le_bytes());
    out.extend(4u64.to_le_bytes());
    out.extend(rows.to_le_bytes());
    out.extend(0u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.resize(out.len().div_ceil(32) * 32, 0);
    for r in 0..rows {
        for _ in 0..4 {
            out.extend((r as f32).to_le_bytes());
        }
    }
    out
}

/// What the fixture server saw: per request, its method, path and whether
/// it carried the token.
type Seen = Arc<Mutex<Vec<(String, String, bool)>>>;

/// A resolver that redirects to a CDN path on the same server, which is
/// another "host" because the redirect names 127.0.0.1 and the resolver
/// localhost.
fn serve(bytes: Vec<u8>) -> (String, Seen) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen: Seen = Arc::default();
    let log = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let bytes = bytes.clone();
            let log = log.clone();
            std::thread::spawn(move || answer(stream, port, &bytes, &log));
        }
    });
    (format!("http://localhost:{port}"), seen)
}

fn answer(mut stream: TcpStream, port: u16, bytes: &[u8], seen: &Seen) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let mut parts = line.split_whitespace();
    let (method, path) = (
        parts.next().unwrap().to_owned(),
        parts.next().unwrap().to_owned(),
    );
    let (mut range, mut token) = (None, false);
    loop {
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        let (name, value) = header.split_once(": ").unwrap();
        match name.to_ascii_lowercase().as_str() {
            "range" => range = Some(value.trim_start_matches("bytes=").to_owned()),
            "authorization" => token = value == "Bearer secret",
            _ => {}
        }
    }
    seen.lock()
        .unwrap()
        .push((method.clone(), path.clone(), token));
    let resolver = path.starts_with("/owner/model/resolve/") && path.ends_with("/sub/model.gguf");
    let response = if resolver && method == "HEAD" {
        format!(
            "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{port}/cdn/model.gguf?sig=1\r\n\
             x-repo-commit: {COMMIT}\r\nx-linked-size: {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            bytes.len()
        )
        .into_bytes()
    } else if path == "/cdn/model.gguf?sig=1" && method == "GET" {
        let (start, end) = range
            .unwrap()
            .split_once('-')
            .map(|(s, e)| (s.parse::<usize>().unwrap(), e.parse::<usize>().unwrap()))
            .unwrap();
        let body = &bytes[start..=end];
        let mut out = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            bytes.len(),
            body.len()
        )
        .into_bytes();
        out.extend_from_slice(body);
        out
    } else {
        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
    };
    let _ = stream.write_all(&response);
}

fn hf_file() -> HfFile {
    HfFile::parse("owner/model@main/sub/model.gguf").unwrap()
}

#[test]
fn a_hugging_face_file_is_read_by_range_into_a_pinned_copy() {
    // past the first megabyte, which the header read takes
    let bytes = gguf(100_000);
    let (endpoint, seen) = serve(bytes.clone());
    let cache = tempfile::tempdir().unwrap();
    let claimed = open_hf(&hf_file(), &endpoint, Some("secret".into()), cache.path()).unwrap();
    assert_eq!(
        claimed.path,
        cache
            .path()
            .join("owner--model")
            .join(COMMIT)
            .join("sub/model.gguf")
    );
    assert_eq!(claimed.weights.row("w", 90_000).unwrap(), vec![90_000.0; 4]);
    let seen = seen.lock().unwrap().clone();
    // the token goes to the resolver, not to the CDN
    assert!(
        seen.iter()
            .any(|(m, p, token)| m == "HEAD" && p.contains("/resolve/") && *token)
    );
    assert!(
        seen.iter()
            .filter(|(_, p, _)| p.starts_with("/cdn/"))
            .all(|(_, _, token)| !token)
    );
    // the copy holds what was read, not the file
    let held = std::fs::read(&claimed.path).unwrap();
    assert_eq!(held.len(), bytes.len());
    let row = claimed.weights.row_range("w", 90_000).unwrap();
    let (start, end) = (row.0 as usize, (row.0 + row.1) as usize);
    assert_eq!(held[start..end], bytes[start..end]);
    assert!(held[start - 16..start].iter().all(|&b| b == 0));
}

#[test]
fn a_reopened_copy_reads_nothing_it_holds() {
    let (endpoint, seen) = serve(gguf(1000));
    let cache = tempfile::tempdir().unwrap();
    open_hf(&hf_file(), &endpoint, None, cache.path())
        .unwrap()
        .weights
        .row("w", 3)
        .unwrap();
    let before = seen.lock().unwrap().len();
    let claimed = open_hf(&hf_file(), &endpoint, None, cache.path()).unwrap();
    assert_eq!(claimed.weights.row("w", 3).unwrap(), vec![3.0; 4]);
    let after: Vec<_> = seen.lock().unwrap()[before..].to_vec();
    // only the resolver again: the header and row were held
    assert_eq!(after.len(), 1, "{after:?}");
    assert_eq!(after[0].0, "HEAD");
}

#[test]
fn a_moved_revision_is_refused_for_a_commit() {
    let (endpoint, _) = serve(gguf(10));
    let cache = tempfile::tempdir().unwrap();
    let mut file = hf_file();
    file.revision = "f".repeat(40);
    // the fixture resolves every revision to its one commit
    assert!(open_hf(&file, &endpoint, None, cache.path()).is_err());
}

#[test]
fn references_parse_as_repo_revision_and_file() {
    assert_eq!(
        HfFile::parse("unsloth/Qwen3-4B-GGUF@main/Q8_0/Qwen3-4B-Q8_0.gguf"),
        Some(HfFile {
            repo: "unsloth/Qwen3-4B-GGUF".into(),
            revision: "main".into(),
            file: "Q8_0/Qwen3-4B-Q8_0.gguf".into(),
        })
    );
    assert_eq!(HfFile::parse("Qwen3-4B-Q8_0.gguf"), None);
    assert_eq!(HfFile::parse("owner/model@main"), None);
    assert_eq!(HfFile::parse("owner/model@main/../x.gguf"), None);
    assert_eq!(HfFile::parse("model@main/x.gguf"), None);
}

/// Shard `no` (from 0) of `count`, holding an f32 matrix `name` of `rows`
/// rows of four values, row `r` holding `r`.
fn shard(no: u64, count: u64, name: &str, rows: u64) -> Vec<u8> {
    let string = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    };
    let uint = |out: &mut Vec<u8>, key: &str, value: u64| {
        string(out, key);
        out.extend(10u32.to_le_bytes());
        out.extend(value.to_le_bytes());
    };
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    out.extend(2u64.to_le_bytes());
    uint(&mut out, "split.no", no);
    uint(&mut out, "split.count", count);
    string(&mut out, name);
    out.extend(2u32.to_le_bytes());
    out.extend(4u64.to_le_bytes());
    out.extend(rows.to_le_bytes());
    out.extend(0u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.resize(out.len().div_ceil(32) * 32, 0);
    for r in 0..rows {
        for _ in 0..4 {
            out.extend((r as f32).to_le_bytes());
        }
    }
    out
}

#[test]
fn a_local_split_gguf_opens_every_shard() {
    let dir = tempfile::tempdir().unwrap();
    for (no, name) in ["a", "b", "c"].iter().enumerate() {
        let path = dir.path().join(format!("m-{:05}-of-00003.gguf", no + 1));
        std::fs::write(path, shard(no as u64, 3, name, 3 + no as u64)).unwrap();
    }
    let first = dir.path().join("m-00001-of-00003.gguf");
    let claimed = open(&first, dir.path()).unwrap();
    assert_eq!(claimed.path, first);
    assert_eq!(claimed.weights.row("a", 2).unwrap(), vec![2.0; 4]);
    assert_eq!(claimed.weights.row("c", 4).unwrap(), vec![4.0; 4]);

    let second = dir.path().join("m-00002-of-00003.gguf");
    let refused = open(&second, dir.path()).err().unwrap().to_string();
    assert!(refused.contains("claim its first shard"), "{refused}");
}
