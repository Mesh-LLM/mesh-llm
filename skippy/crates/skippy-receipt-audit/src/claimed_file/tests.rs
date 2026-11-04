use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use super::*;

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

fn string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

/// A GGUF with one f32 matrix `w` of `rows` rows of four values, row `r`
/// holding `r` in every value.
fn gguf(rows: u64) -> Vec<u8> {
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

/// A GGUF whose header lists the tensor `w` twice, which the runtime's
/// loader refuses.
fn gguf_with_a_tensor_twice() -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(2u64.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    for offset in [0u64, 32] {
        string(&mut out, "w");
        out.extend(2u32.to_le_bytes());
        out.extend(4u64.to_le_bytes());
        out.extend(2u64.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(offset.to_le_bytes());
    }
    out.resize(out.len().div_ceil(32) * 32 + 64, 0);
    out
}

/// What the fixture server saw: per request, its method, path and whether
/// it carried the token.
type Seen = Arc<Mutex<Vec<(String, String, bool)>>>;

/// A resolver that redirects to a CDN path on the same server, which is
/// another origin because the redirect names 127.0.0.1 and the resolver
/// localhost. The repo in the path picks what the resolver says:
///
/// - `owner/model`: the commit, the size and the redirect.
/// - `owner/nocommit`: no commit.
/// - `owner/badcommit`: a commit that is a path.
/// - `owner/direct`: the file itself, whose range requests it redirects.
fn serve(bytes: Vec<u8>) -> (String, Seen) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen: Seen = Arc::default();
    let log = seen.clone();
    let bytes = Arc::new(bytes);
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
    let cdn = format!("Location: http://127.0.0.1:{port}/cdn/model.gguf?sig=1\r\n");
    let size = format!("x-linked-size: {}\r\n", bytes.len());
    let commit = format!("x-repo-commit: {COMMIT}\r\n");
    let repo = path
        .strip_suffix("/sub/model.gguf")
        .and_then(|p| p.split("/resolve/").next())
        .unwrap_or_default();
    let head = method == "HEAD";
    let response = match repo {
        "/owner/model" if head => redirect(&[&cdn, &commit, &size]),
        "/owner/nocommit" if head => redirect(&[&cdn, &size]),
        "/owner/badcommit" if head => redirect(&[&cdn, "x-repo-commit: ../../../etc\r\n", &size]),
        "/owner/direct" if head => format!(
            "HTTP/1.1 200 OK\r\n{commit}{size}Content-Length: 0\r\nConnection: close\r\n\r\n"
        )
        .into_bytes(),
        "/owner/direct" if method == "GET" => redirect(&[&cdn]),
        _ if path == "/cdn/model.gguf?sig=1" && method == "GET" => ranged(bytes, &range.unwrap()),
        _ => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
    };
    let _ = stream.write_all(&response);
}

fn redirect(headers: &[&str]) -> Vec<u8> {
    format!(
        "HTTP/1.1 302 Found\r\n{}Content-Length: 0\r\nConnection: close\r\n\r\n",
        headers.concat()
    )
    .into_bytes()
}

fn ranged(bytes: &[u8], range: &str) -> Vec<u8> {
    let (start, end) = range
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
}

fn hf_file() -> HfFile {
    HfFile::parse("owner/model@main/sub/model.gguf").unwrap()
}

fn in_repo(repo: &str) -> HfFile {
    HfFile::parse(&format!("owner/{repo}@main/sub/model.gguf")).unwrap()
}

/// Every file under `dir`, to show an open that was refused wrote none.
fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(kind) if kind.is_dir() => files.extend(files_under(&path)),
            _ => files.push(path),
        }
    }
    files
}

#[test]
fn a_hugging_face_file_is_read_by_range_into_a_pinned_copy() {
    // past the megabytes the first read of the header takes
    let bytes = gguf(1_000_000);
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
    // the branch resolved to a commit, which the caller gets to keep
    let pinned = format!("owner/model@{COMMIT}/sub/model.gguf");
    assert_eq!(claimed.identity, pinned);
    let kept = claimed.pinned.clone().unwrap();
    assert!(kept.is_pinned());
    assert_eq!(kept.to_string(), pinned);
    assert_eq!(
        claimed.weights.row("w", 900_000).unwrap(),
        vec![900_000.0; 4]
    );
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
    // the header took one request, the row one more
    let fetches = seen.iter().filter(|(m, _, _)| m == "GET").count();
    assert_eq!(fetches, 2, "{seen:?}");
    // the copy holds what was read, not the file
    let held = std::fs::read(&claimed.path).unwrap();
    assert_eq!(held.len(), bytes.len());
    let row = claimed.weights.row_range("w", 900_000).unwrap();
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
    // the fixture resolves every revision to its one commit
    let moved = hf_file().at(&"f".repeat(40));
    assert!(open_hf(&moved, &endpoint, None, cache.path()).is_err());
    assert!(open_hf(&hf_file().at(COMMIT), &endpoint, None, cache.path()).is_ok());
}

#[test]
fn a_revision_the_resolver_does_not_pin_is_refused() {
    let (endpoint, _) = serve(gguf(10));
    let cache = tempfile::tempdir().unwrap();
    // a branch, and no commit from the resolver: the copy would be of
    // whatever the branch is at when a range is read
    let refused = open_hf(&in_repo("nocommit"), &endpoint, None, cache.path())
        .err()
        .unwrap()
        .to_string();
    assert!(refused.contains("which commit"), "{refused}");
    // a commit names itself
    let named = in_repo("nocommit").at(COMMIT);
    let claimed = open_hf(&named, &endpoint, None, cache.path()).unwrap();
    assert_eq!(claimed.pinned, Some(named));
    // a commit that is a path names no directory of the cache
    let refused = open_hf(&in_repo("badcommit"), &endpoint, None, cache.path());
    assert!(refused.is_err());
    assert_eq!(files_under(cache.path()).len(), 3, "the one good copy");
}

#[test]
fn a_receipt_must_name_a_commit() {
    let cache = tempfile::tempdir().unwrap();
    // refused before any request: these reach no server
    for name in [
        "owner/model@main/sub/model.gguf",
        "owner/model@v1.0/sub/model.gguf",
        "owner/model@0123456789abcdef/sub/model.gguf",
    ] {
        let error = open_named_by_receipt(name, cache.path()).err().unwrap();
        assert!(CannotAudit::caused(&error), "{error:#}");
        assert!(error.to_string().contains("not a commit"), "{error}");
        let file = HfFile::parse(name).unwrap();
        assert!(open_remote(&file, cache.path()).is_err());
    }
    assert!(require_pinned(&hf_file().at(COMMIT)).is_ok());
    assert!(files_under(cache.path()).is_empty());
}

#[test]
fn a_name_out_of_a_receipt_never_opens_a_file_here() {
    let dir = tempfile::tempdir().unwrap();
    let local = dir.path().join("model.gguf");
    std::fs::write(&local, gguf(4)).unwrap();
    // the user may name the file
    assert!(open(&local, dir.path()).is_ok());
    // a node may not: its name is a Hugging Face file or nothing
    let cache = dir.path().join("cache");
    let named = local.to_str().unwrap();
    let refused = open_named_by_receipt(named, &cache).err().unwrap();
    assert!(
        refused.to_string().contains("the receipt names"),
        "{refused}"
    );
    for name in ["model.gguf", "./model.gguf", "../model.gguf", ""] {
        assert!(open_named_by_receipt(name, &cache).is_err(), "{name:?}");
    }
    assert!(!cache.exists());
}

#[test]
fn a_name_that_is_a_path_writes_nothing_outside_the_cache() {
    let (endpoint, seen) = serve(gguf(10));
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    let target = dir.path().join("authorized_keys");
    std::fs::write(&target, b"the user's keys").unwrap();
    let absolute = target.to_str().unwrap().to_owned();
    let names = [
        // the doubled slash makes the file an absolute path
        format!("evil/m@main/{absolute}"),
        format!("evil/m@{COMMIT}/{absolute}"),
        format!("evil/m@main//{}", absolute.trim_start_matches('/')),
        "evil/m@main/../../authorized_keys".to_owned(),
        format!("evil/m@main/sub/../../../../{}", "authorized_keys"),
    ];
    for name in &names {
        assert_eq!(HfFile::parse(name), None, "{name}");
        assert!(open_named_by_receipt(name, &cache).is_err());
        assert!(open(Path::new(name), &cache).is_err());
    }
    // a name built past the parser is checked again before it is a URL or
    // a path
    for file in [absolute.as_str(), "../../authorized_keys", "/x.gguf"] {
        let built = hf_file().at(COMMIT).sibling(file);
        assert!(open_hf(&built, &endpoint, None, &cache).is_err(), "{file}");
    }
    assert_eq!(std::fs::read(&target).unwrap(), b"the user's keys");
    assert!(files_under(dir.path()).len() == 1, "only the user's file");
    assert!(seen.lock().unwrap().is_empty(), "no name reached a server");
}

#[cfg(unix)]
#[test]
fn a_symlinked_cache_directory_holds_no_copy() {
    let (endpoint, _) = serve(gguf(10));
    let cache = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(elsewhere.path(), cache.path().join("owner--model")).unwrap();
    let refused = open_hf(&hf_file(), &endpoint, None, cache.path())
        .err()
        .unwrap()
        .to_string();
    assert!(refused.contains("symlink"), "{refused}");
    assert!(files_under(elsewhere.path()).is_empty());
}

#[test]
fn a_redirect_to_another_origin_carries_no_token() {
    let bytes = gguf(1000);
    let (endpoint, seen) = serve(bytes);
    let cache = tempfile::tempdir().unwrap();
    // the resolver serves this file itself, then redirects its range
    // requests to the CDN
    let claimed = open_hf(
        &in_repo("direct"),
        &endpoint,
        Some("secret".into()),
        cache.path(),
    )
    .unwrap();
    assert_eq!(claimed.weights.row("w", 7).unwrap(), vec![7.0; 4]);
    let seen = seen.lock().unwrap().clone();
    let gets = |prefix: &str| -> Vec<bool> {
        seen.iter()
            .filter(|(method, path, _)| method == "GET" && path.starts_with(prefix))
            .map(|(_, _, token)| *token)
            .collect()
    };
    // the resolver's origin sees the token, the origin it redirects to not
    assert!(!gets("/owner/direct/").is_empty());
    assert!(gets("/owner/direct/").iter().all(|token| *token));
    assert!(!gets("/cdn/").is_empty());
    assert!(gets("/cdn/").iter().all(|token| !token), "{seen:?}");
}

#[test]
fn a_header_the_loader_would_not_take_is_refused_at_open() {
    let (endpoint, _) = serve(gguf_with_a_tensor_twice());
    let cache = tempfile::tempdir().unwrap();
    let error = open_hf(&hf_file(), &endpoint, None, cache.path())
        .err()
        .unwrap();
    assert!(CannotAudit::caused(&error), "{error:#}");
    // and again from the copy, before a native load of it
    let copy = cache
        .path()
        .join("owner--model")
        .join(COMMIT)
        .join("sub/model.gguf");
    let error = check_header(&copy).err().unwrap();
    assert!(CannotAudit::caused(&error), "{error:#}");
}

#[test]
fn the_header_of_a_copy_checks_from_the_copy() {
    let (endpoint, _) = serve(gguf(1000));
    let cache = tempfile::tempdir().unwrap();
    let claimed = open_hf(&hf_file(), &endpoint, None, cache.path()).unwrap();
    check_header(&claimed.path).unwrap();
    // a copy that lost its header is a hole, and no GGUF
    let lost = cache.path().join("lost.gguf");
    std::fs::write(&lost, vec![0u8; 4096]).unwrap();
    let error = check_header(&lost).err().unwrap();
    assert!(CannotAudit::caused(&error), "{error:#}");
}

/// Shard `no` (from 0) of `count`, holding an f32 matrix `name` of `rows`
/// rows of four values, row `r` holding `r`.
fn shard(no: u64, count: u64, name: &str, rows: u64) -> Vec<u8> {
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
    assert_eq!(claimed.pinned, None);
    assert_eq!(claimed.weights.row("a", 2).unwrap(), vec![2.0; 4]);
    assert_eq!(claimed.weights.row("c", 4).unwrap(), vec![4.0; 4]);
    check_header(&first).unwrap();

    let second = dir.path().join("m-00002-of-00003.gguf");
    let refused = open(&second, dir.path()).err().unwrap().to_string();
    assert!(refused.contains("claim its first shard"), "{refused}");
}

#[test]
fn a_split_of_too_many_shards_opens_none() {
    let (endpoint, seen) = serve(gguf(10));
    let cache = tempfile::tempdir().unwrap();
    let many = HfFile::parse("owner/model@main/sub/m-00001-of-99999.gguf").unwrap();
    let error = open_hf(&many, &endpoint, None, cache.path()).err().unwrap();
    assert!(CannotAudit::caused(&error), "{error:#}");
    // not even the first shard is asked for
    assert!(seen.lock().unwrap().is_empty());
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("m-00001-of-99999.gguf");
    std::fs::write(&first, shard(0, 99_999, "a", 3)).unwrap();
    let error = open(&first, dir.path()).err().unwrap();
    assert!(CannotAudit::caused(&error), "{error:#}");
}
