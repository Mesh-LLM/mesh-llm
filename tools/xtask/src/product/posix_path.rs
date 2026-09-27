//! `posixpath` behavior the packaging snippets rely on: `abspath`
//! (lexical `normpath` over the working directory), `join` for a relative
//! name, `dirname`, and `realpath` with Python 3.13's `strict` handling.

use crate::artifact::zip_extract::os_error_line;
use std::collections::BTreeMap;
use std::path::Path;

fn cwd() -> Result<String, String> {
    std::env::current_dir()
        .map(|dir| dir.to_string_lossy().into_owned())
        .map_err(|error| os_error_line(&error, "."))
}

/// `os.path.normpath`: two leading slashes survive, `..` above the root vanishes.
pub(super) fn normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".to_owned();
    }
    let root = if path.starts_with("//") && !path.starts_with("///") {
        "//"
    } else if path.starts_with('/') {
        "/"
    } else {
        ""
    };
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." if !root.is_empty() && parts.is_empty() => {}
            ".." if parts.last().is_some_and(|last| *last != "..") => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    let joined = format!("{root}{}", parts.join("/"));
    if joined.is_empty() {
        ".".to_owned()
    } else {
        joined
    }
}

/// `os.path.abspath`.
pub(super) fn abspath(path: &str) -> Result<String, String> {
    if path.starts_with('/') {
        return Ok(normpath(path));
    }
    Ok(normpath(&join(&cwd()?, path)))
}

/// `os.path.join(base, name)` for a relative `name`.
pub(super) fn join(base: &str, name: &str) -> String {
    if base.is_empty() || base.ends_with('/') {
        format!("{base}{name}")
    } else {
        format!("{base}/{name}")
    }
}

/// `os.path.dirname`.
pub(super) fn dirname(path: &str) -> String {
    let head = path.rfind('/').map_or("", |index| &path[..=index]);
    let trimmed = head.trim_end_matches('/');
    if trimmed.is_empty() {
        head.to_owned()
    } else {
        trimmed.to_owned()
    }
}

/// One pending `realpath` part: a name, or the marker that the symlink
/// pushed beneath it is now resolved.
enum Part {
    Name(String),
    Resolved(String),
}

fn push_parts(rest: &mut Vec<Part>, text: &str) -> usize {
    let parts: Vec<&str> = text.split('/').collect();
    let count = parts.len();
    rest.extend(
        parts
            .into_iter()
            .rev()
            .map(|part| Part::Name(part.to_owned())),
    );
    count
}

/// `os.path.realpath(path, strict=strict)` from Python 3.13; a strict
/// failure is the uncaught `OSError` line for the component that failed.
pub(super) fn realpath(path: &str, strict: bool) -> Result<String, String> {
    let mut rest = Vec::new();
    let mut pending = push_parts(&mut rest, path);
    let mut resolved = if path.starts_with('/') {
        "/".to_owned()
    } else {
        cwd()?
    };
    let mut seen: BTreeMap<String, Option<String>> = BTreeMap::new();
    while pending > 0 {
        let name = match rest.pop() {
            Some(Part::Resolved(link)) => {
                seen.insert(link, Some(resolved.clone()));
                continue;
            }
            Some(Part::Name(name)) => name,
            None => break,
        };
        pending -= 1;
        if name.is_empty() || name == "." {
            continue;
        }
        if name == ".." {
            let cut = resolved.rfind('/').unwrap_or(0);
            resolved.truncate(cut);
            if resolved.is_empty() {
                resolved.push('/');
            }
            continue;
        }
        let candidate = if resolved == "/" {
            format!("/{name}")
        } else {
            format!("{resolved}/{name}")
        };
        match step(&candidate, strict, pending > 0, &seen)? {
            Step::Plain => resolved = candidate,
            Step::Cached(target) => resolved = target,
            Step::Link(target) => {
                if target.starts_with('/') {
                    resolved = "/".to_owned();
                }
                seen.insert(candidate.clone(), None);
                rest.push(Part::Resolved(candidate));
                pending += push_parts(&mut rest, &target);
            }
        }
    }
    Ok(resolved)
}

enum Step {
    Plain,
    Cached(String),
    Link(String),
}

fn step(
    candidate: &str,
    strict: bool,
    more: bool,
    seen: &BTreeMap<String, Option<String>>,
) -> Result<Step, String> {
    let raised = |error: std::io::Error| {
        if strict {
            Err(os_error_line(&error, candidate))
        } else {
            Ok(Step::Plain)
        }
    };
    let meta = match std::fs::symlink_metadata(candidate) {
        Ok(meta) => meta,
        Err(error) => return raised(error),
    };
    if !meta.file_type().is_symlink() {
        if strict && more && !meta.is_dir() {
            return raised(std::io::Error::from_raw_os_error(20));
        }
        return Ok(Step::Plain);
    }
    match seen.get(candidate) {
        Some(Some(target)) => return Ok(Step::Cached(target.clone())),
        Some(None) if strict => {
            return std::fs::metadata(candidate).map_or_else(raised, |_| Ok(Step::Plain));
        }
        Some(None) => return Ok(Step::Plain),
        None => {}
    }
    match std::fs::read_link(Path::new(candidate)) {
        Ok(target) => Ok(Step::Link(target.to_string_lossy().into_owned())),
        Err(error) => raised(error),
    }
}

#[cfg(test)]
mod tests {
    use super::{dirname, join, normpath};

    #[test]
    fn migration_product_posix_path_matches_python() {
        assert_eq!(normpath("//a/./b/../c/"), "//a/c");
        assert_eq!(normpath("/../x"), "/x");
        assert_eq!(normpath("a/../../b"), "../b");
        assert_eq!(normpath(""), ".");
        assert_eq!(join("", "archive-0"), "archive-0");
        assert_eq!(join("t/", "archive-0"), "t/archive-0");
        assert_eq!(dirname("/a/b"), "/a");
        assert_eq!(dirname("/b"), "/");
        assert_eq!(dirname("b"), "");
        assert_eq!(dirname("a//b"), "a");
    }
}
