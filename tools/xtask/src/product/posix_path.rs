use std::path::{Component, Path, PathBuf};

fn cwd() -> Result<String, String> {
    std::env::current_dir()
        .map(|directory| directory.to_string_lossy().into_owned())
        .map_err(|error| error.to_string())
}

pub(super) fn normpath(path: &str) -> String {
    let mut normalized = PathBuf::new();
    for component in Path::new(path).components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if normalized.file_name().is_some_and(|name| name != "..") {
                    normalized.pop();
                } else if !normalized.has_root() {
                    normalized.push("..");
                }
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    if normalized.as_os_str().is_empty() {
        ".".into()
    } else {
        normalized.to_string_lossy().into_owned()
    }
}

pub(super) fn abspath(path: &str) -> Result<String, String> {
    if Path::new(path).is_absolute() {
        Ok(normpath(path))
    } else {
        Ok(normpath(&join(&cwd()?, path)))
    }
}

pub(super) fn join(base: &str, name: &str) -> String {
    Path::new(base).join(name).to_string_lossy().into_owned()
}

pub(super) fn dirname(path: &str) -> String {
    Path::new(path)
        .parent()
        .map_or_else(String::new, |parent| parent.to_string_lossy().into_owned())
}

pub(super) fn realpath(path: &str, strict: bool) -> Result<String, String> {
    let mut candidate = PathBuf::from(path);
    let mut suffix = Vec::new();
    loop {
        match std::fs::canonicalize(&candidate) {
            Ok(mut resolved) => {
                for name in suffix.into_iter().rev() {
                    resolved.push(name);
                }
                return Ok(normpath(&resolved.to_string_lossy()));
            }
            Err(error) => {
                if strict
                    || error.kind() != std::io::ErrorKind::NotFound
                    || std::fs::symlink_metadata(&candidate).is_ok()
                {
                    return Err(format!("{}: {error}", candidate.display()));
                }
                let name = candidate
                    .file_name()
                    .ok_or_else(|| format!("{}: {error}", candidate.display()))?
                    .to_os_string();
                suffix.push(name);
                if !candidate.pop() {
                    return Err(format!("{path}: {error}"));
                }
                if candidate.as_os_str().is_empty() {
                    candidate = std::env::current_dir().map_err(|error| error.to_string())?;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{dirname, join, normpath};

    #[test]
    fn lexical_paths_preserve_ancestry_without_interpreter_root_rules() {
        assert_eq!(normpath("//a/./b/../c/"), "/a/c");
        assert_eq!(normpath("/../x"), "/x");
        assert_eq!(normpath("a/../../b"), "../b");
        assert_eq!(normpath(""), ".");
        assert_eq!(join("", "archive-0"), "archive-0");
        assert_eq!(dirname("/a/b"), "/a");
        assert_eq!(dirname("/b"), "/");
    }
}
