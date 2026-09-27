//! `native select-runtime`: the Rust owner of
//! `scripts/select-native-runtime.py`. Selects the one runtime directory
//! under `--root` whose `manifest.json` matches the platform and backend,
//! printing it; an ambiguous or empty selection (and any malformed manifest)
//! fails like the legacy script's uncaught exception, as its final traceback
//! line with status 1.

use super::argv::{Grammar, Opt};
use super::manifest_json::{Raised, load_manifest, python_str, subscript};
use crate::ci_plan::catalog::python_path_display;
use crate::ci_plan::document::Json;
use crate::repository::check_report::CheckReport;
use std::path::Path;

const GRAMMAR: Grammar = Grammar {
    prog: "select-native-runtime.py",
    usage: "\
usage: select-native-runtime.py [-h] --root ROOT --os OS --arch ARCH
                                --backend BACKEND [--cuda-major CUDA_MAJOR]
",
    help: "\
usage: select-native-runtime.py [-h] --root ROOT --os OS --arch ARCH
                                --backend BACKEND [--cuda-major CUDA_MAJOR]

options:
  -h, --help            show this help message and exit
  --root ROOT
  --os OS
  --arch ARCH
  --backend BACKEND
  --cuda-major CUDA_MAJOR
",
    options: &[
        Opt::required("--root"),
        Opt::required("--os"),
        Opt::required("--arch"),
        Opt::required("--backend"),
        Opt::value("--cuda-major"),
    ],
    positional: None,
};

/// The selection request, as the legacy `select_runtime` arguments.
struct Request<'a> {
    root: &'a str,
    os: &'a str,
    arch: &'a str,
    backend: &'a str,
    cuda_major: &'a str,
}

pub(super) fn run(args: &[String]) -> CheckReport {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let request = Request {
        root: parsed.value("--root").unwrap_or_default(),
        os: parsed.value("--os").unwrap_or_default(),
        arch: parsed.value("--arch").unwrap_or_default(),
        backend: parsed.value("--backend").unwrap_or_default(),
        cuda_major: parsed.value("--cuda-major").unwrap_or_default(),
    };
    match select(&request) {
        Ok(path) => CheckReport::success(format!("{path}\n")),
        Err(Raised(line)) => CheckReport::failure(String::new(), format!("{line}\n")),
    }
}

/// `str(Path(root) / name)`.
fn child(root: &str, name: &str) -> String {
    match root {
        "." => name.to_owned(),
        _ if root.ends_with('/') => format!("{root}{name}"),
        _ => format!("{root}/{name}"),
    }
}

/// The runtime directory names `sorted(root.glob("*/manifest.json"))`
/// yields: every entry that is (or links to) a directory holding a
/// `manifest.json` entry of any kind, ordered by name.
fn candidates(root: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .filter(|entry| {
            entry
                .path()
                .join("manifest.json")
                .symlink_metadata()
                .is_ok()
        })
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn select(request: &Request<'_>) -> Result<String, Raised> {
    let expected = match request.backend {
        "cuda-blackwell" => "cuda",
        "hip" => "rocm",
        other => other,
    };
    let root = python_path_display(Path::new(request.root));
    let mut matches = Vec::new();
    if Path::new(&root).is_dir() {
        for name in candidates(Path::new(&root)) {
            let directory = child(&root, &name);
            let manifest = child(&directory, "manifest.json");
            let document = load_manifest(Path::new(&manifest), &manifest)?;
            if matches_request(&document, request, expected)? {
                matches.push(directory);
            }
        }
    }
    match matches.as_slice() {
        [only] => Ok(only.clone()),
        _ => {
            let rendered = if matches.is_empty() {
                "none".to_owned()
            } else {
                matches.join(", ")
            };
            Err(Raised(format!(
                "ValueError: expected exactly one native runtime for {}/{}/{}; found {rendered} under {root}",
                request.os, request.arch, request.backend
            )))
        }
    }
}

/// The legacy filter, evaluated in its order so the first failing
/// subscript raises the same exception.
fn matches_request(document: &Json, request: &Request<'_>, expected: &str) -> Result<bool, Raised> {
    let runtime = subscript(document, "runtime")?;
    let platform = subscript(runtime, "platform")?;
    let backend = subscript(runtime, "backend")?;
    if subscript(platform, "os")?.as_str() != Some(request.os)
        || subscript(platform, "arch")?.as_str() != Some(request.arch)
    {
        return Ok(false);
    }
    if subscript(backend, "kind")?.as_str() != Some(expected) {
        return Ok(false);
    }
    if expected == "cuda" && !request.cuda_major.is_empty() {
        let major = python_str(backend, "cuda", "toolkit_major")?;
        if major != request.cuda_major {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::child;

    #[test]
    fn migration_native_policy_child_paths_follow_pathlib() {
        assert_eq!(child(".", "cpu"), "cpu");
        assert_eq!(child("/", "cpu"), "/cpu");
        assert_eq!(child("dist/runtimes", "cpu"), "dist/runtimes/cpu");
    }
}
