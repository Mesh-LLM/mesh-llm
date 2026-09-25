//! `static-abi-stamp`: the portable llama.cpp static-ABI build-stamp contract
//! from `scripts/verify-static-abi-build-stamp.py`.

use super::python_io;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::{repr, splitlines};
use std::collections::BTreeMap;
use std::path::Path;

const GRAMMAR: Grammar = Grammar {
    usage: "verify-static-abi-build-stamp.py [-h] --backend BACKEND --link-mode LINK_MODE \
            --stamp-version STAMP_VERSION --toolchain-epoch TOOLCHAIN_EPOCH \
            [--patched-sha PATCHED_SHA] stamp",
    values: &[
        "--backend",
        "--link-mode",
        "--stamp-version",
        "--toolchain-epoch",
        "--patched-sha",
    ],
    flags: &[],
};

const REQUIRED_FIELDS: [&str; 5] = [
    "stamp-version",
    "patched-sha",
    "backend",
    "link-mode",
    "toolchain-epoch",
];

/// Expected identity; `patched_sha` is checked only when given.
struct Expected<'a> {
    backend: &'a str,
    link_mode: &'a str,
    stamp_version: &'a str,
    toolchain_epoch: &'a str,
    patched_sha: Option<&'a str>,
}

struct Stamp {
    fields: BTreeMap<String, String>,
    cmake_arguments: usize,
}

pub(super) fn run(args: &[String]) -> CheckReport {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let required = [
        "--backend",
        "--link-mode",
        "--stamp-version",
        "--toolchain-epoch",
    ];
    let missing: Vec<&str> = required
        .into_iter()
        .filter(|name| parsed.last(name).is_none())
        .collect();
    let [stamp] = parsed.positionals.as_slice() else {
        return GRAMMAR.error("expected exactly one stamp argument");
    };
    if !missing.is_empty() {
        let names = missing.join(", ");
        return GRAMMAR.error(&format!("the following arguments are required: {names}"));
    }
    let value = |name| parsed.last(name).unwrap_or_default();
    let expected = Expected {
        backend: value("--backend"),
        link_mode: value("--link-mode"),
        stamp_version: value("--stamp-version"),
        toolchain_epoch: value("--toolchain-epoch"),
        patched_sha: parsed.last("--patched-sha"),
    };
    match verify(Path::new(stamp), &expected) {
        Ok(stamp) => CheckReport::success(format!(
            "verified static ABI build stamp: backend={} cmake_arguments={}\n",
            stamp.fields.get("backend").map_or("", String::as_str),
            stamp.cmake_arguments
        )),
        Err(message) => CheckReport::failure(String::new(), format!("{message}\n")),
    }
}

fn verify(path: &Path, expected: &Expected<'_>) -> Result<Stamp, String> {
    let stamp = parse(path)?;
    require_equal(&stamp, "backend", expected.backend)?;
    require_equal(&stamp, "link-mode", expected.link_mode)?;
    require_equal(&stamp, "stamp-version", expected.stamp_version)?;
    require_equal(&stamp, "toolchain-epoch", expected.toolchain_epoch)?;
    if let Some(sha) = expected.patched_sha {
        require_equal(&stamp, "patched-sha", sha)?;
    }
    Ok(stamp)
}

fn field_name(key: &str) -> bool {
    let mut chars = key.chars();
    chars.next().is_some_and(|ch| ch.is_ascii_lowercase())
        && chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
}

fn parse(path: &Path) -> Result<Stamp, String> {
    let text = python_io::read_text(path)
        .map_err(|error| format!("unable to read static ABI build stamp: {error}"))?;
    let mut fields = BTreeMap::new();
    let mut cmake_arguments = 0;
    for (index, line) in splitlines(&text).into_iter().enumerate() {
        let Some((key, value)) = line.split_once('=').filter(|(key, _)| field_name(key)) else {
            return Err(format!(
                "static ABI build stamp line {} is malformed",
                index + 1
            ));
        };
        if key == "cmake-arg" {
            cmake_arguments += 1;
        } else if fields.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(format!(
                "static ABI build stamp repeats singleton field {}",
                repr(key)
            ));
        }
    }
    let missing: Vec<&str> = REQUIRED_FIELDS
        .into_iter()
        .filter(|name| fields.get(*name).is_none_or(String::is_empty))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "static ABI build stamp is missing required fields: {}",
            missing.join(", ")
        ));
    }
    if cmake_arguments == 0 {
        return Err("static ABI build stamp must contain at least one cmake-arg".to_owned());
    }
    Ok(Stamp {
        fields,
        cmake_arguments,
    })
}

fn require_equal(stamp: &Stamp, name: &str, expected: &str) -> Result<(), String> {
    let actual = stamp.fields.get(name);
    if actual.map(String::as_str) == Some(expected) {
        return Ok(());
    }
    let actual = actual.map_or_else(|| "None".to_owned(), |value| repr(value));
    Err(format!(
        "static ABI build stamp {name} mismatch: expected {}, got {actual}",
        repr(expected)
    ))
}
