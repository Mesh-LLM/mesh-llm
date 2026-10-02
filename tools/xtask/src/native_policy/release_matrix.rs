use crate::ci_plan::document::Json;
use crate::repository::check_report::CheckReport;
use crate::repository::text::repr;
use std::collections::BTreeSet;

const USAGE: &str = "usage: validate-release-native-runtime-matrix.py [-h] --manifest MANIFEST\n                                                 [--required-target REQUIRED_TARGET]\n                                                 [assets ...]\n";

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
struct Target {
    os: String,
    arch: String,
    backend: String,
    major: Option<String>,
}

impl Target {
    fn label(&self) -> String {
        format!(
            "{}/{}/{}{}",
            self.os,
            self.arch,
            self.backend,
            self.major.as_deref().unwrap_or("")
        )
    }

    fn covers(&self, candidate: &Self) -> bool {
        self.os == candidate.os
            && self.arch == candidate.arch
            && self.backend == candidate.backend
            && (self.backend != "cuda" || self.major.is_none() || self.major == candidate.major)
    }
}

fn normalize_major(digits: &str) -> String {
    let normalized = digits.trim_start_matches('0');
    if normalized.is_empty() {
        "0".to_owned()
    } else {
        normalized.to_owned()
    }
}

fn label_target(label: &str) -> Result<Target, String> {
    let words: Vec<&str> = label.split('/').collect();
    let [os, arch, backend] = words.as_slice() else {
        return Err(format!(
            "expected target label as os/arch/backend, got {}",
            repr(label)
        ));
    };
    let (backend, major) = match backend.strip_prefix("cuda") {
        Some(digits) if !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()) => {
            ("cuda", Some(normalize_major(digits)))
        }
        _ => (*backend, None),
    };
    Ok(Target {
        os: (*os).to_owned(),
        arch: (*arch).to_owned(),
        backend: backend.to_owned(),
        major,
    })
}

fn asset_target(asset: &str) -> Result<Option<Target>, String> {
    let name = asset.rsplit(['/', '\\']).next().unwrap_or(asset);
    if !name.starts_with("mesh-llm-") || name.ends_with(".sha256") {
        return Ok(None);
    }
    let Some(stem) = name
        .strip_suffix(".tar.gz")
        .or_else(|| name.strip_suffix(".zip"))
    else {
        return Ok(None);
    };
    let triples = [
        ("aarch64-apple-darwin", "macos", "aarch64"),
        ("x86_64-apple-darwin", "macos", "x86_64"),
        ("x86_64-unknown-linux-gnu", "linux", "x86_64"),
        ("aarch64-unknown-linux-gnu", "linux", "aarch64"),
        ("x86_64-pc-windows-msvc", "windows", "x86_64"),
    ];
    for (triple, os, arch) in triples {
        if let Some((_, suffix)) = stem.split_once(&format!("-{triple}")) {
            let (backend, major) = if suffix.is_empty() {
                (
                    if os == "macos" && arch == "aarch64" {
                        "metal"
                    } else {
                        "cpu"
                    },
                    None,
                )
            } else if let Some(cuda) = suffix.strip_prefix("-cuda") {
                match cuda {
                    "" => ("cuda", None),
                    value
                        if value.starts_with('-')
                            && value[1..].bytes().all(|byte| byte.is_ascii_digit())
                            && value.len() > 1 =>
                    {
                        ("cuda", Some(normalize_major(&value[1..])))
                    }
                    _ => return Err(format!("unsupported CUDA release suffix: {suffix}")),
                }
            } else {
                (suffix.strip_prefix('-').unwrap_or(suffix), None)
            };
            return Ok(Some(Target {
                os: os.to_owned(),
                arch: arch.to_owned(),
                backend: backend.to_owned(),
                major,
            }));
        }
    }
    Ok(None)
}

fn catalog_target(value: &Json) -> Option<Target> {
    let platform = value.get("platform")?;
    let backend = value.get("backend")?;
    let os = platform.get("os")?.as_str()?;
    let arch = platform.get("arch")?.as_str()?;
    let kind = backend.get("kind")?.as_str()?;
    let major = if kind == "cuda" {
        match backend
            .get("cuda")
            .and_then(|cuda| cuda.get("toolkit_major"))
        {
            Some(Json::Number(number)) if number.is_i64() || number.is_u64() => {
                Some(number.to_string())
            }
            Some(Json::Bool(flag)) => Some(if *flag { "1" } else { "0" }.to_owned()),
            _ => None,
        }
    } else {
        None
    };
    Some(Target {
        os: os.to_owned(),
        arch: arch.to_owned(),
        backend: kind.to_owned(),
        major,
    })
}

pub(super) fn run(args: &[String]) -> CheckReport {
    let mut manifest_path = None;
    let mut labels = Vec::new();
    let mut assets = Vec::new();
    let mut index = 0;
    let mut after_separator = false;
    while let Some(arg) = args.get(index) {
        index += 1;
        if arg == "--" && !after_separator {
            after_separator = true;
            continue;
        }
        if !after_separator
            && (arg == "--manifest"
                || arg == "--required-target"
                || arg.starts_with("--manifest=")
                || arg.starts_with("--required-target="))
        {
            let (option, value) = if let Some((key, value)) = arg.split_once('=') {
                (key, value.to_owned())
            } else {
                let Some(next) = args.get(index).filter(|next| !next.starts_with('-')) else {
                    return usage_error(&format!("argument {arg}: expected one argument"));
                };
                index += 1;
                (arg.as_str(), next.clone())
            };
            if option == "--manifest" {
                manifest_path = Some(value);
            } else {
                labels.push(value);
            }
        } else if !after_separator && (arg == "-h" || arg == "--help") {
            return CheckReport::success(format!(
                "{USAGE}\nValidate release binary bundle targets against native-runtimes.json.\n"
            ));
        } else if !after_separator && arg.starts_with('-') {
            return usage_error(&format!("unrecognized arguments: {arg}"));
        } else {
            assets.push(arg.clone());
        }
    }
    let Some(path) = manifest_path else {
        return usage_error("the following arguments are required: --manifest");
    };
    if labels.is_empty() && assets.is_empty() {
        return CheckReport {
            stdout: String::new(),
            stderr:
                "release matrix error: assets are required unless --required-target is provided\n"
                    .to_owned(),
            code: 2,
        };
    }
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => return CheckReport::failure(String::new(), format!("{error}\n")),
    };
    let manifest = match Json::parse(&bytes) {
        Ok(value) => value,
        Err(error) => return CheckReport::failure(String::new(), format!("{error}\n")),
    };
    let mut required = BTreeSet::new();
    for label in &labels {
        match label_target(label) {
            Ok(target) => {
                required.insert(target);
            }
            Err(error) => {
                return CheckReport {
                    stdout: String::new(),
                    stderr: format!("release matrix error: {error}\n"),
                    code: 2,
                };
            }
        }
    }
    if required.is_empty() {
        for asset in assets {
            match asset_target(&asset) {
                Ok(Some(target)) => {
                    required.insert(target);
                }
                Ok(None) => {}
                Err(error) => {
                    return CheckReport::failure(String::new(), format!("ValueError: {error}\n"));
                }
            }
        }
    }
    let candidates: Vec<Target> = manifest
        .get("artifacts")
        .and_then(Json::as_array)
        .unwrap_or_default()
        .iter()
        .filter_map(catalog_target)
        .collect();
    let stderr: String = required
        .iter()
        .filter(|target| !candidates.iter().any(|candidate| target.covers(candidate)))
        .map(|target| {
            format!(
                "release matrix error: missing native runtime for binary target {}\n",
                target.label()
            )
        })
        .collect();
    if stderr.is_empty() {
        CheckReport::success("release native runtime matrix is complete\n".to_owned())
    } else {
        CheckReport::failure(String::new(), stderr)
    }
}

fn usage_error(message: &str) -> CheckReport {
    CheckReport {
        stdout: String::new(),
        stderr: format!("{USAGE}validate-release-native-runtime-matrix.py: error: {message}\n"),
        code: 2,
    }
}
