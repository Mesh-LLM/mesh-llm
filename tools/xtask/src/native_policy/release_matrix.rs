use crate::ci_plan::document::Json;
use crate::repository::check_report::CheckReport;
use crate::repository::text::repr;
use std::collections::BTreeSet;

const USAGE: &str = "usage: native release-matrix [-h] --manifest MANIFEST\n                                                 [--required-target REQUIRED_TARGET]\n                                                 [assets ...]\n";

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

/// GPU arches a release target exists for. The runtime resolver rejects a
/// runtime whose declared arches omit the host GPU, so a runtime missing one of
/// these would be unusable on exactly the hardware its lane was added for.
fn required_gpu_arches(target: &Target) -> &'static [&'static str] {
    match (
        target.os.as_str(),
        target.arch.as_str(),
        target.backend.as_str(),
        target.major.as_deref(),
    ) {
        ("windows", "x86_64", "cuda", Some("13")) => &["120"],
        _ => &[],
    }
}

fn gpu_arches(value: &Json) -> BTreeSet<&str> {
    let backend = value.get("backend");
    backend
        .and_then(|backend| backend.get("kind")?.as_str())
        .and_then(|kind| backend?.get(kind)?.get("gpu_arches")?.as_array())
        .map(|arches| arches.iter().filter_map(Json::as_str).collect())
        .unwrap_or_default()
}

fn matrix_violation(required: &Target, candidates: &[(Target, &Json)]) -> Option<String> {
    let matches: Vec<&Json> = candidates
        .iter()
        .filter(|(candidate, _)| required.covers(candidate))
        .map(|(_, artifact)| *artifact)
        .collect();
    if matches.is_empty() {
        return Some(format!(
            "release matrix error: missing native runtime for binary target {}\n",
            required.label()
        ));
    }
    let arches = required_gpu_arches(required);
    if matches.iter().any(|artifact| {
        let declared = gpu_arches(artifact);
        arches.iter().all(|arch| declared.contains(arch))
    }) {
        return None;
    }
    Some(format!(
        "release matrix error: native runtime for binary target {} does not declare required GPU arches {}\n",
        required.label(),
        arches.join(", ")
    ))
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
                    return CheckReport::failure(
                        String::new(),
                        format!("release matrix error: {error}\n"),
                    );
                }
            }
        }
    }
    let candidates: Vec<(Target, &Json)> = manifest
        .get("artifacts")
        .and_then(Json::as_array)
        .unwrap_or_default()
        .iter()
        .filter_map(|artifact| Some((catalog_target(artifact)?, artifact)))
        .collect();
    let stderr: String = required
        .iter()
        .filter_map(|target| matrix_violation(target, &candidates))
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
        stderr: format!("{USAGE}native release-matrix: error: {message}\n"),
        code: 2,
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    #[test]
    fn unsupported_asset_suffix_rejects_without_a_completion_claim_or_writes() {
        let scratch = tempfile::tempdir().unwrap();
        let manifest = scratch.path().join("native-runtimes.json");
        let original = b"{\"artifacts\":[]}";
        std::fs::write(&manifest, original).unwrap();
        let report = run(&[
            "--manifest".into(),
            manifest.to_str().unwrap().into(),
            "mesh-llm-v1-x86_64-unknown-linux-gnu-cuda-invalid.tar.gz".into(),
        ]);
        assert_eq!(report.code, 1);
        assert!(report.stdout.is_empty());
        assert!(report.stderr.contains("unsupported CUDA release suffix"));
        assert_eq!(std::fs::read(manifest).unwrap(), original);
        assert_eq!(std::fs::read_dir(scratch.path()).unwrap().count(), 1);
    }

    fn windows_cuda13(gpu_arches: &str) -> CheckReport {
        let scratch = tempfile::tempdir().unwrap();
        let manifest = scratch.path().join("native-runtimes.json");
        std::fs::write(
            &manifest,
            format!(
                "{{\"artifacts\":[{{\"platform\":{{\"os\":\"windows\",\"arch\":\"x86_64\"}},\"backend\":{{\"kind\":\"cuda\",\"cuda\":{{\"toolkit_major\":13,\"gpu_arches\":{gpu_arches}}}}}}}]}}"
            ),
        )
        .unwrap();
        run(&[
            "--manifest".into(),
            manifest.to_str().unwrap().into(),
            "mesh-llm-v1-x86_64-pc-windows-msvc-cuda-13.zip".into(),
        ])
    }

    #[test]
    fn windows_cuda13_runtime_must_declare_blackwell_arch() {
        assert_eq!(windows_cuda13("[\"89\",\"120\"]").code, 0);
        let missing = windows_cuda13("[\"89\"]");
        assert_eq!(missing.code, 1);
        assert_eq!(
            missing.stderr,
            "release matrix error: native runtime for binary target windows/x86_64/cuda13 does not declare required GPU arches 120\n"
        );
    }
}
