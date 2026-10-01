use super::runtime_seed_io::{capture, empty, environment, files, monotonic, read, save};
use super::runtime_seed_preflight;
use super::runtime_seed_stats::Snapshot;
use super::runtime_seed_types::{Arm, BUILD, Context, ResultEvidence, Timestamp, elapsed};
use crate::{command::DynResult, repository::check_report::CheckReport};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

pub(super) fn run(args: &[String]) -> CheckReport {
    if args == ["--help"] {
        return CheckReport::success("usage: ci-ops runtime-seed {preflight|restore_start|restored|start|finish|summarize} DIRECTORY\n".into());
    }
    let [command, directory] = args else {
        return CheckReport::failure(String::new(), "usage: ci-ops runtime-seed {preflight|restore_start|restored|start|finish|summarize} DIRECTORY\n".into());
    };
    let directory = Path::new(directory);
    if command == "preflight" && (directory.exists() || directory.is_symlink()) {
        return CheckReport::failure(
            String::new(),
            "runtime seed canary: inconclusive: fresh evidence directory required\n".into(),
        );
    }
    let result = execute(command, directory);
    match result {
        Ok(code) => CheckReport {
            stdout: String::new(),
            stderr: String::new(),
            code,
        },
        Err(error) => {
            let failure = serde_json::json!({"classification":"inconclusive", "phase":command, "reason":error.to_string(), "eligibility_changed":false});
            let destination = match command.as_str() {
                "preflight" if directory.is_dir() => Some("preflight-failure.json"),
                "summarize" => Some("summary.json"),
                _ => None,
            };
            if let Some(name) = destination {
                let retained = fs::create_dir_all(directory)
                    .map_err(Into::into)
                    .and_then(|()| save(&directory.join(name), &failure));
                if let Err(retention) = retained {
                    return CheckReport::failure(
                        String::new(),
                        format!(
                            "runtime seed canary: {error}; evidence retention failed: {retention}\n"
                        ),
                    );
                }
            }
            CheckReport::failure(
                String::new(),
                format!("runtime seed canary: inconclusive: {error}\n"),
            )
        }
    }
}

fn execute(command: &str, directory: &Path) -> DynResult<i32> {
    match command {
        "preflight" => runtime_seed_preflight::run(directory)?,
        "restore_start" => save(
            &directory.join("restore-start.json"),
            &Timestamp {
                monotonic: monotonic()?,
            },
        )?,
        "restored" => restored(directory)?,
        "start" => {
            capture("sccache", &["--zero-stats"])?;
            let initial = snapshot()?;
            if initial.stats.scalars["compile_requests"] != 0 {
                return Err("counters did not reset".into());
            }
            save(&directory.join("initial-stats.json"), &initial)?;
            save(
                &directory.join("start.json"),
                &Timestamp {
                    monotonic: monotonic()?,
                },
            )?;
        }
        "finish" => return finish(directory),
        "summarize" => super::runtime_seed_summary::run(directory)?,
        _ => return Err("unknown runtime-seed operation".into()),
    }
    Ok(0)
}

fn snapshot() -> DynResult<Snapshot> {
    Snapshot::parse(&capture(
        "sccache",
        &["--show-stats", "--stats-format=json"],
    )?)
}

fn restored(directory: &Path) -> DynResult<()> {
    let mut context: Context = read(&directory.join("context.json"))?;
    let start: Timestamp = read(&directory.join("restore-start.json"))?;
    context.restore_seconds = Some(elapsed(monotonic()?, start.monotonic)?);
    let cache = runtime_seed_preflight::fetch_cache(&context.cache.key)?;
    if !same_restore(&cache, &context.cache) {
        return Err("restored cache identity changed".into());
    }
    let hit = std::env::var("CANARY_CACHE_HIT").is_ok_and(|value| value == "true");
    match context.arm {
        Arm::Warm if !hit => return Err("warm restore missed: inconclusive".into()),
        Arm::Cold if !empty(Path::new(&environment("SCCACHE_DIR")?))? => {
            return Err("cold cache populated before build".into());
        }
        Arm::Warm | Arm::Cold => (),
    }
    context.cache_after_restore = Some(cache);
    context.cache_hit = Some(hit);
    save(&directory.join("context.json"), &context)
}

fn same_restore(
    actual: &super::runtime_seed_types::Cache,
    expected: &super::runtime_seed_types::Cache,
) -> bool {
    actual.id == expected.id && actual.version == expected.version
}

fn finish(directory: &Path) -> DynResult<i32> {
    let context: Context = read(&directory.join("context.json"))?;
    let now = monotonic()?;
    let start: Timestamp = read(&directory.join("start.json"))?;
    let restore: Timestamp = read(&directory.join("restore-start.json"))?;
    let raw = snapshot()?;
    save(&directory.join("raw-stats.json"), &raw)?;
    let measurement = raw.measurement()?;
    if environment("CANARY_BUILD_OUTCOME")? != "success" {
        return Err("package verification failed".into());
    }
    let cmake = fs::read(Path::new(BUILD).join("CMakeCache.txt"))?;
    let text = std::str::from_utf8(&cmake)?;
    for compiler in ["C", "CXX"] {
        let prefix = format!("CMAKE_{compiler}_COMPILER_LAUNCHER:");
        if !text
            .lines()
            .filter(|line| line.starts_with(&prefix))
            .any(|line| {
                line.split_once('=').is_some_and(|(_, launcher)| {
                    Path::new(launcher)
                        .file_name()
                        .is_some_and(|name| name == "sccache")
                })
            })
        {
            return Err("native compiler launcher missing".into());
        }
    }
    save(
        &directory.join("native-build-evidence.json"),
        &serde_json::json!({"cmake_cache_sha256":hex::encode(Sha256::digest(&cmake)), "compiler_launchers":"sccache", "package_action_verified":true}),
    )?;
    let mut hashes = BTreeMap::new();
    for path in files(Path::new("runtime-input"), None)? {
        if path
            .extension()
            .is_some_and(|extension| extension == "json" || extension == "sha256")
        {
            hashes.insert(
                path.to_string_lossy().into_owned(),
                hex::encode(Sha256::digest(fs::read(&path)?)),
            );
        }
    }
    if hashes.is_empty() {
        return Err("missing verified package manifests".into());
    }
    let floor = context.arm == Arm::Cold || measurement.hit_rate >= 0.01;
    let result = ResultEvidence {
        context,
        action_seconds: elapsed(now, start.monotonic)?,
        total_seconds: elapsed(now, restore.monotonic)?,
        native_preparation_and_build_seconds: None,
        packaging_seconds: None,
        phase_timing_note: "derive optional observed split from timestamped job log".into(),
        classification: if floor {
            "measured"
        } else {
            "warm-floor-failure"
        }
        .into(),
        warm_floor_passed: floor,
        hit_rate: measurement.hit_rate,
        native_hits: measurement.native_hits,
        native_cacheable_requests: measurement.native_requests,
        assembler_hits: raw
            .stats
            .cache_hits
            .counts
            .get("Assembler")
            .copied()
            .unwrap_or(0),
        assembler_misses: raw
            .stats
            .cache_misses
            .counts
            .get("Assembler")
            .copied()
            .unwrap_or(0),
        language_hits: raw.stats.cache_hits.counts,
        language_misses: raw.stats.cache_misses.counts,
        manifest_and_checksum_hashes: hashes,
        eligibility_changed: false,
        verified: true,
    };
    save(&directory.join("result.json"), &result)?;
    Ok(i32::from(!floor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_identity_rejects_changed_id_or_version() {
        let cache = super::super::runtime_seed_types::Cache {
            id: 123,
            key: "fixture".into(),
            version: "a".repeat(64),
            reference: "refs/heads/main".into(),
            size_in_bytes: 1024,
            extra: BTreeMap::new(),
        };
        assert!(same_restore(&cache, &cache));
        let mut changed = cache.clone();
        changed.id = 456;
        assert!(!same_restore(&changed, &cache));
        changed.id = cache.id;
        changed.version = "b".repeat(64);
        assert!(!same_restore(&changed, &cache));
    }
}
