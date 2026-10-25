use super::runtime_seed_io::{capture, empty, environment, save};
use super::runtime_seed_types::{Arm, BUILD, Cache, Context, EPOCH, IMAGE, PREFIX, hexadecimal};
use crate::command::DynResult;
use serde::Deserialize;
use std::{collections::BTreeMap, fs, io::Write, path::Path};

#[derive(Deserialize)]
struct Listing {
    total_count: usize,
    actions_caches: Vec<Cache>,
}

pub(super) fn fetch_cache(key: &str) -> DynResult<Cache> {
    if !key
        .strip_prefix(PREFIX)
        .is_some_and(|hash| hexadecimal(hash, 64))
    {
        return Err("source seed recipe hash changed".into());
    }
    let endpoint = format!(
        "https://api.github.com/repos/Mesh-LLM/mesh-llm/actions/caches?key={key}&per_page=100"
    );
    let authorization = format!("Authorization: Bearer {}", environment("GH_TOKEN")?);
    let listing: Listing = serde_json::from_slice(&capture(
        "curl",
        &[
            "--fail",
            "--silent",
            "--max-time",
            "30",
            "--header",
            &authorization,
            "--header",
            "Accept: application/vnd.github+json",
            &endpoint,
        ],
    )?)?;
    admit_cache(listing, key)
}

fn admit_cache(listing: Listing, key: &str) -> DynResult<Cache> {
    if listing.total_count > 100 {
        return Err("cache listing pagination requires review".into());
    }
    let mut matches = listing
        .actions_caches
        .into_iter()
        .filter(|item| item.key == key);
    let cache = matches
        .next()
        .ok_or("missing seed or branch-shadow cache")?;
    if matches.next().is_some() {
        return Err("missing seed or branch-shadow cache".into());
    }
    cache.validate()?;
    Ok(cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_seed_rejects_missing_shadow_and_pagination() {
        let cache = || Cache {
            id: 123,
            key: "fixture".into(),
            version: "a".repeat(64),
            reference: "refs/heads/main".into(),
            size_in_bytes: 1024,
            extra: BTreeMap::new(),
        };
        assert!(
            admit_cache(
                Listing {
                    total_count: 1,
                    actions_caches: vec![cache()]
                },
                "fixture"
            )
            .is_ok()
        );
        for listing in [
            Listing {
                total_count: 0,
                actions_caches: vec![],
            },
            Listing {
                total_count: 2,
                actions_caches: vec![cache(), cache()],
            },
            Listing {
                total_count: 101,
                actions_caches: vec![cache()],
            },
        ] {
            assert!(admit_cache(listing, "fixture").is_err());
        }
    }
}

pub(super) fn run(directory: &Path) -> DynResult<()> {
    if directory.exists() || directory.is_symlink() {
        return Err("fresh evidence directory required".into());
    }
    fs::create_dir(directory)?;
    for (name, value) in [
        ("GITHUB_EVENT_NAME", "workflow_dispatch"),
        ("RUNNER_ENVIRONMENT", "github-hosted"),
        ("RUNNER_ARCH", "X64"),
        ("LLAMA_STAGE_BUILD_DIR", BUILD),
        ("RUSTC_WRAPPER", "sccache"),
        ("MESH_LLM_REQUIRE_SCCACHE", "1"),
        ("CARGO_INCREMENTAL", "0"),
        ("CACHE_NAMESPACE", "mesh-llm"),
        ("SCCACHE_GHA_ENABLED", "false"),
        ("SCCACHE_MULTILEVEL_CHAIN", "disk"),
        ("SCCACHE_CACHE_SIZE", "2G"),
        ("LLAMA_STAGE_BACKEND", "cpu"),
    ] {
        if environment(name)? != value {
            return Err(format!("canary environment drift: {name}").into());
        }
    }
    for name in ["LLAMA_STAGE_USE_SCCACHE", "SKIPPY_USE_SCCACHE"] {
        if std::env::var(name).unwrap_or_else(|_| "1".into()) != "1" {
            return Err("compiler cache disabled".into());
        }
    }
    for name in [
        "MESH_NATIVE_RUNTIME_MODEL_PACKAGE_TOOL",
        "CARGO_TARGET_DIR",
        "SKIPPY_LLAMA_BUILD_DIR",
        "LLAMA_STAGE_FORCE_BUILD",
        "SKIPPY_FORCE_LLAMA_BUILD",
        "MESH_NATIVE_RUNTIME_GPU_BENCHMARK_TOOL",
    ] {
        if std::env::var_os(name).is_some_and(|value| !value.is_empty()) {
            return Err(format!("unexpected override: {name}").into());
        }
    }
    for name in [BUILD, "target", "runtime-input"] {
        if Path::new(name).exists() || Path::new(name).is_symlink() {
            return Err(format!("existing native/Cargo/package output: {name}").into());
        }
    }
    if !empty(&Path::new(&environment("RUNNER_TEMP")?).join("mesh-llm-sccache"))? {
        return Err("initial compiler cache is not empty".into());
    }
    let source = String::from_utf8(capture("git", &["rev-parse", "HEAD"])?)?
        .trim()
        .to_owned();
    if source != environment("GITHUB_SHA")? {
        return Err("source differs from dispatch revision".into());
    }
    let run_id = positive_environment("GITHUB_RUN_ID")?;
    let run_attempt = positive_environment("GITHUB_RUN_ATTEMPT")?;
    let pair: u8 = environment("CANARY_PAIR")?.parse()?;
    if !(1..=3).contains(&pair) {
        return Err("invalid sample metadata".into());
    }
    let arm = match environment("CANARY_ARM")?.as_str() {
        "cold" => Arm::Cold,
        "warm" => Arm::Warm,
        _ => return Err("invalid sample metadata".into()),
    };
    let kernel = String::from_utf8(capture("uname", &["-srm"])?)?
        .trim()
        .to_owned();
    if !kernel.starts_with("Linux ") || !kernel.ends_with(" x86_64") {
        return Err("invalid kernel identity".into());
    }
    let key = environment("CANARY_KEY")?;
    let cache = fetch_cache(&key)?;
    let cpu: Cpu = serde_json::from_slice(&capture("lscpu", &["--json"])?)?;
    let fields: BTreeMap<_, _> = cpu
        .lscpu
        .into_iter()
        .map(|row| (row.field, row.data))
        .collect();
    let host_cpu = [
        "Architecture:",
        "CPU(s):",
        "Model name:",
        "Vendor ID:",
        "Thread(s) per core:",
    ]
    .into_iter()
    .map(|name| {
        fields
            .get(name)
            .filter(|value| !value.is_empty())
            .map(|value| (name.to_owned(), value.clone()))
            .ok_or("missing host CPU identity")
    })
    .collect::<Result<BTreeMap<_, _>, _>>()?;
    let context = Context {
        schema: 1,
        source,
        image: IMAGE.into(),
        epoch: EPOCH.into(),
        cache,
        pair,
        arm,
        run_id,
        run_attempt,
        build_dir: BUILD.into(),
        initial_outputs_absent: true,
        host_cpu,
        runner_class: "github-hosted/ubuntu-24.04/X64".into(),
        kernel,
        runner_image_os: std::env::var("ImageOS").ok(),
        runner_image_version: std::env::var("ImageVersion").ok(),
        restore_seconds: None,
        cache_after_restore: None,
        cache_hit: None,
    };
    save(&directory.join("context.json"), &context)?;
    writeln!(
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(environment("GITHUB_OUTPUT")?)?,
        "key={key}"
    )?;
    Ok(())
}

fn positive_environment(name: &str) -> DynResult<String> {
    let value = environment(name)?;
    if value.parse::<u64>().is_err() || value.parse::<u64>()? == 0 {
        return Err(format!("invalid run metadata: {name}").into());
    }
    Ok(value)
}

#[derive(Deserialize)]
struct Cpu {
    lscpu: Vec<CpuField>,
}
#[derive(Deserialize)]
struct CpuField {
    field: String,
    data: String,
}
