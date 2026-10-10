//! Project deployment controls onto one selected, already admitted WAN stage.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};
#[path = "wan_stage_deployment/tests.rs"]
#[cfg(test)]
mod tests;
const INPUT_LIMIT: u64 = 16 * 1024 * 1024;
pub(crate) const USAGE: &str = "automation wan-stage-deployment --input PLAN --output CONFIG --stage-index N --stage-count N [--bind-port N] [--run-id ID] [--topology-id ID] [--n-batch N] [--n-ubatch N] [--cache-type-k TYPE] [--cache-type-v TYPE] [--flash-attn-type TYPE]";
struct Controls {
    index: u32,
    count: u32,
    port: u16,
    run_id: String,
    topology_id: String,
    batch: Option<u32>,
    ubatch: Option<u32>,
    cache_k: String,
    cache_v: String,
    flash: String,
}
fn scalar(values: &BTreeMap<&str, &str>, key: &str, default: &str) -> DynResult<String> {
    let value = values.get(key).copied().unwrap_or(default);
    if value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control) {
        return Err(format!("invalid WAN deployment {key}").into());
    }
    Ok(value.into())
}
fn number<T: std::str::FromStr>(
    values: &BTreeMap<&str, &str>,
    key: &str,
    default: Option<&str>,
) -> DynResult<T> {
    values
        .get(key)
        .copied()
        .or(default)
        .ok_or_else(|| format!("missing {key}"))?
        .parse()
        .map_err(|_| format!("invalid {key}").into())
}
fn optional_number(values: &BTreeMap<&str, &str>, key: &str) -> DynResult<Option<u32>> {
    values
        .get(key)
        .map(|_| number(values, key, None))
        .transpose()
}
fn parse(args: &[String]) -> DynResult<(PathBuf, PathBuf, Controls)> {
    let (pairs, remainder) = args.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(USAGE.into());
    }
    let allowed = [
        "--input",
        "--output",
        "--stage-index",
        "--stage-count",
        "--bind-port",
        "--run-id",
        "--topology-id",
        "--n-batch",
        "--n-ubatch",
        "--cache-type-k",
        "--cache-type-v",
        "--flash-attn-type",
    ];
    let mut values = BTreeMap::new();
    for pair in pairs {
        if !allowed.contains(&pair[0].as_str())
            || values.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err(USAGE.into());
        }
    }
    let index = number(&values, "--stage-index", None)?;
    let count = number(&values, "--stage-count", None)?;
    let port = number(&values, "--bind-port", Some("19000"))?;
    if !(2..=10000).contains(&count) || index >= count || port == 0 {
        return Err("invalid WAN stage placement".into());
    }
    let flash = scalar(&values, "--flash-attn-type", "disabled")?;
    if !["auto", "disabled", "enabled"].contains(&flash.as_str()) {
        return Err("invalid WAN flash attention control".into());
    }
    let controls = Controls {
        index,
        count,
        port,
        flash,
        run_id: scalar(&values, "--run-id", "skippy-docker-wan")?,
        topology_id: scalar(&values, "--topology-id", "docker-wan-four-stage")?,
        batch: optional_number(&values, "--n-batch")?,
        ubatch: optional_number(&values, "--n-ubatch")?,
        cache_k: scalar(&values, "--cache-type-k", "f16")?,
        cache_v: scalar(&values, "--cache-type-v", "f16")?,
    };
    let input = PathBuf::from(*values.get("--input").ok_or("missing --input")?);
    let output = PathBuf::from(*values.get("--output").ok_or("missing --output")?);
    Ok((input, output, controls))
}
fn peer(index: u32, port: u16) -> Value {
    json!({"stage_id":format!("stage-{index}"),"stage_index":index,"endpoint":format!("tcp://stage{index}:{port}")})
}
fn project(mut plan: Value, controls: &Controls) -> DynResult<Value> {
    let object = plan
        .as_object_mut()
        .ok_or("WAN selected plan must be an object")?;
    if object.get("stage_index").and_then(Value::as_u64) != Some(u64::from(controls.index))
        || object.get("stage_id").and_then(Value::as_str)
            != Some(format!("stage-{}", controls.index).as_str())
    {
        return Err("WAN selected plan stage identity mismatch".into());
    }
    for (key, value) in [
        ("run_id", json!(controls.run_id)),
        ("topology_id", json!(controls.topology_id)),
        ("bind_addr", json!(format!("0.0.0.0:{}", controls.port))),
        (
            "upstream",
            controls
                .index
                .checked_sub(1)
                .map_or(Value::Null, |i| peer(i, controls.port)),
        ),
        (
            "downstream",
            if controls.index + 1 < controls.count {
                peer(controls.index + 1, controls.port)
            } else {
                Value::Null
            },
        ),
        ("n_batch", json!(controls.batch)),
        ("n_ubatch", json!(controls.ubatch)),
        ("cache_type_k", json!(controls.cache_k)),
        ("cache_type_v", json!(controls.cache_v)),
        ("flash_attn_type", json!(controls.flash)),
    ] {
        object.insert(key.into(), value);
    }
    Ok(plan)
}
fn read_plan(path: &Path) -> DynResult<Value> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > INPUT_LIMIT {
        return Err("WAN selected plan must be a bounded regular file".into());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err("WAN selected plan reparse point refused".into());
        }
    }
    let mut bytes = Vec::new();
    file.take(INPUT_LIMIT + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > INPUT_LIMIT {
        return Err("WAN selected plan exceeds input bound".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn publish(output: &Path, bytes: &[u8]) -> DynResult<()> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let previous = match fs::symlink_metadata(output) {
        Ok(metadata) if metadata.is_file() => {
            #[cfg(windows)]
            {
                use std::os::windows::fs::MetadataExt;
                if metadata.file_attributes() & 0x400 != 0 {
                    return Err("WAN generated config reparse point refused".into());
                }
            }
            Some(metadata.permissions())
        }
        Ok(_) => {
            return Err(
                "WAN generated config must be a regular file, not a symlink or directory".into(),
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let mut staged = tempfile::NamedTempFile::new_in(parent)?;
    staged.write_all(bytes)?;
    if let Some(permissions) = previous {
        staged.as_file().set_permissions(permissions)?;
    }
    staged.flush()?;
    staged.persist(output).map_err(|error| error.error)?;
    Ok(())
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        writeln!(std::io::stdout().lock(), "{USAGE}")?;
        return Ok(());
    }
    let (input, output, controls) = parse(args)?;
    let projected = project(read_plan(&input)?, &controls)?;
    let mut bytes = serde_json::to_vec_pretty(&projected)?;
    bytes.push(b'\n');
    publish(&output, &bytes)
}
