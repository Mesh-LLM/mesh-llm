use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, io::Write, path::Path};

#[derive(Deserialize, Serialize)]
struct Status {
    effective: Effective,
    configured: Configuration,
    #[serde(default)]
    inventory: Vec<Inventory>,
    #[serde(default)]
    activity: Activity,
    usage: DiskUsage,
    #[serde(flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}
#[derive(Deserialize, Serialize)]
struct Effective {
    state: String,
    reason: Option<serde_json::Value>,
    #[serde(flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}
#[derive(Deserialize, Serialize)]
struct Configuration {
    mode: String,
    budget_bytes: u64,
    minimum_free_bytes: u64,
    sources: BTreeMap<String, String>,
    #[serde(flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}
#[derive(Deserialize, Serialize)]
struct Inventory {
    payload_kind: Option<String>,
    #[serde(flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}
#[derive(Default, Deserialize, Serialize)]
struct Activity {
    #[serde(default)]
    writes: u64,
    #[serde(default)]
    fills: u64,
    #[serde(flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}
#[derive(Deserialize, Serialize)]
struct DiskUsage {
    used_bytes: u64,
    #[serde(flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}
#[derive(Deserialize)]
struct Response {
    choices: Vec<Choice>,
    usage: ResponseUsage,
}
#[derive(Deserialize)]
struct Choice {
    message: Message,
}
#[derive(Deserialize)]
struct Message {
    content: String,
}
#[derive(Deserialize)]
struct ResponseUsage {
    prompt_tokens: u64,
    prompt_tokens_details: Details,
}
#[derive(Deserialize)]
struct Details {
    cached_tokens: u64,
}

fn statuses(root: &Path, phase: &str) -> DynResult<BTreeMap<String, Status>> {
    ["seed", "worker"]
        .into_iter()
        .map(|node| {
            Ok((
                node.to_owned(),
                serde_json::from_slice(&fs::read(root.join(format!("{phase}-{node}.json")))?)?,
            ))
        })
        .collect()
}
fn active(status: &Status) -> bool {
    status.effective.state == "active"
        && status.effective.reason.is_none()
        && status.configured.mode == "fixed"
        && status.configured.budget_bytes == 2 * 1024 * 1024 * 1024
        && status.configured.minimum_free_bytes == 1024 * 1024 * 1024
        && ["mode", "budget", "directory", "minimum_free"]
            .iter()
            .all(|field| {
                status
                    .configured
                    .sources
                    .get(*field)
                    .is_some_and(|source| source == "cli")
            })
}
fn total(statuses: &BTreeMap<String, Status>, value: impl Fn(&Status) -> u64) -> DynResult<u64> {
    statuses.values().try_fold(0_u64, |sum, status| {
        sum.checked_add(value(status))
            .ok_or_else(|| "durable metric overflow".into())
    })
}
fn populated(statuses: &BTreeMap<String, Status>) -> DynResult<bool> {
    Ok(statuses
        .values()
        .all(|status| status.effective.state == "active")
        && statuses.values().any(|status| !status.inventory.is_empty())
        && total(statuses, |status| status.activity.writes)? > 0)
}
fn record(args: &[String]) -> DynResult<()> {
    let [
        records,
        label,
        model,
        artifact,
        digest,
        kind,
        seed_root,
        worker_root,
        directory,
    ] = args
    else {
        return Err("durable-record requires nine arguments".into());
    };
    let root = Path::new(directory);
    let before = statuses(root, "before")?;
    let restart = statuses(root, "restart-before")?;
    let after = statuses(root, "after")?;
    let cleared = statuses(root, "cleared")?;
    for states in [&before, &restart, &after, &cleared] {
        if !states.values().all(active) {
            return Err("durable tier configuration or source changed".into());
        }
    }
    if !before.values().any(|status| !status.inventory.is_empty())
        || !restart.values().any(|status| !status.inventory.is_empty())
    {
        return Err("durable inventory missing across restart".into());
    }
    let fills = total(&after, |status| status.activity.fills)?;
    if total(&before, |status| status.activity.writes)? == 0
        || total(&restart, |status| status.activity.fills)? != 0
        || fills == 0
    {
        return Err("restart did not prove a fresh durable L3 fill".into());
    }
    let warm: Response = serde_json::from_slice(&fs::read(root.join("warm-response.json"))?)?;
    let restored: Response =
        serde_json::from_slice(&fs::read(root.join("restored-response.json"))?)?;
    if restored.usage.prompt_tokens_details.cached_tokens == 0
        || warm.choices.first().map(|choice| &choice.message.content)
            != restored
                .choices
                .first()
                .map(|choice| &choice.message.content)
        || restored.choices.is_empty()
    {
        return Err("restored tokens or exact output comparison failed".into());
    }
    let kinds = restart
        .values()
        .flat_map(|status| {
            status
                .inventory
                .iter()
                .filter_map(|entry| entry.payload_kind.clone())
        })
        .collect::<std::collections::BTreeSet<_>>();
    if kinds.is_empty() || (!kind.is_empty() && !kinds.contains(kind)) {
        return Err("durable payload kind missing".into());
    }
    if cleared
        .values()
        .any(|status| !status.inventory.is_empty() || status.usage.used_bytes != 0)
    {
        return Err("clear left managed bytes or inventory".into());
    }
    let record = serde_json::json!({
        "model":{"label":label,"artifact_id":artifact,"sha256":digest,"path":model},
        "configuration":{"source":"cli","mode":"fixed","budget_bytes":2_u64*1024*1024*1024,"minimum_free_bytes":1024_u64*1024*1024,"roots":{"seed":seed_root,"worker":worker_root}},
        "cache_root_lifecycle":"preserved","process_boundary":true,"payload_kinds":kinds,
        "statuses":{"before_stop":before,"after_restart_before_request":restart,"after_restore":after,"after_clear":cleared},
        "restored_prompt_tokens":restored.usage.prompt_tokens,"restored_cached_tokens":restored.usage.prompt_tokens_details.cached_tokens,
        "l3_fill_count":fills,"exact_output_match":true,"status_command":"mesh-llm kv-cache status --json","clear_command":"mesh-llm kv-cache clear --yes --json"
    });
    let mut handle = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(records)?;
    serde_json::to_writer(&mut handle, &record)?;
    handle.write_all(b"\n")?;
    Ok(())
}

pub(super) fn run(args: &[String]) -> DynResult<String> {
    match args {
        [verb, prefix] if verb == "durable-ready" => {
            let seed: Status = serde_json::from_slice(&fs::read(format!("{prefix}-seed.json"))?)?;
            let worker: Status =
                serde_json::from_slice(&fs::read(format!("{prefix}-worker.json"))?)?;
            if !populated(&BTreeMap::from([
                ("seed".into(), seed),
                ("worker".into(), worker),
            ]))? {
                return Err("durable population not ready".into());
            }
        }
        [verb, rest @ ..] if verb == "durable-record" => record(rest)?,
        [verb, records, output, labels] if verb == "durable-evidence" => {
            #[derive(Deserialize)]
            struct Record {
                model: Model,
            }
            #[derive(Deserialize)]
            struct Model {
                label: String,
            }
            let records = fs::read_to_string(records)?
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(serde_json::from_str::<serde_json::Value>)
                .collect::<Result<Vec<_>, _>>()?;
            let observed = records
                .iter()
                .map(|record| {
                    serde_json::from_value::<Record>(record.clone())
                        .map(|record| record.model.label)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if observed != labels.split(',').collect::<Vec<_>>() {
                return Err("durable evidence model order differs".into());
            }
            let mut bytes = serde_json::to_vec_pretty(
                &serde_json::json!({"schema_version":1,"kind":"mesh-llm-durable-l3-restart","status":"passed","models":records}),
            )?;
            bytes.push(b'\n');
            super::write(output, &bytes)?;
        }
        _ => return Err("unsupported durable probe arguments".into()),
    }
    Ok(String::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(fills: u64, inventory: bool) -> serde_json::Value {
        serde_json::json!({"effective":{"state":"active","reason":null},"configured":{"mode":"fixed","budget_bytes":2147483648_u64,"minimum_free_bytes":1073741824_u64,"sources":{"mode":"cli","budget":"cli","directory":"cli","minimum_free":"cli"}},"inventory":if inventory {vec![serde_json::json!({"payload_kind":"kv-recurrent"})]} else {vec![]},"activity":{"writes":1,"fills":fills},"usage":{"used_bytes":if inventory {1} else {0}}})
    }
    #[test]
    fn restart_requires_fresh_fill_exact_output_and_clear() -> DynResult<()> {
        let root = tempfile::tempdir()?;
        for phase in ["before", "restart-before", "after", "cleared"] {
            for node in ["seed", "worker"] {
                fs::write(
                    root.path().join(format!("{phase}-{node}.json")),
                    serde_json::to_vec(&fixture(u64::from(phase == "after"), phase != "cleared"))?,
                )?;
            }
        }
        let response = serde_json::json!({"choices":[{"message":{"content":"exact"}}],"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":64}}});
        for name in ["warm-response.json", "restored-response.json"] {
            fs::write(root.path().join(name), serde_json::to_vec(&response)?)?;
        }
        let records = root.path().join("records.jsonl");
        let args = [
            records.to_string_lossy().into_owned(),
            "recurrent".into(),
            "model.gguf".into(),
            "artifact".into(),
            "a".repeat(64),
            "kv-recurrent".into(),
            "seed-root".into(),
            "worker-root".into(),
            root.path().to_string_lossy().into_owned(),
        ];
        record(&args)?;
        let evidence: serde_json::Value = serde_json::from_slice(&fs::read(&records)?)?;
        assert_eq!(evidence["l3_fill_count"], 2);
        fs::write(
            root.path().join("restart-before-seed.json"),
            serde_json::to_vec(&fixture(1, true))?,
        )?;
        assert!(record(&args).is_err());
        fs::write(
            root.path().join("restart-before-seed.json"),
            serde_json::to_vec(&fixture(0, true))?,
        )?;
        fs::write(
            root.path().join("cleared-worker.json"),
            serde_json::to_vec(&fixture(0, true))?,
        )?;
        assert!(record(&args).is_err());
        fs::write(
            root.path().join("cleared-worker.json"),
            serde_json::to_vec(&fixture(0, false))?,
        )?;
        let mut changed = response.clone();
        changed["choices"][0]["message"]["content"] = serde_json::json!("different");
        fs::write(
            root.path().join("restored-response.json"),
            serde_json::to_vec(&changed)?,
        )?;
        assert!(record(&args).is_err());
        Ok(())
    }
}
