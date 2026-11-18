use super::{runtime_seed_io::save, runtime_seed_stats::Snapshot};
use crate::command::DynResult;
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Default, Serialize)]
pub(super) struct Inventory {
    pub objects: BTreeMap<String, u64>,
    pub bytes: u64,
}

pub(super) fn inventory(directory: &Path) -> DynResult<Inventory> {
    let mut result = Inventory::default();
    visit(directory, directory, &mut result)?;
    Ok(result)
}

fn visit(root: &Path, directory: &Path, result: &mut Inventory) -> DynResult<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err("cache inventory contains symlink".into());
        }
        if kind.is_dir() {
            visit(root, &entry.path(), result)?;
        } else if kind.is_file() {
            let size = entry.metadata()?.len();
            let key = entry
                .path()
                .strip_prefix(root)?
                .to_string_lossy()
                .into_owned();
            result.bytes = result
                .bytes
                .checked_add(size)
                .ok_or("inventory size overflow")?;
            result.objects.insert(key, size);
        } else {
            return Err("cache inventory contains nonregular object".into());
        }
    }
    Ok(())
}

pub(super) fn change(before: &Inventory, after: &Inventory) -> serde_json::Value {
    let removed: BTreeMap<_, _> = before
        .objects
        .iter()
        .filter(|(key, _)| !after.objects.contains_key(*key))
        .collect();
    let added: BTreeMap<_, _> = after
        .objects
        .iter()
        .filter(|(key, _)| !before.objects.contains_key(*key))
        .collect();
    serde_json::json!({"removed_objects":removed,"removed_bytes":removed.values().map(|size| **size).sum::<u64>(),"added_objects":added,"added_bytes":added.values().map(|size| **size).sum::<u64>(),"interpretation":"removed disk object keys establish disappearance; with no deletion by this command they indicate cache eviction/replacement, not an inferred counter"})
}

pub(super) fn measurement(bytes: &[u8]) -> DynResult<serde_json::Value> {
    let snapshot = Snapshot::parse(bytes)?;
    let measurement = snapshot.measurement()?;
    if snapshot.stats.scalars["cache_timeouts"] != 0 {
        return Err("cache timeouts: inconclusive".into());
    }
    Ok(
        serde_json::json!({"hit_rate":measurement.hit_rate,"hits":snapshot.stats.cache_hits.counts,"misses":snapshot.stats.cache_misses.counts,"compile_requests":snapshot.stats.scalars["compile_requests"],"compile_fails":snapshot.stats.scalars["compile_fails"],"interpretation":"phase counters reset before command; hit rate is hits/(hits+misses), not hits/compile_requests"}),
    )
}

pub(super) fn host(output: &Path) -> DynResult<()> {
    for (source, name) in [
        ("/proc/cpuinfo", "cpuinfo.txt"),
        ("/proc/meminfo", "meminfo.txt"),
        ("/proc/self/cgroup", "cgroup.txt"),
    ] {
        fs::write(output.join(name), fs::read(source)?)?;
    }
    let mut observations = BTreeMap::new();
    for name in ["memory.peak", "memory.max", "memory.current", "cpu.max"] {
        let path = Path::new("/sys/fs/cgroup").join(name);
        match fs::read_to_string(path) {
            Ok(text) => {
                observations.insert(name, Some(text));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                observations.insert(name, None);
            }
            Err(error) => return Err(error.into()),
        }
    }
    save(
        &output.join("resource-observation.json"),
        &serde_json::json!({"cgroup":observations,"note":"container-wide observation; not attributable per phase, not reset by the experiment"}),
    )
}
