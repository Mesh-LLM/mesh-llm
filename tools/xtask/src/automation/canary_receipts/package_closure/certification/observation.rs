use crate::{
    command::DynResult,
    process::{self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
};
use std::{collections::BTreeMap, path::Path, time::Duration};

#[derive(Clone, Copy, Debug)]
pub(super) struct Host {
    pub(super) total: u64,
    pub(super) available: u64,
}

pub(super) fn available(text: &str) -> DynResult<u64> {
    let after = text
        .split_once("page size of ")
        .ok_or("vm_stat lacks page size")?
        .1;
    let size: u64 = after
        .split_once(" bytes")
        .ok_or("vm_stat lacks page extent")?
        .0
        .parse()?;
    if size == 0 {
        return Err("invalid vm_stat page size".into());
    }
    let mut pages = BTreeMap::new();
    for line in text.lines() {
        if let Some((key, value)) = line.split_once(':')
            && ["Pages free", "Pages inactive", "Pages speculative"].contains(&key)
            && pages
                .insert(
                    key,
                    value
                        .trim()
                        .strip_suffix('.')
                        .ok_or("invalid vm_stat page count")?
                        .parse::<u64>()?,
                )
                .is_some()
        {
            return Err("duplicate vm_stat available field".into());
        }
    }
    let mut count = 0_u64;
    for key in ["Pages free", "Pages inactive", "Pages speculative"] {
        count = count
            .checked_add(*pages.get(key).ok_or("vm_stat available field missing")?)
            .ok_or("vm_stat page overflow")?;
    }
    count
        .checked_mul(size)
        .ok_or_else(|| "vm_stat byte overflow".into())
}
pub(super) fn admission(host: Host, peak: u64) -> DynResult<u64> {
    if host.total == 0 || host.available > host.total || peak == 0 {
        return Err("invalid host/family memory observation".into());
    }
    let reserve = host
        .total
        .checked_mul(10)
        .and_then(|value| value.checked_add(99))
        .ok_or("memory reserve overflow")?
        / 100;
    if peak > host.total.saturating_sub(reserve) {
        return Err("assigned host is too small after 10% reserve".into());
    }
    if peak > host.available.saturating_sub(reserve) {
        return Err("insufficient available memory after 10% reserve".into());
    }
    Ok(reserve)
}
pub(super) fn observe(root: &Path, cancel: &Cancellation) -> DynResult<Host> {
    if !cfg!(target_os = "macos") {
        return Err("family memory guard requires macOS".into());
    }
    let total = probe(root, "/usr/sbin/sysctl", &["-n", "hw.memsize"], cancel)?
        .trim()
        .parse()?;
    let available = available(&probe(root, "/usr/bin/vm_stat", &[], cancel)?)?;
    if total == 0 || available > total {
        return Err("invalid host memory observation".into());
    }
    Ok(Host { total, available })
}
fn probe(root: &Path, executable: &str, args: &[&str], cancel: &Cancellation) -> DynResult<String> {
    let report = process::supervise(
        &ProcessSpec {
            executable: executable.into(),
            arguments: args
                .iter()
                .map(|arg| Value::Public((*arg).into()))
                .collect(),
            cwd: root.to_owned(),
            environment: BTreeMap::from([
                (
                    "PATH".into(),
                    Value::Public("/usr/bin:/bin:/usr/sbin:/sbin".into()),
                ),
                ("LC_ALL".into(), Value::Public("C".into())),
            ]),
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        OutputFiles::default(),
    )?;
    if !report.success() || report.stdout.truncated || report.stderr.truncated {
        return Err(format!(
            "host memory probe failed: {:?}, {:?}",
            report.outcome, report.status
        )
        .into());
    }
    Ok(String::from_utf8(report.stdout.bytes_retained)?)
}
