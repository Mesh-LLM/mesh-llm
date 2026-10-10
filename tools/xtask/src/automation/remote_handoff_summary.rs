//! Render supplied remote-handoff reports without running inference or contacting a receiver.
use crate::command::DynResult;
use serde::Deserialize;
use std::{fmt::Write as _, io::Read, path::Path};
const FILE_LIMIT: usize = 8 * 1024 * 1024;
const TOTAL_LIMIT: usize = 32 * 1024 * 1024;
const ROW_LIMIT: usize = 10_000;
const USAGE: &str = "automation remote-handoff-summary --reports-dir DIRECTORY; supplied send-*.json only, max10000 reports/8MiB each/32MiB total";
#[derive(Deserialize)]
struct Receiver {
    kv_attach_ms: f64,
}
#[derive(Deserialize)]
struct Report {
    prompt_token_count: u64,
    state_bytes: u64,
    transfer_gbps: f64,
    source_prefill_ms: f64,
    state_export_ms: f64,
    transfer_ms: f64,
    receiver: Receiver,
    ttft_disaggregated_ms: f64,
    ttft_local_ms: Option<f64>,
    ttft_speedup: Option<f64>,
    matches: bool,
}
impl Report {
    fn validate(&self) -> DynResult<()> {
        let values = [
            Some(self.transfer_gbps),
            Some(self.source_prefill_ms),
            Some(self.state_export_ms),
            Some(self.transfer_ms),
            Some(self.receiver.kv_attach_ms),
            Some(self.ttft_disaggregated_ms),
            self.ttft_local_ms,
            self.ttft_speedup,
        ];
        if values
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite() || v < 0.0)
        {
            return Err(
                "remote-handoff observations must be finite nonnegative measurements".into(),
            );
        }
        Ok(())
    }
}
fn reports(directory: &Path) -> DynResult<Vec<Report>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        if name
            .to_str()
            .is_some_and(|n| n.starts_with("send-") && n.ends_with(".json"))
        {
            if !entry.file_type()?.is_file() {
                return Err("handoff report must be a regular file, not a symlink".into());
            }
            paths.push(entry.path());
            if paths.len() > ROW_LIMIT {
                return Err("handoff report row bound exceeded".into());
            }
        }
    }
    paths.sort();
    let mut total = 0;
    let mut rows = Vec::new();
    for path in paths {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take((FILE_LIMIT + 1) as u64)
            .read_to_end(&mut bytes)?;
        total += bytes.len();
        if bytes.len() > FILE_LIMIT || total > TOTAL_LIMIT {
            return Err("handoff supplied report byte bound exceeded".into());
        }
        let report: Report = serde_json::from_slice(&bytes)
            .map_err(|_| "invalid typed handoff observation report")?;
        report.validate()?;
        rows.push(report);
    }
    // Stable prefix order retains filename order for duplicate prefixes.
    rows.sort_by_key(|r| r.prompt_token_count);
    Ok(rows)
}
fn render(rows: &[Report]) -> DynResult<String> {
    let mut out = format!(
        "{:>7} {:>7} {:>6} {:>8} {:>7} {:>7} {:>7} {:>8} {:>8} {:>7} match\n",
        "prefix",
        "MiB",
        "Gbps",
        "prefill",
        "export",
        "xfer",
        "attach",
        "ttft-pd",
        "ttft-lo",
        "speedup"
    );
    for r in rows {
        r.validate()?;
        let local = r
            .ttft_local_ms
            .map_or_else(|| "       -".into(), |v| format!("{v:8.0}"));
        let speedup = r
            .ttft_speedup
            .map_or_else(|| "      -".into(), |v| format!("{v:7.2}"));
        let matches = if r.matches { "True" } else { "False" };
        writeln!(
            out,
            "{:>7} {:7.1} {:6.2} {:8.0} {:7.0} {:7.0} {:7.0} {:8.0} {} {} {}",
            r.prompt_token_count,
            r.state_bytes as f64 / 1_048_576.0,
            r.transfer_gbps,
            r.source_prefill_ms,
            r.state_export_ms,
            r.transfer_ms,
            r.receiver.kv_attach_ms,
            r.ttft_disaggregated_ms,
            local,
            speedup,
            matches
        )?;
    }
    Ok(out)
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let [flag, directory] = args else {
        return Err(USAGE.into());
    };
    if flag != "--reports-dir" {
        return Err(USAGE.into());
    }
    let text = render(&reports(Path::new(directory))?)?;
    print!("{text}");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    pub(super) fn fixture(prefix: u64, matches: bool) -> serde_json::Value {
        serde_json::json!({"prompt_token_count":prefix,"state_bytes":1572864,"transfer_gbps":1.25,"source_prefill_ms":200.0,"state_export_ms":3.0,"transfer_ms":4.0,"receiver":{"kv_attach_ms":5.0},"ttft_disaggregated_ms":212.0,"matches":matches,"unknown_producer_field":"ignored"})
    }
    #[test]
    fn handoff_summary_sorts_numeric_prefix_and_retains_measurements_null_baseline_and_mismatch() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("send-2.json"),
            serde_json::to_vec(&fixture(2, false)).unwrap(),
        )
        .unwrap();
        let mut row = fixture(10, true);
        row["ttft_local_ms"] = 1000.into();
        row["ttft_speedup"] = 2.25.into();
        std::fs::write(
            root.path().join("send-10.json"),
            serde_json::to_vec(&row).unwrap(),
        )
        .unwrap();
        std::fs::write(root.path().join("unrelated.json"), b"not-json").unwrap();
        let out = render(&reports(root.path()).unwrap()).unwrap();
        let lines: Vec<_> = out.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(
            lines[1].split_whitespace().collect::<Vec<_>>(),
            [
                "2", "1.5", "1.25", "200", "3", "4", "5", "212", "-", "-", "False"
            ]
        );
        assert_eq!(
            lines[2].split_whitespace().collect::<Vec<_>>(),
            [
                "10", "1.5", "1.25", "200", "3", "4", "5", "212", "1000", "2.25", "True"
            ]
        );
        root.close().unwrap();
    }
    #[test]
    fn handoff_summary_refuses_invalid_measurement_or_shape_before_rendering_any_rows() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("send-1.json");
        for value in [
            serde_json::json!({}),
            {
                let mut v = fixture(1, true);
                v["transfer_gbps"] = (-1).into();
                v
            },
            {
                let mut v = fixture(1, true);
                v["matches"] = "true".into();
                v
            },
            {
                let mut v = fixture(1, true);
                v["prompt_token_count"] = 1.5.into();
                v
            },
        ] {
            std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(reports(root.path()).is_err());
        }
        std::fs::write(&path, b"not-json").unwrap();
        assert!(reports(root.path()).is_err());
        std::fs::remove_file(&path).unwrap();
        assert_eq!(
            render(&reports(root.path()).unwrap())
                .unwrap()
                .lines()
                .count(),
            1
        );
        root.close().unwrap();
    }
}
