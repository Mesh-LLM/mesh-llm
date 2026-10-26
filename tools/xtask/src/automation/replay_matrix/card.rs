//! Published replay card: complete latest results and the seven-date trend.
use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Deserialize)]
struct Row {
    complete: bool,
    created_utc: String,
    source_sha: String,
    model: Model,
    replay: Replay,
    decode_tokens_per_second: f64,
    end_to_end_tokens_per_second: f64,
    ttft_ms_mean: f64,
    ttft_ms_p90: f64,
    cache_hit_pct: Option<f64>,
    finish_reason_length_pct: Option<f64>,
}
#[derive(Clone, Deserialize)]
struct Model {
    family: String,
    repo: String,
    quant: String,
    class: String,
}
#[derive(Clone, Deserialize)]
struct Replay {
    concurrency: u64,
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix card --latest PATH --output PATH [--history DIRECTORY]",
        values: &["--latest", "--output", "--history"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(value) => value,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let latest = load(Path::new(
        parsed.last("--latest").ok_or("missing --latest")?,
    ))?;
    let mut previous = Vec::new();
    if let Some(root) = parsed.last("--history") {
        let root = Path::new(root);
        if root.is_dir() {
            let mut paths = Vec::new();
            collect(root, &mut paths)?;
            paths.sort();
            for path in paths {
                previous.extend(load(&path)?);
            }
        }
    }
    let output = Path::new(parsed.last("--output").ok_or("missing --output")?);
    let card = render(&latest, previous)?;
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, card)?;
    CheckReport::success(format!("wrote card to {}\n", output.display())).emit()
}
fn collect(root: &Path, paths: &mut Vec<PathBuf>) -> DynResult<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect(&entry.path(), paths)?;
        } else if kind.is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "jsonl")
        {
            paths.push(entry.path());
        } else if kind.is_symlink() {
            return Err("history card does not follow shard symlinks".into());
        }
    }
    Ok(())
}
fn load(path: &Path) -> DynResult<Vec<Row>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let row: Row = serde_json::from_str(line)?;
            validate(&row)?;
            Ok(row)
        })
        .collect()
}
fn validate(row: &Row) -> DynResult<()> {
    if row.created_utc.len() < 10
        || !row.created_utc.is_char_boundary(10)
        || row.source_sha.len() < 9
        || !row.source_sha.is_char_boundary(9)
        || row.replay.concurrency == 0
    {
        return Err("invalid card history identity".into());
    }
    for text in [
        &row.created_utc,
        &row.source_sha,
        &row.model.family,
        &row.model.repo,
        &row.model.quant,
        &row.model.class,
    ] {
        if text.trim().is_empty() || text.contains(['\n', '\r', '|']) {
            return Err("invalid card table text".into());
        }
    }
    for metric in [
        Some(row.decode_tokens_per_second),
        Some(row.end_to_end_tokens_per_second),
        Some(row.ttft_ms_mean),
        Some(row.ttft_ms_p90),
        row.cache_hit_pct,
        row.finish_reason_length_pct,
    ]
    .into_iter()
    .flatten()
    {
        if !metric.is_finite() || metric < 0.0 {
            return Err("invalid card history metric".into());
        }
    }
    Ok(())
}
fn percentage(value: Option<f64>) -> String {
    value.map_or_else(|| "—".into(), |value| format!("{value:.1}"))
}
fn latest_table(out: &mut String, latest: &[&Row]) -> DynResult<()> {
    if latest.is_empty() {
        out.push_str("_No complete results in the latest run._\n\n");
        return Ok(());
    }
    let newest = latest.iter().map(|row| &row.created_utc).max().unwrap();
    writeln!(
        out,
        "Latest complete run: **{newest}** @ `{}`\n",
        &latest[0].source_sha[..9]
    )?;
    out.push_str("| Model | Class | Conc. | Decode tok/s | E2E tok/s | TTFT mean | TTFT p90 | Cache hit % | Finish=length % |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut sorted = latest.to_vec();
    sorted.sort_by(|a, b| {
        b.decode_tokens_per_second
            .total_cmp(&a.decode_tokens_per_second)
    });
    for row in sorted {
        writeln!(
            out,
            "| {} {} | {} | {} | {:.1} | {:.1} | {:.0} ms | {:.0} ms | {} | {} |",
            row.model.repo,
            row.model.quant,
            row.model.class,
            row.replay.concurrency,
            row.decode_tokens_per_second,
            row.end_to_end_tokens_per_second,
            row.ttft_ms_mean,
            row.ttft_ms_p90,
            percentage(row.cache_hit_pct),
            percentage(row.finish_reason_length_pct)
        )?;
    }
    Ok(())
}
fn render(latest: &[Row], mut previous: Vec<Row>) -> DynResult<String> {
    let complete = latest.iter().filter(|row| row.complete).collect::<Vec<_>>();
    let mut out = String::from(
        "# MeshLLM Coding-Agent Serving Benchmark\n\nNightly agentic-replay results on the pinned micstudio runner (Apple M3 Ultra, 256 GB, Metal 4). Full history in this dataset; schema in `schema.json`.\n\n## Latest run\n\n",
    );
    latest_table(&mut out, &complete)?;
    previous.extend(complete.into_iter().cloned());
    let mut recent = previous
        .iter()
        .map(|row| row.created_utc[..10].to_owned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if recent.len() > 7 {
        recent.drain(..recent.len() - 7);
    }
    let mut cohorts = BTreeMap::<String, Vec<&Row>>::new();
    for row in &previous {
        if row.complete && row.replay.concurrency == 4 {
            cohorts
                .entry(row.model.family.clone())
                .or_default()
                .push(row);
        }
    }
    writeln!(out, "\n## Trend — decode tok/s (concurrency 4)\n")?;
    writeln!(out, "| Model | {} |", recent.join(" | "))?;
    writeln!(out, "{}|", "|---".repeat(recent.len() + 1))?;
    for (family, mut rows) in cohorts {
        rows.sort_by(|a, b| a.created_utc.cmp(&b.created_utc));
        let cells = recent
            .iter()
            .map(|date| {
                rows.iter()
                    .rev()
                    .find(|row| row.created_utc.starts_with(date))
                    .map_or_else(
                        || "—".into(),
                        |row| format!("{:.1}", row.decode_tokens_per_second),
                    )
            })
            .collect::<Vec<_>>();
        writeln!(out, "| {family} | {} |", cells.join(" | "))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(date: &str, family: &str, concurrency: u64, speed: f64, complete: bool) -> Row {
        Row {
            complete,
            created_utc: format!("{date}T01:00:00Z"),
            source_sha: "abcdef0123456789".into(),
            model: Model {
                family: family.into(),
                repo: format!("fixture/{family}"),
                quant: "Q8_0".into(),
                class: "dense".into(),
            },
            replay: Replay { concurrency },
            decode_tokens_per_second: speed,
            end_to_end_tokens_per_second: speed / 2.0,
            ttft_ms_mean: 12.0,
            ttft_ms_p90: 20.0,
            cache_hit_pct: None,
            finish_reason_length_pct: Some(0.0),
        }
    }
    #[test]
    fn complete_latest_rows_sort_by_decode_and_render_nullable_percentages() {
        let rows = vec![
            row("2026-10-02", "slow", 1, 10.0, true),
            row("2026-10-02", "fast", 4, 30.0, true),
            row("2026-10-02", "failed", 4, 90.0, false),
        ];
        let card = render(&rows, vec![]).unwrap();
        assert!(card.find("fixture/fast").unwrap() < card.find("fixture/slow").unwrap());
        assert!(!card.contains("failed"));
        assert!(card.contains("| 30.0 | 15.0 | 12 ms | 20 ms | — | 0.0 |"));
        assert!(card.contains("@ `abcdef012`"));
    }
    #[test]
    fn trend_uses_seven_dates_fixed_concurrency_and_latest_daily_complete_sample() {
        let mut prior = (1..=9)
            .map(|day| {
                row(
                    &format!("2026-09-{day:02}"),
                    "dense",
                    4,
                    f64::from(day),
                    true,
                )
            })
            .collect::<Vec<_>>();
        let mut last = row("2026-09-09", "dense", 4, 19.0, true);
        last.created_utc = "2026-09-09T23:00:00Z".into();
        prior.push(last);
        prior.push(row("2026-09-09", "other-level", 1, 999.0, true));
        prior.push(row("2026-09-09", "failed", 4, 999.0, false));
        let card = render(&[], prior).unwrap();
        assert!(card.contains("_No complete results in the latest run._"));
        assert!(!card.contains("2026-09-01"));
        assert!(!card.contains("2026-09-02"));
        assert!(card.contains("| dense | 3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0 | 19.0 |"));
        assert!(!card.contains("other-level"));
        assert!(!card.contains("failed"));
    }
    #[test]
    fn malformed_json_and_metrics_fail_before_output_creation() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("latest.jsonl");
        let output = root.path().join("README.md");
        fs::write(&input, "{\"complete\":\"True\"}\n").unwrap();
        let args = vec![
            "--latest".into(),
            input.display().to_string(),
            "--output".into(),
            output.display().to_string(),
        ];
        assert!(run(&args).is_err());
        assert!(!output.exists());
        let mut bad = row("2026-10-02", "dense", 4, 1.0, true);
        bad.ttft_ms_mean = f64::NAN;
        assert!(validate(&bad).is_err());
    }
    #[test]
    fn recursive_shards_are_loaded_in_stable_path_order_and_missing_history_bootstraps() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("nested")).unwrap();
        fs::write(root.path().join("nested/b.jsonl"), b"\n").unwrap();
        fs::write(root.path().join("a.jsonl"), b"\n").unwrap();
        fs::write(root.path().join("ignored.txt"), b"not JSON").unwrap();
        let mut paths = vec![];
        collect(root.path(), &mut paths).unwrap();
        paths.sort();
        assert_eq!(paths.len(), 2);
        assert!(paths[0].ends_with("a.jsonl"));
        let output = root.path().join("card.md");
        let args = vec![
            "--latest".into(),
            paths[0].display().to_string(),
            "--output".into(),
            output.display().to_string(),
            "--history".into(),
            root.path().join("missing").display().to_string(),
        ];
        run(&args).unwrap();
        assert!(
            fs::read_to_string(output)
                .unwrap()
                .contains("No complete results")
        );
    }
}
