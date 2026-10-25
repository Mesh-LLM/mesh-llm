//! Offline aggregation of downloaded compiler-cache counters.
mod input;
mod options;

use crate::repository::check_report::CheckReport;
use serde::Serialize;

#[derive(Serialize)]
struct Summary {
    file_count: usize,
    cache_hits: u64,
    cache_misses: u64,
    cache_requests: u64,
    hit_rate: Option<f64>,
    minimum_hit_rate: Option<f64>,
    passed: bool,
}

pub(crate) fn run(arguments: &[String]) -> CheckReport {
    let options = match options::parse(arguments) {
        Ok(Some(options)) => options,
        Ok(None) => return CheckReport::success(format!("{}\n", options::USAGE)),
        Err(message) => return CheckReport::usage(options::USAGE, &message),
    };
    match aggregate(&options) {
        Ok(summary) => {
            let stdout = if options.json {
                format!(
                    "{}\n",
                    serde_json::to_string(&summary).expect("finite summary")
                )
            } else {
                render(&summary)
            };
            CheckReport {
                stdout,
                stderr: String::new(),
                code: i32::from(!summary.passed),
            }
        }
        Err(error) => {
            CheckReport::failure(String::new(), format!("sccache summary error: {error}\n"))
        }
    }
}

fn aggregate(options: &options::Options) -> Result<Summary, String> {
    let files = input::discover(&options.paths)?;
    let mut hits = 0_u64;
    let mut misses = 0_u64;
    let mut bytes = 0_usize;
    for path in &files {
        let (file_hits, file_misses) = input::read(path, &mut bytes)?;
        hits = hits
            .checked_add(file_hits)
            .ok_or("cache hit total overflow")?;
        misses = misses
            .checked_add(file_misses)
            .ok_or("cache miss total overflow")?;
    }
    let requests = hits
        .checked_add(misses)
        .ok_or("cache request total overflow")?;
    let hit_rate = (requests != 0).then(|| hits as f64 / requests as f64);
    let passed = options
        .minimum
        .is_none_or(|minimum| hit_rate.is_some_and(|rate| rate >= minimum));
    Ok(Summary {
        file_count: files.len(),
        cache_hits: hits,
        cache_misses: misses,
        cache_requests: requests,
        hit_rate,
        minimum_hit_rate: options.minimum,
        passed,
    })
}

fn render(summary: &Summary) -> String {
    let rate = summary
        .hit_rate
        .map_or_else(|| "n/a".to_owned(), |rate| format!("{:.2}%", rate * 100.0));
    let mut text = format!(
        "Sccache cache-hit summary\nEvidence files: {}\nCache hits: {}\nCache misses: {}\nHit rate: {rate}\n",
        summary.file_count, summary.cache_hits, summary.cache_misses
    );
    if let Some(minimum) = summary.minimum_hit_rate {
        text.push_str(&format!(
            "Minimum hit rate: {:.2}% ({})\n",
            minimum * 100.0,
            if summary.passed { "PASS" } else { "FAIL" }
        ));
    }
    text
}
