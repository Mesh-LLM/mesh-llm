//! A bounded repair index derived exclusively from an admitted feedback capsule.
use super::super::{
    Error, Family,
    storage::{RESULTS_LIMIT, read_bounded},
};
use super::VerifiedFeedback;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Read},
};

const MAXIMUM_OUTPUT: usize = 64 * 1024;
const MAXIMUM_LOG_SCAN: u64 = 1024 * 1024;
const MAXIMUM_TRACE_CHARACTERS: usize = 300;
const HEADING: &str = "Grouped candidate failures (first useful trace per family):\n";
const FOOTER: &str = "Run the affected family or a reduced reproducer first; the trusted wrapper still runs every final gate.\n";
const TRUNCATED: &str = "[Summary truncated; see the remaining verified family evidence.]\n";

type Groups = BTreeMap<String, Vec<(String, String)>>;

pub(super) fn render(feedback: &VerifiedFeedback) -> Result<String, Error> {
    let mut groups = Groups::new();
    for family in feedback.candidate_failures() {
        let path = feedback
            .snapshot
            .root()
            .join(family.as_str())
            .join("results.jsonl");
        let bytes = read_bounded(&path, RESULTS_LIMIT)?;
        let (lane, note) = first_failure(&bytes)?;
        let detail = match note {
            Some(note) => note,
            None => log_trace(feedback, family)?.unwrap_or_else(|| "see family evidence".into()),
        };
        groups.entry(lane).or_default().push((
            family.to_string(),
            single_line(&detail, MAXIMUM_TRACE_CHARACTERS),
        ));
    }
    Ok(render_groups(&groups))
}

fn first_failure(bytes: &[u8]) -> Result<(String, Option<String>), Error> {
    for row in serde_json::Deserializer::from_slice(bytes).into_iter::<Value>() {
        let row = row?;
        let Some(outcomes) = row.get("outcomes").and_then(Value::as_array) else {
            continue;
        };
        for outcome in outcomes {
            if outcome.get("status").and_then(Value::as_str) != Some("fail") {
                continue;
            }
            let lane = outcome
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("unclassified");
            let lane = single_line(lane, 80);
            let note = outcome
                .get("note")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|note| !note.is_empty())
                .map(str::to_owned);
            return Ok((
                if lane.is_empty() {
                    "unclassified".into()
                } else {
                    lane
                },
                note,
            ));
        }
    }
    Ok(("unclassified".into(), None))
}

fn log_trace(feedback: &VerifiedFeedback, family: &Family) -> Result<Option<String>, Error> {
    let prefix = format!("{family}/");
    let mut remaining = MAXIMUM_LOG_SCAN;
    // The snapshot manifest is already sorted and contains only admitted members.
    for path in feedback.snapshot.files.keys() {
        let Some(relative) = path
            .strip_prefix(&prefix)
            .filter(|path| path.ends_with(".log"))
        else {
            continue;
        };
        if remaining == 0 {
            break;
        }
        let source = File::open(feedback.snapshot.root().join(path))?;
        let mut reader = BufReader::new(source.take(remaining));
        let mut line = Vec::new();
        loop {
            line.clear();
            let count = reader.read_until(b'\n', &mut line)?;
            if count == 0 {
                break;
            }
            remaining -= u64::try_from(count).expect("log reads are bounded to one MiB");
            if useful_trace(&line) {
                let line = String::from_utf8_lossy(&line);
                return Ok(Some(format!("{relative}: {}", line.trim())));
            }
        }
    }
    Ok(None)
}

fn useful_trace(line: &[u8]) -> bool {
    [
        b"error".as_slice(),
        b"fail",
        b"mismatch",
        b"assert",
        b"panic",
    ]
    .iter()
    .any(|word| {
        line.windows(word.len())
            .any(|window| window.eq_ignore_ascii_case(word))
    })
}

fn single_line(text: &str, maximum: usize) -> String {
    text.trim()
        .chars()
        .take(maximum)
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect()
}

fn render_groups(groups: &Groups) -> String {
    let mut output = HEADING.to_owned();
    let body_limit = MAXIMUM_OUTPUT - FOOTER.len() - TRUNCATED.len();
    'groups: for (lane, entries) in groups {
        let heading = format!("- {lane} ({}):\n", entries.len());
        if !append_line(&mut output, &heading, body_limit) {
            break;
        }
        for (family, trace) in entries {
            if !append_line(&mut output, &format!("  {family}: {trace}\n"), body_limit) {
                break 'groups;
            }
        }
    }
    output.push_str(FOOTER);
    output
}

fn append_line(output: &mut String, line: &str, body_limit: usize) -> bool {
    if output.len() + line.len() > body_limit {
        output.push_str(TRUNCATED);
        false
    } else {
        output.push_str(line);
        true
    }
}

#[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
#[path = "summary_tests.rs"]
mod tests;
