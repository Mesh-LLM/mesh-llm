use crate::command::DynResult;
use serde::Serialize;

/// AgentX 1.0.7 omits grace-period cancellations from its aggregate JSON.
/// Its pinned phase-complete log is the authoritative request accounting.
#[derive(Default, Debug, Serialize, PartialEq)]
pub(super) struct PhaseOutcome {
    pub completed: u64,
    pub cancelled: u64,
    pub errors: u64,
}

pub(super) fn parse(log: &str, expected_completed: u64) -> DynResult<PhaseOutcome> {
    let mut outcome = PhaseOutcome::default();
    let mut phases = 0;
    for line in log.lines() {
        let Some((_, counters)) = line.split_once("(profiling) complete | ") else {
            continue;
        };
        let counters = counters
            .split(" | ")
            .next()
            .ok_or("missing phase counters")?;
        let values = counters.split(", ").collect::<Vec<_>>();
        let [completed, cancelled, errors] = values.as_slice() else {
            return Err("invalid AgentX phase counters".into());
        };
        for (total, field, prefix) in [
            (&mut outcome.completed, *completed, "completed="),
            (&mut outcome.cancelled, *cancelled, "cancelled="),
            (&mut outcome.errors, *errors, "errors="),
        ] {
            let value: u64 = field
                .strip_prefix(prefix)
                .ok_or("missing phase counter")?
                .parse()?;
            *total = total.checked_add(value).ok_or("phase counter overflow")?;
        }
        phases += 1;
    }
    if phases == 0 || outcome.completed != expected_completed {
        return Err(
            "AgentX phase accounting missing or disagrees with exported completions".into(),
        );
    }
    if outcome.errors != 0 {
        return Err("AgentX profiling phase contains request errors".into());
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRAINED: &str = "12:21:29 NOTICE Phase profiling (profiling) complete | completed=6, cancelled=1, errors=0 | sessions: completed=0, cancelled=0 | elapsed=210.01s | grace_period_timeout=True";

    #[test]
    fn preserves_grace_cancellations_and_excludes_warmup() {
        let log = format!(
            "Phase warmup (warmup) complete | completed=2, cancelled=0, errors=0\n{DRAINED}"
        );
        assert_eq!(
            parse(&log, 6).unwrap(),
            PhaseOutcome {
                completed: 6,
                cancelled: 1,
                errors: 0
            }
        );
    }

    #[test]
    fn rejects_missing_malformed_failed_and_inconsistent_accounting() {
        for (log, expected) in [
            (String::new(), 6),
            (DRAINED.to_owned(), 7),
            (DRAINED.replace("cancelled=1", "cancelled=-1"), 6),
            (DRAINED.replace("errors=0", "errors=1"), 6),
        ] {
            assert!(parse(&log, expected).is_err());
        }
    }

    #[test]
    fn sums_multiple_measured_phases() {
        assert_eq!(
            parse(&format!("{DRAINED}\n{DRAINED}"), 12)
                .unwrap()
                .cancelled,
            2
        );
    }
}
