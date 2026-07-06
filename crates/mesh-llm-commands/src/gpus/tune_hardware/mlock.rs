use super::types::{
    TuneMemoryBudget, TuneMemorySource, TuneMlockEvaluation, format_bytes, memory_label,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TuneMlockProbe {
    /// `Supported` is never constructed on non-Linux targets, so Clippy
    /// flags its fields as dead code when checked without `--cfg target_os`.
    #[allow(dead_code)]
    Supported { limit: TuneMlockLimit },
    Unsupported { reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Only constructed on Linux via `read_linux_mlock_limit`; dead-code warning
/// suppressed for cross-compilation targets that skip the Linux helpers.
#[allow(dead_code)]
pub(crate) enum TuneMlockLimit {
    Unlimited,
    Bytes(u64),
}

pub(super) fn detect_mlock_probe() -> TuneMlockProbe {
    #[cfg(target_os = "linux")]
    {
        if let Some(limit) = read_linux_mlock_limit() {
            return TuneMlockProbe::Supported { limit };
        }
        TuneMlockProbe::Unsupported {
            reason: "mlock unavailable: could not read /proc/self/limits for the current process, and tune will not attempt privilege changes"
                .to_string(),
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        TuneMlockProbe::Unsupported {
            reason: "mlock availability reporting is not implemented for this platform in v1; tune will not attempt privilege changes"
                .to_string(),
        }
    }
}

pub(super) fn evaluate_mlock(
    memory: &TuneMemoryBudget,
    probe: TuneMlockProbe,
) -> TuneMlockEvaluation {
    match probe {
        TuneMlockProbe::Supported {
            limit: TuneMlockLimit::Unlimited,
        } => TuneMlockEvaluation {
            available: true,
            reason: format!(
                "mlock is available for the evaluated {} budget of {} under an unlimited lock limit; tune still reports it without writing a config change",
                memory_label(memory.source),
                format_bytes(memory.allocatable_bytes),
            ),
        },
        TuneMlockProbe::Supported {
            limit: TuneMlockLimit::Bytes(limit_bytes),
        } => build_limited_mlock_report(memory.source, memory.allocatable_bytes, limit_bytes),
        TuneMlockProbe::Unsupported { reason } => TuneMlockEvaluation {
            available: false,
            reason,
        },
    }
}

fn build_limited_mlock_report(
    source: TuneMemorySource,
    allocatable_bytes: u64,
    limit_bytes: u64,
) -> TuneMlockEvaluation {
    if limit_bytes >= allocatable_bytes {
        return TuneMlockEvaluation {
            available: true,
            reason: format!(
                "mlock is available for the evaluated {} budget of {} because the current lock limit is {}",
                memory_label(source),
                format_bytes(allocatable_bytes),
                format_bytes(limit_bytes),
            ),
        };
    }

    TuneMlockEvaluation {
        available: false,
        reason: format!(
            "mlock unavailable for the evaluated {} budget of {}: current lock limit is {}. Tune will not attempt privilege changes; raise RLIMIT_MEMLOCK or container IPC_LOCK if you need full locking.",
            memory_label(source),
            format_bytes(allocatable_bytes),
            format_bytes(limit_bytes),
        ),
    }
}

#[cfg(target_os = "linux")]
fn read_linux_mlock_limit() -> Option<TuneMlockLimit> {
    let limits = std::fs::read_to_string("/proc/self/limits").ok()?;
    limits.lines().find_map(parse_linux_mlock_limit)
}

#[cfg(target_os = "linux")]
fn parse_linux_mlock_limit(line: &str) -> Option<TuneMlockLimit> {
    let mut columns = line.split_whitespace();
    let first = columns.next()?;
    let second = columns.next()?;
    let third = columns.next()?;
    let soft = columns.next()?;
    if first != "Max" || second != "locked" || third != "memory" {
        return None;
    }
    match soft {
        "unlimited" => Some(TuneMlockLimit::Unlimited),
        value => value.parse::<u64>().ok().map(TuneMlockLimit::Bytes),
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::{TuneMlockLimit, parse_linux_mlock_limit};

    #[test]
    fn parses_linux_max_locked_memory_soft_limit() {
        let line = "Max locked memory         8241545216           8241545216           bytes";

        assert_eq!(
            parse_linux_mlock_limit(line),
            Some(TuneMlockLimit::Bytes(8_241_545_216))
        );
    }

    #[test]
    fn parses_linux_unlimited_mlock_limit() {
        let line = "Max locked memory         unlimited            unlimited            bytes";

        assert_eq!(parse_linux_mlock_limit(line), Some(TuneMlockLimit::Unlimited));
    }
}
