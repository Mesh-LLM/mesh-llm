/// Tolerance (in seconds) when comparing a recorded start time against the
/// live process start time. A difference of up to this many seconds is treated
/// as the same process.
pub const START_TIME_TOLERANCE_SECS: i64 = 2;

/// Liveness state inferred from whether the process comm is readable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Liveness {
    /// Process is alive because comm was readable.
    Alive,
    /// Process is gone because the PID was not found.
    Dead,
    /// Liveness could not be determined.
    Unknown,
}

#[cfg(target_os = "linux")]
mod platform {
    use std::sync::OnceLock;

    static BTIME: OnceLock<i64> = OnceLock::new();

    fn btime() -> i64 {
        *BTIME.get_or_init(|| {
            (|| -> anyhow::Result<i64> {
                let content = std::fs::read_to_string("/proc/stat")?;
                for line in content.lines() {
                    if let Some(rest) = line.strip_prefix("btime ") {
                        return Ok(rest.trim().parse()?);
                    }
                }
                anyhow::bail!("btime line not found in /proc/stat")
            })()
            .unwrap_or(0)
        })
    }

    pub fn process_comm(pid: u32) -> anyhow::Result<Option<String>> {
        let path = format!("/proc/{pid}/comm");
        match std::fs::read_to_string(&path) {
            Ok(s) => Ok(Some(s.trim().to_string())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                tracing::debug!(pid, path, "permission denied reading process comm");
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn process_executable_name(pid: u32) -> anyhow::Result<Option<String>> {
        let path = format!("/proc/{pid}/exe");
        match std::fs::read_link(&path) {
            Ok(target) => Ok(target
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                tracing::debug!(
                    pid,
                    path,
                    "permission denied reading process executable path"
                );
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
        let path = format!("/proc/{pid}/stat");
        let content = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                tracing::debug!(pid, "permission denied reading /proc/{pid}/stat");
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };

        let rparen = content
            .rfind(')')
            .ok_or_else(|| anyhow::anyhow!("malformed /proc/{pid}/stat: no closing ')' found"))?;
        let after_comm = content.get(rparen + 2..).unwrap_or("");
        let fields: Vec<&str> = after_comm.split_whitespace().collect();

        let starttime_ticks: u64 = fields
            .get(19)
            .ok_or_else(|| anyhow::anyhow!("starttime field missing in /proc/{pid}/stat"))?
            .parse()
            .map_err(|e| anyhow::anyhow!("failed to parse starttime in /proc/{pid}/stat: {e}"))?;

        let clk_tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        if clk_tck <= 0 {
            anyhow::bail!("sysconf(_SC_CLK_TCK) returned {clk_tck}");
        }

        let bt = btime();
        if bt == 0 {
            anyhow::bail!("could not determine boot time from /proc/stat");
        }

        Ok(Some(bt + (starttime_ticks as i64 / clk_tck)))
    }
}

#[cfg(target_os = "macos")]
mod platform {
    pub fn process_comm(pid: u32) -> anyhow::Result<Option<String>> {
        let output = std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "comm="])
            .output()?;
        let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if s.is_empty() {
            return Ok(None);
        }
        let basename = std::path::Path::new(&s)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or(s);
        Ok(Some(basename))
    }

    pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
        let output = std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "lstart="])
            .env("LANG", "C")
            .env("LC_ALL", "C")
            .output()?;
        let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if s.is_empty() {
            return Ok(None);
        }
        super::parse_lstart(&s)
    }

    pub fn process_executable_name(pid: u32) -> anyhow::Result<Option<String>> {
        process_comm(pid)
    }
}

/// Parse the output of `ps -o lstart=` under `LANG=C`/`LC_ALL=C`, e.g.
/// `Wed Sep  9 18:43:39 2026`: weekday, month, day, time, year, and convert
/// it to a Unix timestamp in the local timezone.
#[cfg(target_os = "macos")]
fn parse_lstart(s: &str) -> anyhow::Result<Option<i64>> {
    use chrono::{Local, TimeZone};

    let Some(naive_dt) = parse_lstart_naive(s) else {
        return Ok(None);
    };

    match Local.from_local_datetime(&naive_dt).single() {
        Some(dt) => Ok(Some(dt.timestamp())),
        None => Ok(None),
    }
}

/// Parses the weekday/month/day/time/year fields into a naive (timezone-free)
/// datetime. Split out of [`parse_lstart`] and out of the macOS-only platform
/// module (its only real caller) so this field decoding is unit-tested on
/// every CI runner rather than only a macOS one, and so the test doesn't
/// depend on the runner's local timezone.
#[cfg(any(test, target_os = "macos"))]
fn parse_lstart_naive(s: &str) -> Option<chrono::NaiveDateTime> {
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 5 {
        return None;
    }

    let month: u32 = match parts[1] {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let day: u32 = parts[2].parse().ok()?;
    let year: i32 = parts[4].parse().ok()?;

    let time_parts: Vec<&str> = parts[3].split(':').collect();
    if time_parts.len() != 3 {
        return None;
    }
    let hour: u32 = time_parts[0].parse().ok()?;
    let min: u32 = time_parts[1].parse().ok()?;
    let sec: u32 = time_parts[2].parse().ok()?;

    let date = NaiveDate::from_ymd_opt(year, month, day)?;
    let time = NaiveTime::from_hms_opt(hour, min, sec)?;
    Some(NaiveDateTime::new(date, time))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    pub fn process_comm(_pid: u32) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    pub fn process_started_at_unix(_pid: u32) -> anyhow::Result<Option<i64>> {
        Ok(None)
    }

    pub fn process_executable_name(_pid: u32) -> anyhow::Result<Option<String>> {
        Ok(None)
    }
}

/// Read the command name of the given process.
pub fn process_comm(pid: u32) -> anyhow::Result<Option<String>> {
    platform::process_comm(pid)
}

/// Read the Unix start time of the given process.
pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
    platform::process_started_at_unix(pid)
}

/// Read the executable basename for the given process when available.
pub fn process_executable_name(pid: u32) -> anyhow::Result<Option<String>> {
    platform::process_executable_name(pid)
}

/// Determine liveness of a process by attempting to read its comm.
pub fn process_liveness(pid: u32) -> Liveness {
    match process_comm(pid) {
        Ok(Some(_)) => Liveness::Alive,
        Ok(None) => Liveness::Dead,
        Err(_) => Liveness::Unknown,
    }
}

/// Returns true iff the live process name matches the expected spawned binary.
pub fn process_name_matches(pid: u32, expected_comm: &str) -> bool {
    process_executable_name(pid)
        .ok()
        .flatten()
        .is_some_and(|name| name == expected_comm)
        || process_comm(pid)
            .ok()
            .flatten()
            .is_some_and(|name| name == expected_comm)
}

/// Returns true iff the live process matches the expected name and start time.
#[cfg(not(windows))]
pub fn validate_pid_matches(pid: u32, expected_comm: &str, expected_started_at_unix: i64) -> bool {
    match process_started_at_unix(pid) {
        Ok(Some(t)) => {
            process_name_matches(pid, expected_comm)
                && (t - expected_started_at_unix).abs() <= START_TIME_TOLERANCE_SECS
        }
        _ => false,
    }
}

/// Returns the Unix start time of the current process.
pub fn current_process_start_time_unix() -> anyhow::Result<i64> {
    process_started_at_unix(std::process::id())?
        .ok_or_else(|| anyhow::anyhow!("could not determine start time of current process"))
}

#[cfg(test)]
mod tests {
    use super::parse_lstart_naive;
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

    fn naive(year: i32, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> NaiveDateTime {
        NaiveDateTime::new(
            NaiveDate::from_ymd_opt(year, month, day).unwrap(),
            NaiveTime::from_hms_opt(hour, min, sec).unwrap(),
        )
    }

    #[test]
    fn parses_single_digit_day_padded_with_a_double_space() {
        // macOS `ps -o lstart=` right-aligns the day to two columns, so a
        // single-digit day leaves two spaces before it — exactly what F8
        // observed (galaxy's process started on Sep 9).
        let parsed = parse_lstart_naive("Wed Sep  9 18:43:39 2026").unwrap();

        assert_eq!(parsed, naive(2026, 9, 9, 18, 43, 39));
    }

    #[test]
    fn parses_double_digit_day() {
        let parsed = parse_lstart_naive("Mon Jan 12 09:05:00 2026").unwrap();

        assert_eq!(parsed, naive(2026, 1, 12, 9, 5, 0));
    }

    #[test]
    fn does_not_transpose_a_month_number_larger_than_any_day() {
        // Regression guard for the day/month swap: December (month 12) used
        // to get read as if it were the day field.
        let parsed = parse_lstart_naive("Thu Dec  3 00:00:00 2026").unwrap();

        assert_eq!(parsed, naive(2026, 12, 3, 0, 0, 0));
    }

    #[test]
    fn rejects_malformed_field_count() {
        assert!(parse_lstart_naive("Sep 9 18:43:39 2026").is_none());
    }

    #[test]
    fn rejects_unknown_month_name() {
        assert!(parse_lstart_naive("Wed Foo 9 18:43:39 2026").is_none());
    }
}
