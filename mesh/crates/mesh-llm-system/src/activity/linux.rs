use std::fs;
use std::process;

use super::{HostActivity, HostActivityDetector, PriorityController, PriorityFailure};

const CPU_ACTIVE_PERCENT_THRESHOLD: u64 = 10;

#[derive(Debug, Clone, Copy)]
struct CpuSnapshot {
    total: u64,
    idle: u64,
}

#[derive(Debug, Default)]
pub struct NativeHostActivityDetector {
    previous_cpu: Option<CpuSnapshot>,
}

impl HostActivityDetector for NativeHostActivityDetector {
    fn sample(&mut self) -> HostActivity {
        let current = match read_cpu_snapshot() {
            Some(value) => value,
            None => {
                self.previous_cpu = None;
                return HostActivity::Unknown;
            }
        };

        let observed = match self.previous_cpu {
            Some(previous) => classify_cpu_activity(previous, current),
            None => HostActivity::Unknown,
        };

        self.previous_cpu = Some(current);
        observed
    }
}

#[derive(Debug)]
pub struct NativePriorityController {
    pid: u32,
    original_nice: Option<i32>,
}

impl Default for NativePriorityController {
    fn default() -> Self {
        Self {
            pid: process::id(),
            original_nice: None,
        }
    }
}

impl PriorityController for NativePriorityController {
    fn reduce_priority(&mut self) -> Result<(), PriorityFailure> {
        #[cfg(target_os = "linux")]
        {
            if !can_restore_priority() {
                return Err(PriorityFailure::Unsupported);
            }
            let current = read_process_nice(self.pid).ok_or(PriorityFailure::ApplyFailed)?;
            if self.original_nice.is_none() {
                let target = (current + 10).clamp(-20, 19);
                set_process_nice(self.pid, target).map_err(|_| PriorityFailure::ApplyFailed)?;
                self.original_nice = Some(current);
            }
            Ok(())
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(PriorityFailure::Unsupported)
        }
    }

    fn restore_priority(&mut self) -> Result<(), PriorityFailure> {
        #[cfg(target_os = "linux")]
        {
            let original_nice = match self.original_nice {
                Some(value) => value,
                None => return Ok(()),
            };

            set_process_nice(self.pid, original_nice)
                .map_err(|_| PriorityFailure::RestoreFailed)?;

            self.original_nice = None;

            Ok(())
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(PriorityFailure::Unsupported)
        }
    }
}

#[cfg(target_os = "linux")]
fn can_restore_priority() -> bool {
    (unsafe { libc::geteuid() }) == 0
        || fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status
                    .lines()
                    .find_map(|line| line.strip_prefix("CapEff:\t"))
                    .and_then(|value| u64::from_str_radix(value, 16).ok())
            })
            .is_some_and(|capabilities| capabilities & (1 << 23) != 0)
}

fn classify_cpu_activity(previous: CpuSnapshot, current: CpuSnapshot) -> HostActivity {
    let total_delta = current.total.saturating_sub(previous.total);
    let idle_delta = current.idle.saturating_sub(previous.idle);

    if total_delta == 0 {
        return HostActivity::Unknown;
    }

    let active_delta = total_delta.saturating_sub(idle_delta);
    let active_percent = (active_delta.saturating_mul(100)) / total_delta;

    if active_percent >= CPU_ACTIVE_PERCENT_THRESHOLD {
        HostActivity::Active
    } else {
        HostActivity::Idle
    }
}

fn read_cpu_snapshot() -> Option<CpuSnapshot> {
    let stat = fs::read_to_string("/proc/stat").ok()?;
    let line = stat.lines().find(|line| line.starts_with("cpu "))?;

    let values = line
        .split_whitespace()
        .skip(1)
        .map(|value| value.parse::<u64>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;

    if values.len() < 5 {
        return None;
    }

    let total = values.iter().sum::<u64>();
    let idle = values[3] + values[4];

    Some(CpuSnapshot { total, idle })
}

fn read_process_nice(pid: u32) -> Option<i32> {
    let path = format!("/proc/{pid}/stat");
    let stat = fs::read_to_string(path).ok()?;
    let end_of_comm = stat.rfind(')')?;
    let fields = stat
        .get(end_of_comm + 2..)?
        .split_whitespace()
        .collect::<Vec<_>>();
    let nice = fields.get(16)?.parse::<i32>().ok()?;

    Some(nice)
}

#[cfg(target_os = "linux")]
fn set_process_nice(pid: u32, nice: i32) -> Result<(), ()> {
    if !(-20..=19).contains(&nice) {
        return Err(());
    }

    let result =
        unsafe { libc::setpriority(libc::PRIO_PROCESS, pid as libc::id_t, nice as libc::c_int) };

    if result == 0 { Ok(()) } else { Err(()) }
}
