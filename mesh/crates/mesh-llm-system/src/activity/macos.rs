use std::process;

use super::{HostActivity, HostActivityDetector, PriorityController, PriorityFailure};

const HID_ACTIVE_IDLE_NANOS: u64 = 1_000_000_000;

#[derive(Debug, Default)]
pub struct MacHostActivityDetector {
    last_idle_nanos: Option<u64>,
}

impl HostActivityDetector for MacHostActivityDetector {
    fn sample(&mut self) -> HostActivity {
        let current = match read_hid_idle_nanos() {
            Some(value) => value,
            None => {
                self.last_idle_nanos = None;
                return HostActivity::Unknown;
            }
        };

        let activity = classify_hid_idle_nanos(current);
        self.last_idle_nanos = Some(current);
        activity
    }
}

fn classify_hid_idle_nanos(idle_nanos: u64) -> HostActivity {
    if idle_nanos < HID_ACTIVE_IDLE_NANOS {
        HostActivity::Active
    } else {
        HostActivity::Idle
    }
}

#[derive(Debug)]
pub struct MacPriorityController {
    #[cfg(target_os = "macos")]
    pid: u32,
    #[cfg(target_os = "macos")]
    original_nice: Option<i32>,
    #[cfg(not(target_os = "macos"))]
    _marker: (),
}

impl Default for MacPriorityController {
    fn default() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self {
                pid: process::id(),
                original_nice: None,
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self { _marker: () }
        }
    }
}

impl PriorityController for MacPriorityController {
    fn reduce_priority(&mut self) -> Result<(), PriorityFailure> {
        #[cfg(target_os = "macos")]
        {
            let current = read_process_nice(self.pid).ok_or(PriorityFailure::ApplyFailed)?;

            if self.original_nice.is_none() {
                let target = (current + 5).clamp(-20, 19);
                set_process_nice(self.pid, target).map_err(|_| PriorityFailure::ApplyFailed)?;
                self.original_nice = Some(current);
            }

            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err(PriorityFailure::Unsupported)
        }
    }

    fn restore_priority(&mut self) -> Result<(), PriorityFailure> {
        #[cfg(target_os = "macos")]
        {
            let Some(original_nice) = self.original_nice else {
                return Ok(());
            };

            set_process_nice(self.pid, original_nice)
                .map_err(|_| PriorityFailure::RestoreFailed)?;
            self.original_nice = None;

            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err(PriorityFailure::Unsupported)
        }
    }
}

#[cfg(target_os = "macos")]
fn read_hid_idle_nanos() -> Option<u64> {
    let output = process::Command::new("ioreg")
        .args(["-c", "IOHIDSystem"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_hid_idle_nanos(&String::from_utf8_lossy(&output.stdout))
}

fn parse_hid_idle_nanos(output: &str) -> Option<u64> {
    output.lines().find_map(|line| {
        if !line.contains("\"HIDIdleTime\"") {
            return None;
        }
        line.split_once('=')
            .and_then(|(_, value)| value.trim().parse().ok())
    })
}

#[cfg(target_os = "macos")]
fn read_process_nice(pid: u32) -> Option<i32> {
    unsafe {
        *libc::__error() = 0;
        let priority = libc::getpriority(libc::PRIO_PROCESS, pid);
        (*libc::__error() == 0).then_some(priority)
    }
}

#[cfg(target_os = "macos")]
fn set_process_nice(pid: u32, nice: i32) -> Result<(), ()> {
    (unsafe { libc::setpriority(libc::PRIO_PROCESS, pid, nice) } == 0)
        .then_some(())
        .ok_or(())
}

#[cfg(test)]
mod tests {
    use super::{
        HID_ACTIVE_IDLE_NANOS, HostActivity, classify_hid_idle_nanos, parse_hid_idle_nanos,
    };

    #[test]
    fn parses_hid_idle_time() {
        assert_eq!(
            parse_hid_idle_nanos(r#"    "HIDIdleTime" = 123456789"#),
            Some(123_456_789)
        );
        assert_eq!(parse_hid_idle_nanos("unavailable"), None);
    }

    #[test]
    fn classifies_activity_from_absolute_idle_duration() {
        assert_eq!(classify_hid_idle_nanos(0), HostActivity::Active);
        assert_eq!(
            classify_hid_idle_nanos(HID_ACTIVE_IDLE_NANOS - 1),
            HostActivity::Active
        );
        assert_eq!(
            classify_hid_idle_nanos(HID_ACTIVE_IDLE_NANOS),
            HostActivity::Idle
        );
        assert_eq!(
            classify_hid_idle_nanos(HID_ACTIVE_IDLE_NANOS + 1),
            HostActivity::Idle
        );
    }
}
