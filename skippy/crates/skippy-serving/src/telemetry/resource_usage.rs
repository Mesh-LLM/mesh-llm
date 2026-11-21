//! Point-in-time readings of this process's and the system's CPU and memory
//! use, from the operating system. macOS reads Mach and libproc counters
//! (the footprint Activity Monitor shows, wired memory, page-ins, the memory
//! compressor and swap); Linux reads `getrusage` and `/proc`. Elsewhere there
//! are no readings.

use std::time::Duration;

/// Cumulative CPU time and current memory of this process.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct ProcessUsage {
    pub(super) user: Duration,
    pub(super) system: Duration,
    pub(super) resident_bytes: u64,
    /// Memory charged to the process; on macOS this includes Metal buffers.
    pub(super) footprint_bytes: u64,
    pub(super) wired_bytes: Option<u64>,
    /// Pages read in from disk since start: mapped weights being faulted in.
    pub(super) pageins: u64,
    pub(super) disk_read_bytes: Option<u64>,
}

/// Cumulative system paging, in pages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Paging {
    pub(super) pageins: u64,
    pub(super) pageouts: u64,
    pub(super) swapins: u64,
    pub(super) swapouts: u64,
}

/// System memory, by state.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct SystemMemory {
    pub(super) total_bytes: u64,
    /// `(state, bytes)`, such as `("free", ..)` or `("compressed", ..)`.
    pub(super) states: Vec<(&'static str, u64)>,
    pub(super) swap_used_bytes: u64,
    /// macOS memory pressure: 1 normal, 2 warning, 4 critical.
    pub(super) pressure_level: Option<u32>,
    pub(super) paging: Option<Paging>,
    pub(super) page_bytes: u64,
}

pub(super) fn process_usage() -> Option<ProcessUsage> {
    platform::process_usage()
}

pub(super) fn system_memory() -> Option<SystemMemory> {
    platform::system_memory()
}

/// User and system CPU time of this process, from `getrusage`.
#[cfg(unix)]
fn cpu_times() -> Option<(Duration, Duration)> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage fills the struct for RUSAGE_SELF.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: initialised by the successful call above.
    let usage = unsafe { usage.assume_init() };
    let time = |tv: libc::timeval| {
        Duration::from_secs(u64::try_from(tv.tv_sec).unwrap_or(0))
            + Duration::from_micros(u64::try_from(tv.tv_usec).unwrap_or(0))
    };
    Some((time(usage.ru_utime), time(usage.ru_stime)))
}

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::{CStr, c_void};
    use std::mem::{MaybeUninit, size_of};
    use std::sync::OnceLock;

    use super::{Paging, ProcessUsage, SystemMemory, cpu_times};

    pub(super) fn process_usage() -> Option<ProcessUsage> {
        let (user, system) = cpu_times()?;
        let mut info = MaybeUninit::<libc::rusage_info_v2>::zeroed();
        // SAFETY: proc_pid_rusage writes a rusage_info_v2 for RUSAGE_INFO_V2.
        let status = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V2,
                info.as_mut_ptr().cast(),
            )
        };
        if status != 0 {
            return Some(ProcessUsage {
                user,
                system,
                ..ProcessUsage::default()
            });
        }
        // SAFETY: initialised by the successful call above.
        let info = unsafe { info.assume_init() };
        Some(ProcessUsage {
            user,
            system,
            resident_bytes: info.ri_resident_size,
            footprint_bytes: info.ri_phys_footprint,
            wired_bytes: Some(info.ri_wired_size),
            pageins: info.ri_pageins,
            disk_read_bytes: Some(info.ri_diskio_bytesread),
        })
    }

    pub(super) fn system_memory() -> Option<SystemMemory> {
        let page_bytes = page_bytes();
        let mut stats = MaybeUninit::<libc::vm_statistics64>::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        // SAFETY: host_statistics64 writes at most `count` integers of stats.
        let status = unsafe {
            libc::host_statistics64(
                host_port(),
                libc::HOST_VM_INFO64,
                stats.as_mut_ptr().cast(),
                &mut count,
            )
        };
        if status != libc::KERN_SUCCESS {
            return None;
        }
        // SAFETY: initialised (up to what the kernel knows) by the call above.
        let stats = unsafe { stats.assume_init() };
        let bytes = |pages: u64| pages.saturating_mul(page_bytes);
        let swap = sysctl::<libc::xsw_usage>(c"vm.swapusage");
        Some(SystemMemory {
            total_bytes: sysctl::<u64>(c"hw.memsize").unwrap_or(0),
            states: vec![
                ("free", bytes(u64::from(stats.free_count))),
                ("active", bytes(u64::from(stats.active_count))),
                ("inactive", bytes(u64::from(stats.inactive_count))),
                ("wired", bytes(u64::from(stats.wire_count))),
                ("compressed", bytes(u64::from(stats.compressor_page_count))),
            ],
            swap_used_bytes: swap.map_or(0, |swap| swap.xsu_used),
            pressure_level: sysctl::<u32>(c"kern.memorystatus_vm_pressure_level"),
            paging: Some(Paging {
                pageins: stats.pageins,
                pageouts: stats.pageouts,
                swapins: stats.swapins,
                swapouts: stats.swapouts,
            }),
            page_bytes,
        })
    }

    unsafe extern "C" {
        // libc deprecates its binding in favour of the mach2 crate.
        fn mach_host_self() -> libc::mach_port_t;
    }

    /// The host port, taken once: each `mach_host_self` adds a port reference.
    fn host_port() -> libc::mach_port_t {
        static PORT: OnceLock<libc::mach_port_t> = OnceLock::new();
        // SAFETY: returns a send right to this host; no arguments.
        *PORT.get_or_init(|| unsafe { mach_host_self() })
    }

    fn page_bytes() -> u64 {
        // SAFETY: sysconf takes no pointers.
        u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(16_384)
    }

    /// A fixed-size sysctl value; `None` when absent or of another size.
    fn sysctl<T: Copy>(name: &CStr) -> Option<T> {
        let mut value = MaybeUninit::<T>::zeroed();
        let mut len = size_of::<T>();
        // SAFETY: sysctlbyname writes at most `len` bytes into `value`.
        let status = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                value.as_mut_ptr().cast::<c_void>(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        // SAFETY: fully written when the call succeeded with the full size.
        (status == 0 && len == size_of::<T>()).then(|| unsafe { value.assume_init() })
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::{ProcessUsage, SystemMemory, cpu_times};

    pub(super) fn process_usage() -> Option<ProcessUsage> {
        let (user, system) = cpu_times()?;
        let page_bytes = page_bytes();
        let resident_bytes = std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|statm| statm.split_whitespace().nth(1)?.parse::<u64>().ok())
            .map_or(0, |pages| pages.saturating_mul(page_bytes));
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: getrusage fills the struct for RUSAGE_SELF.
        let pageins = if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0 {
            // SAFETY: initialised by the successful call above.
            u64::try_from(unsafe { usage.assume_init() }.ru_majflt).unwrap_or(0)
        } else {
            0
        };
        Some(ProcessUsage {
            user,
            system,
            resident_bytes,
            footprint_bytes: resident_bytes,
            wired_bytes: None,
            pageins,
            disk_read_bytes: None,
        })
    }

    pub(super) fn system_memory() -> Option<SystemMemory> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let field = |name: &str| meminfo_bytes(&meminfo, name).unwrap_or(0);
        let total_bytes = field("MemTotal");
        let available = field("MemAvailable");
        Some(SystemMemory {
            total_bytes,
            states: vec![
                ("free", field("MemFree")),
                ("available", available),
                ("used", total_bytes.saturating_sub(available)),
            ],
            swap_used_bytes: field("SwapTotal").saturating_sub(field("SwapFree")),
            pressure_level: None,
            paging: None,
            page_bytes: page_bytes(),
        })
    }

    /// A `/proc/meminfo` field, which the kernel reports in KiB.
    pub(super) fn meminfo_bytes(meminfo: &str, name: &str) -> Option<u64> {
        meminfo.lines().find_map(|line| {
            let rest = line.strip_prefix(name)?.strip_prefix(':')?;
            let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            Some(kib.saturating_mul(1024))
        })
    }

    fn page_bytes() -> u64 {
        // SAFETY: sysconf takes no pointers.
        u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(4096)
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod platform {
    use super::{ProcessUsage, SystemMemory};

    pub(super) fn process_usage() -> Option<ProcessUsage> {
        None
    }

    pub(super) fn system_memory() -> Option<SystemMemory> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn this_process_has_cpu_time_and_resident_memory() {
        let usage = process_usage().expect("process usage");
        assert!(usage.resident_bytes > 0);
        assert!(usage.footprint_bytes > 0);
        let busy = std::time::Instant::now();
        while busy.elapsed() < Duration::from_millis(20) {
            std::hint::black_box(busy.elapsed());
        }
        let later = process_usage().expect("process usage");
        assert!(later.user + later.system > usage.user + usage.system);
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn the_system_reports_total_and_free_memory() {
        let memory = system_memory().expect("system memory");
        assert!(memory.total_bytes > 0);
        let free = memory
            .states
            .iter()
            .find(|(state, _)| *state == "free")
            .map(|(_, bytes)| *bytes)
            .expect("free memory");
        assert!(free <= memory.total_bytes);
        assert!(memory.page_bytes >= 4096);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn meminfo_fields_are_read_in_kib() {
        let meminfo = "MemTotal:       16384 kB\nMemFree:  2048 kB\nMemAvailableX: 1 kB\n";
        assert_eq!(
            platform::meminfo_bytes(meminfo, "MemTotal"),
            Some(16_384 * 1024)
        );
        assert_eq!(
            platform::meminfo_bytes(meminfo, "MemFree"),
            Some(2048 * 1024)
        );
        assert_eq!(platform::meminfo_bytes(meminfo, "MemAvailable"), None);
    }
}
