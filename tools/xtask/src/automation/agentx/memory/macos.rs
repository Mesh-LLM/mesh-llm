//! Process memory from Darwin's libproc; neither field is GPU residency.
use super::Sample;
use std::mem::MaybeUninit;

pub(super) fn sample(pid: u32, elapsed_ms: u128) -> Sample {
    let usage = usage(pid);
    Sample {
        elapsed_ms,
        process_rss_bytes: usage.as_ref().map(|value| value.ri_resident_size),
        process_physical_footprint_bytes: usage.as_ref().map(|value| value.ri_phys_footprint),
        // rusage_info_v2 exposes instantaneous values, not an RSS high-water mark.
        ..Sample::default()
    }
}

fn usage(pid: u32) -> Option<libc::rusage_info_v2> {
    let pid = i32::try_from(pid).ok().filter(|pid| *pid > 0)?;
    let mut usage = MaybeUninit::<libc::rusage_info_v2>::uninit();
    // SAFETY: the buffer has the layout/size selected by RUSAGE_INFO_V2. libproc
    // initializes it on success; failures never read the uninitialized buffer.
    let status =
        unsafe { libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V2, usage.as_mut_ptr().cast()) };
    if status != 0 {
        return None;
    }
    // SAFETY: the successful call above initialized the complete v2 structure.
    Some(unsafe { usage.assume_init() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observes_live_process_memory_without_claiming_high_water_or_gpu_residency() {
        let sample = sample(std::process::id(), 123);
        assert!(sample.process_rss_bytes.is_some_and(|bytes| bytes > 0));
        assert!(
            sample
                .process_physical_footprint_bytes
                .is_some_and(|bytes| bytes > 0)
        );
        assert_eq!(sample.elapsed_ms, 123);
        assert_eq!(sample.process_rss_high_water_bytes, None);
        assert_eq!(sample.container_memory_bytes, None);
    }

    #[test]
    fn unavailable_process_is_missing_not_zero() {
        for pid in [0, u32::MAX, i32::MAX as u32] {
            let sample = sample(pid, 0);
            assert_eq!(sample.process_rss_bytes, None);
            assert_eq!(sample.process_physical_footprint_bytes, None);
        }
    }
}
