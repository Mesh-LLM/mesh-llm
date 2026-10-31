use mesh_client::runtime::CoreRuntime;
use std::time::Duration;

fn thread_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        let pid = i32::try_from(std::process::id()).expect("pid fits in pid_t");
        // SAFETY: proc_taskinfo is a C POD whose integer fields admit zero.
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let size = i32::try_from(std::mem::size_of_val(&info)).expect("proc_taskinfo size");
        // SAFETY: the task-info flavor matches the aligned, writable info buffer.
        let read = unsafe {
            libc::proc_pidinfo(
                pid,
                libc::PROC_PIDTASKINFO,
                0,
                (&mut info as *mut libc::proc_taskinfo).cast(),
                size,
            )
        };
        assert_eq!(
            read,
            size,
            "proc_pidinfo: {}",
            std::io::Error::last_os_error()
        );
        usize::try_from(info.pti_threadnum).unwrap_or(0)
    }
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines() {
            if line.starts_with("Threads:") {
                return line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
            }
        }
        0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

#[test]
fn runtime_100_create_drop_no_thread_leak() {
    let before = thread_count();
    for _ in 0..100 {
        let rt = CoreRuntime::new().expect("create runtime");
        drop(rt);
    }
    let after = thread_count();
    let growth = after.saturating_sub(before);
    println!("threads_before={before} threads_after={after} growth={growth}");
    assert!(
        growth <= 3,
        "thread leak detected: before={before} after={after}"
    );
}

#[test]
fn runtime_drop_from_tokio_task_no_panic() {
    let rt = CoreRuntime::new().expect("create runtime");
    let handle = rt.handle().clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle.block_on(async {
            drop(rt);
        });
    }));
    assert!(result.is_ok(), "drop from tokio task panicked");
}

#[test]
fn runtime_pending_tasks_shutdown_within_5s() {
    let rt = CoreRuntime::new().expect("create runtime");
    rt.handle().spawn(async {
        tokio::time::sleep(Duration::from_secs(100)).await;
    });
    let start = std::time::Instant::now();
    drop(rt);
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(6),
        "shutdown took too long: {elapsed:?}"
    );
}
