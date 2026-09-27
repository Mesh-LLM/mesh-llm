use super::super::*;
use std::{
    env,
    ffi::CString,
    fs,
    io::Write,
    ptr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

struct FlushCountingWriter {
    flush_count: Arc<AtomicUsize>,
}

impl Write for FlushCountingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flush_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn native_log_writer_flush_helper_explicitly_flushes_line_writer() {
    let flush_count = Arc::new(AtomicUsize::new(0));
    let writer = FlushCountingWriter {
        flush_count: flush_count.clone(),
    };
    let mut writer = Some(LineWriter::new(writer));
    writer
        .as_mut()
        .expect("writer should exist")
        .write_all(b"buffered native log line\n")
        .expect("write to buffered test writer should succeed");

    flush_native_log_writer(&mut writer);

    assert_eq!(flush_count.load(Ordering::SeqCst), 1);
}

#[test]
fn native_log_writer_flushes_newline_and_partial_line() -> anyhow::Result<()> {
    let _native_log_guard = native_log_test_guard();

    struct RestoreNativeLogs;

    impl Drop for RestoreNativeLogs {
        fn drop(&mut self) {
            restore_native_logs();
        }
    }

    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = env::temp_dir().join(format!(
        "skippy-native-log-buffer-test-{}-{nanos}.log",
        std::process::id()
    ));
    let _guard = RestoreNativeLogs;
    redirect_native_logs_to_file(&path)?;

    let message = CString::new("buffered native log line\n")?;
    unsafe {
        write_native_log(0, message.as_ptr(), ptr::null_mut());
    }

    let contents = fs::read_to_string(&path)?;
    restore_native_logs();

    fs::remove_file(&path)?;
    assert_eq!(contents, "buffered native log line\n");

    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = env::temp_dir().join(format!(
        "skippy-native-log-partial-line-test-{}-{nanos}.log",
        std::process::id()
    ));
    let _guard = RestoreNativeLogs;
    redirect_native_logs_to_file(&path)?;

    let message = CString::new("partial native log line")?;
    unsafe {
        write_native_log(0, message.as_ptr(), ptr::null_mut());
    }
    restore_native_logs();

    let contents = fs::read_to_string(&path)?;
    fs::remove_file(&path)?;
    assert_eq!(contents, "partial native log line");
    Ok(())
}

#[test]
fn native_log_note_writes_sanitized_flushed_context() -> anyhow::Result<()> {
    let _native_log_guard = native_log_test_guard();

    struct RestoreNativeLogs;

    impl Drop for RestoreNativeLogs {
        fn drop(&mut self) {
            restore_native_logs();
        }
    }

    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = env::temp_dir().join(format!(
        "skippy-native-log-note-test-{}-{nanos}.log",
        std::process::id()
    ));
    let _guard = RestoreNativeLogs;
    redirect_native_logs_to_file(&path)?;

    write_native_log_note("native call begin\nwith context");

    let contents = fs::read_to_string(&path)?;
    restore_native_logs();
    fs::remove_file(&path)?;

    assert!(
        contents.ends_with("mesh-llm: native call begin with context\n"),
        "unexpected native log contents: {contents:?}"
    );
    assert!(
        !contents.contains("native call begin\nwith context"),
        "native log note was not sanitized: {contents:?}"
    );
    Ok(())
}
