//! Retain bounded native failure evidence before the smoke's private state is deleted.

use crate::process::OutputFiles;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const TAIL_BYTES: u64 = 12_000;
const MAX_RUNTIME_ENTRIES: usize = 8;

pub(super) fn collect(root: &Path, output: &OutputFiles) -> String {
    let mut diagnostics = String::new();
    for (label, path) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
        if let Some(path) = path {
            append_tail(&mut diagnostics, label, path);
        }
    }
    let runtime = root.join("runtime");
    if !is_directory(&runtime) {
        return diagnostics;
    }
    let Ok(instances) = fs::read_dir(runtime) else {
        diagnostics.push_str("\nnative logs: runtime directory could not be read\n");
        return diagnostics;
    };
    for (index, instance) in instances.take(MAX_RUNTIME_ENTRIES + 1).enumerate() {
        if index == MAX_RUNTIME_ENTRIES {
            diagnostics.push_str("\nnative logs: runtime scan limit reached\n");
            break;
        }
        let Ok(instance) = instance else { continue };
        let path = instance.path();
        let logs = path.join("logs");
        if is_directory(&path) && is_directory(&logs) {
            append_tail(
                &mut diagnostics,
                "native log",
                &logs.join("skippy-native.log"),
            );
        }
    }
    diagnostics
}

fn is_directory(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_dir())
}

fn append_tail(diagnostics: &mut String, label: &str, path: &Path) {
    // The retained-session owner has stopped the process tree before collection.
    // Ignore links and special files rather than following them out of its state.
    let result = (|| -> std::io::Result<Option<Vec<u8>>> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => return Ok(None),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let mut log = fs::File::open(path)?;
        log.seek(SeekFrom::Start(metadata.len().saturating_sub(TAIL_BYTES)))?;
        let mut bytes = Vec::new();
        log.take(TAIL_BYTES).read_to_end(&mut bytes)?;
        Ok(Some(bytes))
    })();
    match result {
        Ok(Some(bytes)) => {
            diagnostics.push_str(&format!("\n--- {label} (tail) ---\n"));
            diagnostics.push_str(&String::from_utf8_lossy(&bytes));
        }
        Ok(None) => {}
        Err(error) => {
            // A diagnostic read error must not replace the process failure.
            diagnostics.push_str(&format!(
                "\n{label}: diagnostic unavailable ({:?})\n",
                error.kind()
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_failure_tail_is_bounded_and_other_runtime_files_are_excluded() {
        let state = tempfile::tempdir().unwrap();
        let logs = state.path().join("runtime/instance/logs");
        fs::create_dir_all(&logs).unwrap();
        let mut bytes = vec![b'x'; TAIL_BYTES as usize + 17];
        bytes.extend_from_slice(b"cudaMalloc failed: fixture error\n");
        fs::write(logs.join("skippy-native.log"), bytes).unwrap();
        fs::write(logs.join("unrelated.log"), "unrelated private content").unwrap();
        let diagnostics = collect(state.path(), &OutputFiles::default());
        assert!(diagnostics.ends_with("cudaMalloc failed: fixture error\n"));
        assert!(!diagnostics.contains("unrelated private content"));
        assert_eq!(
            diagnostics.len(),
            TAIL_BYTES as usize + "\n--- native log (tail) ---\n".len()
        );
    }

    #[test]
    fn absent_native_logs_preserve_stderr() {
        let state = tempfile::tempdir().unwrap();
        let stderr = state.path().join("stderr.log");
        fs::write(&stderr, "original startup failure").unwrap();
        let output = OutputFiles {
            stdout: None,
            stderr: Some(stderr),
        };
        assert!(collect(state.path(), &output).contains("original startup failure"));
    }

    #[test]
    fn unreadable_diagnostic_does_not_hide_other_failure_output() {
        let state = tempfile::tempdir().unwrap();
        let stderr = state.path().join("stderr.log");
        fs::write(&stderr, "original startup failure").unwrap();
        let output = OutputFiles {
            stdout: Some(state.path().join("invalid\0path")),
            stderr: Some(stderr),
        };
        let diagnostics = collect(state.path(), &output);
        assert!(diagnostics.contains("stdout: diagnostic unavailable"));
        assert!(diagnostics.contains("original startup failure"));
    }

    #[test]
    fn native_log_collection_bounds_the_number_of_runtime_entries() {
        let state = tempfile::tempdir().unwrap();
        for index in 0..=MAX_RUNTIME_ENTRIES {
            let logs = state.path().join(format!("runtime/{index}/logs"));
            fs::create_dir_all(&logs).unwrap();
            fs::write(logs.join("skippy-native.log"), "native failure\n").unwrap();
        }
        let diagnostics = collect(state.path(), &OutputFiles::default());
        assert_eq!(
            diagnostics.matches("native failure").count(),
            MAX_RUNTIME_ENTRIES
        );
        assert!(diagnostics.contains("runtime scan limit reached"));
    }

    #[cfg(unix)]
    #[test]
    fn native_log_symlinks_are_not_followed() {
        use std::os::unix::fs::symlink;
        let state = tempfile::tempdir().unwrap();
        let logs = state.path().join("runtime/instance/logs");
        fs::create_dir_all(&logs).unwrap();
        let outside = state.path().join("outside");
        fs::create_dir_all(outside.join("logs")).unwrap();
        fs::write(outside.join("skippy-native.log"), "must not be read").unwrap();
        fs::write(outside.join("logs/skippy-native.log"), "must not be read").unwrap();
        symlink(
            outside.join("skippy-native.log"),
            logs.join("skippy-native.log"),
        )
        .unwrap();
        symlink(&outside, state.path().join("runtime/linked-instance")).unwrap();
        let other = state.path().join("runtime/other");
        fs::create_dir(&other).unwrap();
        symlink(&outside, other.join("logs")).unwrap();
        assert!(collect(state.path(), &OutputFiles::default()).is_empty());
    }
}
