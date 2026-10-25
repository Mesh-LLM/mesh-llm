use super::*;

#[test]
fn ci_metrics_transport_reads_valid_bytes_and_refuses_oversize_before_allocation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("runs.json");
    fs::write(&path, b"[]").unwrap();
    assert_eq!(read_file(&path, &Cancellation::default()).unwrap(), b"[]");
    File::create(&path)
        .unwrap()
        .set_len(MAX_INPUT_BYTES + 1)
        .unwrap();
    assert!(read_file(&path, &Cancellation::default()).is_err());
}

#[test]
fn ci_metrics_transport_cancellation_precedes_missing_path_probe() {
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let error = read_file(Path::new("missing-input.json"), &cancellation).unwrap_err();
    assert!(matches!(error, Failure::Reported(message) if message == "metrics input cancelled"));
}

#[test]
fn ci_metrics_transport_rejects_growth_and_truncation_of_open_handle() {
    for new_bytes in [b"longer document".as_slice(), b"x".as_slice()] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("runs.json");
        fs::write(&path, b"[]").unwrap();
        let mut file = File::open(&path).unwrap();
        let opened = file.metadata().unwrap();
        fs::write(&path, new_bytes).unwrap();
        assert!(read_opened(&mut file, &path, &opened, &Cancellation::default()).is_err());
    }
}

#[cfg(unix)]
#[test]
fn ci_metrics_transport_refuses_fifo_socket_directory_and_symlink_without_opening() {
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::symlink, net::UnixListener},
    };
    let dir = tempfile::tempdir().unwrap();
    let fifo = dir.path().join("fifo");
    let name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let socket = dir.path().join("socket");
    let _listener = UnixListener::bind(&socket).unwrap();
    let regular = dir.path().join("regular");
    fs::write(&regular, b"[]").unwrap();
    let link = dir.path().join("link");
    symlink(&regular, &link).unwrap();
    for path in [&fifo, &socket, dir.path(), &link] {
        assert!(read_file(path, &Cancellation::default()).is_err());
    }
}
