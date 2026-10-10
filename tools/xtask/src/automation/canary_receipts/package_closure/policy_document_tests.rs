use super::*;

#[test]
fn opened_policy_reader_rejects_concurrent_replacement_and_in_place_growth() {
    for replace in [true, false] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let path = root.join("policy.json");
        fs::write(&path, b"original").unwrap();
        let mut file = fs::File::open(&path).unwrap();
        let opened = file.metadata().unwrap();
        let writer_path = path.clone();
        // A separate writer races the admitted handle, with a join fixing the
        // admission/read boundary so failure does not depend on scheduler luck.
        std::thread::spawn(move || {
            if replace {
                let replacement = writer_path.with_extension("replacement");
                fs::write(&replacement, b"replaced").unwrap();
                fs::rename(replacement, writer_path).unwrap();
            } else {
                fs::write(writer_path, b"original extended").unwrap();
            }
        })
        .join()
        .unwrap();
        assert!(read_opened(&root, &path, &mut file, &opened).is_err());
    }
}

#[test]
fn opened_policy_reader_honors_cancellation_without_mutating_source() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let path = root.join("policy.json");
    fs::write(&path, b"unchanged").unwrap();
    let mut file = fs::File::open(&path).unwrap();
    let opened = file.metadata().unwrap();
    let outcome = process::operation(|| {
        process::cancellation().cancel();
        assert!(read_opened(&root, &path, &mut file, &opened).is_err());
        Ok(())
    });
    // The final Interrupt::finish must also report the cancelled operation.
    assert!(outcome.is_err());
    assert_eq!(fs::read(path).unwrap(), b"unchanged");
}
