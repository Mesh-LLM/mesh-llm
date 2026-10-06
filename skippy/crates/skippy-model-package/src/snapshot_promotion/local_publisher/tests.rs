use super::*;
#[test]
fn publisher_terminal_cancel_deadline_or_input_drift_retains_failed_commit_observations() {
    for mode in 0..3 {
        let root = tempfile::tempdir().unwrap();
        let base = root.path().canonicalize().unwrap();
        let path = base.join("input");
        std::fs::write(&path, b"original").unwrap();
        let mut file = regular_input::open(&path, 512 * 1024, false).unwrap();
        let hash = digest(b"original");
        let output = output::Output::fresh(&base.join("out")).unwrap();
        let until = if mode == 1 {
            Instant::now()
        } else {
            Instant::now() + Duration::from_secs(10)
        };
        if mode == 2 {
            std::fs::write(&path, b"changed").unwrap();
        }
        let publication = super::super::model_publication::Receipt {
            schema_version: 1,
            repo: "fixture/model".into(),
            parent_commit: "a".repeat(40),
            ordered_paths: vec!["model.gguf".into()],
            objects: vec![],
            object_attempted_paths: vec!["model.gguf".into()],
            commit_attempted: true,
            commit_oid: Some("c".repeat(40)),
            remote_verified_paths: vec!["model.gguf".into()],
            final_source_custody_verified: true,
            completed: true,
            error: None,
        };
        let mut terminal = Finalization {
            output: &output,
            hash: &hash,
            input_file: &mut file,
            raw_hash: &hash,
            until,
        };
        assert!(
            !terminal
                .finish(Operation::Publish, true, Some(publication), || mode == 0)
                .unwrap()
        );
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(base.join("out/publication.json")).unwrap())
                .unwrap();
        assert_eq!(value["status"], "FAILED");
        assert_eq!(value["publication"]["completed"], false);
        assert_eq!(value["publication"]["commit_oid"], "c".repeat(40));
        assert_eq!(
            value["publication"]["remote_verified_paths"][0],
            "model.gguf"
        );
        drop(file);
        root.close().unwrap();
    }
}

#[test]
fn publisher_terminal_self_signal_without_tokio_driver_retains_failed_commit_observations() {
    for signal in [libc::SIGTERM, libc::SIGINT] {
        let root = tempfile::tempdir().unwrap();
        let base = root.path().canonicalize().unwrap();
        let path = base.join("input");
        std::fs::write(&path, b"original").unwrap();
        let mut file = regular_input::open(&path, 512 * 1024, false).unwrap();
        let hash = digest(b"original");
        let output = output::Output::fresh(&base.join("out")).unwrap();
        let latch = SignalLatch::install().unwrap();
        let publication = super::super::model_publication::Receipt {
            schema_version: 1,
            repo: "fixture/model".into(),
            parent_commit: "a".repeat(40),
            ordered_paths: vec!["model.gguf".into()],
            objects: vec![],
            object_attempted_paths: vec!["model.gguf".into()],
            commit_attempted: true,
            commit_oid: Some("c".repeat(40)),
            remote_verified_paths: vec!["model.gguf".into()],
            final_source_custody_verified: true,
            completed: true,
            error: None,
        };
        let mut finalization = Finalization {
            output: &output,
            hash: &hash,
            input_file: &mut file,
            raw_hash: &hash,
            until: Instant::now() + Duration::from_secs(10),
        };
        assert!(
            !finalization
                .finish(Operation::Publish, true, Some(publication), || {
                    // SAFETY: raise targets only the calling thread inside this owned fixture;
                    // its dedicated registry callback has already been installed.
                    assert_eq!(unsafe { libc::raise(signal) }, 0);
                    latch.cancelled()
                })
                .unwrap()
        );
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(base.join("out/publication.json")).unwrap())
                .unwrap();
        assert_eq!(value["status"], "FAILED");
        assert_eq!(value["publication"]["completed"], false);
        assert_eq!(value["publication"]["commit_oid"], "c".repeat(40));
        assert!(latch.cancelled());
        drop(latch);
        drop(file);
        root.close().unwrap();
    }
}
