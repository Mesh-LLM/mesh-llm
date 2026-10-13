//! Actual publication ownership: identical bytes from a distinct atomic writer are not ours.
use super::{
    dirty_file,
    observation::Observation,
    provenance,
    publication::{Phase, Prepared},
};
use crate::process::Cancellation;
use std::{fs, io::Write, path::Path, time::Duration};
fn observation() -> Observation {
    Observation::new(Cancellation::default(), Duration::from_secs(10)).unwrap()
}
fn replacement(path: &Path, bytes: &[u8]) -> fs::File {
    let mut file = tempfile::NamedTempFile::new_in(path.parent().unwrap()).unwrap();
    file.write_all(bytes).unwrap();
    file.as_file().sync_all().unwrap();
    file.persist(path).unwrap()
}
#[test]
fn release_inventory_publication_same_bytes_atomic_writer_before_replacement_is_preserved() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("report.json");
    fs::write(&path, b"old bytes").unwrap();
    let scope = observation();
    let prepared = Prepared::stage(&path, b"new report", &scope).unwrap();
    let writer = replacement(&path, b"old bytes");
    let expected = dirty_file::report_identity(&writer).unwrap();
    assert!(prepared.publish(&scope, |_, _| Ok(())).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"old bytes");
    assert_eq!(
        dirty_file::report_state(root.path(), "report.json", &observation(), 100)
            .unwrap()
            .1,
        expected
    );
}
#[test]
fn release_inventory_publication_same_bytes_atomic_writer_survives_failure_and_cancellation_rollback()
 {
    for prior in [true, false] {
        for cancel in [true, false] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("report.json");
            if prior {
                fs::write(&path, b"old bytes").unwrap();
            }
            let scope = observation();
            let prepared = Prepared::stage(&path, b"new report", &scope).unwrap();
            let mut writer_identity = None;
            let result = prepared.publish(&scope, |phase, _| match phase {
                Phase::BeforeReplacement => Ok(()),
                Phase::AfterReplacement => {
                    let writer = replacement(&path, b"new report");
                    writer_identity = Some(dirty_file::report_identity(&writer).unwrap());
                    if cancel {
                        scope.cancellation.cancel();
                        scope.check()
                    } else {
                        Err(provenance::Error("source identity changed".into()))
                    }
                }
            });
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("refuses unsafe rollback")
            );
            assert_eq!(fs::read(&path).unwrap(), b"new report");
            assert_eq!(
                dirty_file::report_state(root.path(), "report.json", &observation(), 100)
                    .unwrap()
                    .1,
                writer_identity.unwrap()
            );
            assert_eq!(
                fs::read_dir(root.path()).unwrap().count(),
                1,
                "private stage/backup removed"
            );
        }
    }
}
