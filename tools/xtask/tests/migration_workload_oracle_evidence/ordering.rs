use super::super::{Error, verify_evidence, write_evidence};
use super::fixture::Fixture;
use std::fs;

#[test]
fn writer_rejects_lane_then_log_then_hashes_before_tts_and_output() {
    let mut fixture = Fixture::new();
    fixture.select("speech_synthesis", "llama-tts");
    fixture.write.smoke_lane = "bad".into();
    fs::remove_file(&fixture.write.comparison_log).unwrap();
    fs::remove_file(&fixture.write.oracle_executable).unwrap();
    fs::write(&fixture.write.output, b"RETAINED").unwrap();
    assert!(matches!(
        write_evidence(&fixture.write),
        Err(Error::SmokeLane(_))
    ));

    fixture.write.smoke_lane = "speech-smoke".into();
    assert!(
        matches!(write_evidence(&fixture.write), Err(Error::Io { path, .. }) if path == fixture.write.comparison_log)
    );
    fs::write(&fixture.write.comparison_log, "bad").unwrap();
    assert!(matches!(
        write_evidence(&fixture.write),
        Err(Error::ComparatorLog)
    ));
    fs::write(
        &fixture.write.comparison_log,
        "speech_synthesis local-monolithic oracle passed: fixture",
    )
    .unwrap();
    assert!(
        matches!(write_evidence(&fixture.write), Err(Error::Io { path, .. }) if path == fixture.write.oracle_executable)
    );
    assert_eq!(fs::read(&fixture.write.output).unwrap(), b"RETAINED");
}

#[test]
fn verifier_checks_prerequisites_parse_hashes_identity_basename_and_metrics_in_order() {
    let mut fixture = Fixture::new();
    fixture.select("speech_synthesis", "wrong-reference");
    fs::write(&fixture.write.output, b"{").unwrap();
    assert!(matches!(
        verify_evidence(&fixture.verify()),
        Err(Error::Projector(_))
    ));
    fixture.projector();
    assert!(matches!(
        verify_evidence(&fixture.verify()),
        Err(Error::Json(_))
    ));
    let mut body = fixture.body();
    body["status"] = "fail".into();
    fixture.save(&body);
    fs::remove_file(&fixture.model).unwrap();
    assert!(
        matches!(verify_evidence(&fixture.verify()), Err(Error::Io { path, .. }) if path == fixture.model)
    );
    fs::write(&fixture.model, b"abc").unwrap();
    assert!(matches!(
        verify_evidence(&fixture.verify()),
        Err(Error::Identity("status"))
    ));
    body["status"] = "pass".into();
    body["oracle_executable"] = "llama-tts".into();
    body["comparison"] = "bad".into();
    fixture.save(&body);
    assert!(matches!(
        verify_evidence(&fixture.verify()),
        Err(Error::Executable)
    ));
    fixture.select("speech_synthesis", "llama-tts");
    assert!(matches!(
        verify_evidence(&fixture.verify()),
        Err(Error::Comparison)
    ));
    body["comparison"] = "speech_synthesis local-monolithic oracle passed: fixture".into();
    fixture.save(&body);
    assert!(matches!(
        verify_evidence(&fixture.verify()),
        Err(Error::Metrics)
    ));
}
