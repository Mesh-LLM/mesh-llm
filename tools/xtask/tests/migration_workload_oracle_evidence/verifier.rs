use super::super::{WorkloadClass, verify_evidence};
use super::fixture::{Fixture, METRICS};
use std::fs;

#[test]
fn evidence_passes_when_each_class_uses_its_source_mapped_executable() {
    for (class, executable, projector) in [
        ("embedding", "llama-server", false),
        ("rerank", "llama-server", false),
        ("encoder_decoder", "llama-completion", false),
        ("ocr", "llama-server", true),
        ("speech_synthesis", "llama-tts", true),
        ("speech_recognition", "llama-server", true),
    ] {
        let mut fixture = Fixture::new();
        fixture.select(class, executable);
        if projector {
            fixture.projector();
        }
        let mut body = fixture.body();
        if class == "speech_synthesis" {
            body["metrics"] = serde_json::from_str(METRICS).unwrap();
        }
        fixture.save(&body);

        let result = verify_evidence(&fixture.verify());

        assert!(result.is_ok(), "{class}: {result:?}");
    }
}

#[test]
fn evidence_is_rejected_when_any_identity_field_changes() {
    for field in [
        "status",
        "class",
        "smoke_lane",
        "oracle_lane",
        "model_id",
        "model_sha256",
        "projector_sha256",
        "oracle_executable",
        "oracle_executable_sha256",
        "candidate_executable_sha256",
        "pinned_patch_sha",
    ] {
        let fixture = Fixture::new();
        let mut body = fixture.body();
        body[field] = "different".into();
        fixture.save(&body);

        let result = verify_evidence(&fixture.verify());

        assert!(result.is_err(), "{field}");
    }
}

#[test]
fn evidence_is_rejected_when_independent_artifact_bytes_change() {
    for artifact in ["model", "candidate", "oracle", "projector"] {
        let mut fixture = Fixture::new();
        fixture.projector();
        fixture.save(&fixture.body());
        let path = match artifact {
            "model" => &fixture.model,
            "candidate" => &fixture.write.candidate_executable,
            "oracle" => &fixture.write.oracle_executable,
            "projector" => fixture.write.projector_path.as_ref().unwrap(),
            _ => unreachable!(),
        };
        fs::write(path, b"changed bytes").unwrap();

        let result = verify_evidence(&fixture.verify());

        assert!(result.is_err(), "{artifact}");
    }
}

#[test]
fn projector_is_required_when_class_has_a_sidecar_contract() {
    for class in ["ocr", "speech_synthesis", "speech_recognition"] {
        let mut fixture = Fixture::new();
        fixture.select(class, "llama-server");
        fixture.save(&fixture.body());

        let result = verify_evidence(&fixture.verify());

        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires a projector path")
        );
    }
}

#[test]
fn executable_basename_is_checked_when_digest_is_unchanged() {
    let mut fixture = Fixture::new();
    fixture.write.oracle_executable = fixture.directory.path().join("renamed-reference");
    fs::write(&fixture.write.oracle_executable, b"hello").unwrap();
    let mut body = fixture.body();
    body["oracle_executable"] = "llama-server".into();
    fixture.save(&body);

    let result = verify_evidence(&fixture.verify());

    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("wrong oracle executable")
    );
}

#[test]
fn comparison_is_rejected_when_not_an_explicit_class_specific_pass() {
    for comparison in [
        serde_json::Value::Null,
        true.into(),
        "embedding OpenAI HTTP smoke passed".into(),
        "rerank local-monolithic oracle passed: fixture".into(),
        " embedding local-monolithic oracle passed: fixture".into(),
    ] {
        let fixture = Fixture::new();
        let mut body = fixture.body();
        body["comparison"] = comparison;
        fixture.save(&body);

        assert!(verify_evidence(&fixture.verify()).is_err());
    }
}

#[test]
fn independent_lane_labels_are_accepted_when_they_match_supplied_inputs() {
    let fixture = Fixture::new();
    let mut request = fixture.verify();
    request.smoke_lane = "not-a-smoke-suffix".into();
    request.oracle_lane = "independent-lane".into();
    request.pinned_patch_sha = "opaque-pin".into();
    let mut body = fixture.body();
    body["smoke_lane"] = request.smoke_lane.clone().into();
    body["oracle_lane"] = request.oracle_lane.clone().into();
    body["pinned_patch_sha"] = request.pinned_patch_sha.clone().into();
    fixture.save(&body);

    assert!(verify_evidence(&request).is_ok());
}

#[test]
fn absent_projector_and_ignored_fields_pass_when_no_sidecar_is_supplied() {
    let fixture = Fixture::new();
    let mut body = fixture.body();
    body.as_object_mut().unwrap().remove("projector_sha256");
    body["metrics"] = "not used for embedding".into();
    body["extra"] = serde_json::json!({"arbitrary": [1, true, null]});
    fixture.save(&body);

    assert!(verify_evidence(&fixture.verify()).is_ok());
}

#[test]
fn malformed_or_nonobject_evidence_is_rejected() {
    for raw in [
        b"".as_slice(),
        b"[]",
        b"true",
        b"null",
        b"{",
        b"\xef\xbb\xbf{}",
        b"\xff",
    ] {
        let fixture = Fixture::new();
        fs::write(&fixture.write.output, raw).unwrap();

        assert!(verify_evidence(&fixture.verify()).is_err(), "{raw:?}");
    }
}

#[test]
fn missing_evidence_is_rejected_without_creating_it() {
    let fixture = Fixture::new();

    let result = verify_evidence(&fixture.verify());

    assert!(result.is_err());
    assert!(!fixture.write.output.exists());
}

#[test]
fn unknown_verifier_classes_are_rejected() {
    assert!(WorkloadClass::parse("unknown").is_err());
}

#[test]
fn required_identity_fields_reject_missing_null_and_wrong_types() {
    for field in [
        "status",
        "class",
        "smoke_lane",
        "oracle_lane",
        "model_id",
        "model_sha256",
        "oracle_executable",
        "oracle_executable_sha256",
        "candidate_executable_sha256",
        "pinned_patch_sha",
        "comparison",
    ] {
        for replacement in [
            None,
            Some(serde_json::Value::Null),
            Some(true.into()),
            Some(1.into()),
        ] {
            let fixture = Fixture::new();
            let mut body = fixture.body();
            match replacement {
                Some(value) => body[field] = value,
                None => {
                    body.as_object_mut().unwrap().remove(field);
                }
            }
            fixture.save(&body);

            let result = verify_evidence(&fixture.verify());

            assert!(result.is_err(), "{field}: {body}");
        }
    }
}
