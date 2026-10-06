use super::*;
#[test]
fn composition_terminal_preserves_rows_and_prior_error_and_refuses_late_success() {
    for mode in ["success", "cancel", "deadline", "finish", "prior"] {
        let cancel = Cancellation::default();
        if mode == "cancel" {
            cancel.cancel();
        }
        let deadline =
            Instant::now() + Duration::from_secs(if mode == "deadline" { 0 } else { 60 });
        let mut e =
            json!({"status":"FAILED","native_composition":{"rows":[{"sha256":"a".repeat(64)}]}});
        let result = if mode == "prior" {
            Err("prior native refusal".into())
        } else {
            Ok(json!({"entries":[1,2,3]}))
        };
        let finish = if mode == "finish" {
            Err("owned finish failure".into())
        } else {
            Ok(())
        };
        assert_eq!(
            finalize(result, finish, deadline, &cancel, &mut e).is_ok(),
            mode == "success"
        );
        assert_eq!(e["native_composition"]["rows"].as_array().unwrap().len(), 1);
        assert_eq!(e["status"] == "COMPOSED_PUBLISHED", mode == "success");
        if mode == "prior" {
            assert!(
                e["error"]
                    .as_str()
                    .unwrap()
                    .contains("prior native refusal")
            );
        }
    }
}
#[test]
fn ordered_publication_preserves_first_middle_last_order_and_rejects_duplicate_name() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().canonicalize().unwrap();
    let rows=(1..=3).map(|i|{let p=root.join(format!("part{i}.gguf"));std::fs::write(&p,[i]).unwrap();json!({"local":p,"sha256":admission::digest(&[i]),"path_in_repo":format!("Composite-{i:05}-of-00003.gguf")})}).collect::<Vec<_>>();
    let mut plan = json!({"entries":rows});
    let ordered = publication::ordered(&plan).unwrap();
    assert_eq!(ordered[1]["path_in_repo"], "Composite-00002-of-00003.gguf");
    assert_eq!(ordered[1]["byte_size"], 1);
    plan["entries"][2]["path_in_repo"] = plan["entries"][0]["path_in_repo"].clone();
    assert!(publication::ordered(&plan).is_err());
    t.close().unwrap();
}
#[test]
fn stage_envelope_cannot_replace_effective_checkpoint_roster_or_profile() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().canonicalize().unwrap();
    let source = contract::Source {
        repo: "owner/source".into(),
        revision: "a".repeat(40),
        files: std::collections::BTreeMap::from([("tokenizer.json".into(), "b".repeat(64))]),
    };
    let request = contract::Staging {
        schema_version: 1,
        checkpoint: source.clone(),
        tokenizer_source: source,
        tokenizer_profile: root.join("profile.json"),
        tokenizer_profile_sha256: "c".repeat(64),
        output_directory: root.join("stage"),
        credential_file: None,
        timeout_seconds: 5,
        maximum_bytes: 1024,
    };
    let mut receipt = json!({"checkpoint_files":[{"path":request.output_directory.join("mtp-src/tokenizer.json"),"sha256":"b".repeat(64)}],"tokenizer_profile":{"path":request.output_directory.join("tokenizer-profile.json"),"sha256":"c".repeat(64)},"checkpoint_source":{"repo":"owner/source","revision":"a".repeat(40),"files":{"tokenizer.json":"b".repeat(64)}},"tokenizer_source":{"repo":"owner/source","revision":"a".repeat(40),"files":{"tokenizer.json":"b".repeat(64)}}});
    staging::correlate(&request, &receipt).unwrap();
    receipt["checkpoint_files"][0]["sha256"] = json!("d".repeat(64));
    assert!(staging::correlate(&request, &receipt).is_err());
    t.close().unwrap();
}
