use super::*;
fn template(root: &Path) -> Template {
    let pin = |path| Artifact {
        path,
        sha256: "a".repeat(64),
    };
    Template {
        checkpoint_directory: root.join("source"),
        checkpoint_files: ["config.json", "tokenizer.json", "head.safetensors"]
            .into_iter()
            .map(|name| pin(root.join("source").join(name)))
            .collect(),
        tokenizer_profile: pin(root.join("profile.json")),
        target_parts: (1..=3)
            .map(|i| pin(root.join(format!("Target-{i:05}-of-00003.gguf"))))
            .collect(),
        target_basename: "Target".into(),
        composite_basename: "Composite".into(),
        expected_parts: 3,
        mtp_block: 3,
        composite_repo: "fixture/composite".into(),
    }
}
#[test]
fn raw_conversion_closes_bf16_profile_roster_and_existing_compose_projection() {
    let root = tempfile::tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();
    let input = template(&root_path);
    input.validate().unwrap();
    let args = input.convert_args(&root_path).unwrap();
    for (key, value) in [
        ("--backend", "native-rust"),
        ("--output-type", "bf16"),
        ("--expected-splits", "1"),
        ("--window-size", "1"),
        ("--split-max-size", "0"),
        ("--target-prefix", ""),
    ] {
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == key && pair[1] == value)
        );
    }
    assert!(
        args.windows(2)
            .any(|pair| pair[0] == "--nemotron-mtp-tokenizer-profile"
                && pair[1] == input.tokenizer_profile.path.to_str().unwrap())
    );
    let mut duplicate = input.clone();
    duplicate
        .checkpoint_files
        .push(duplicate.checkpoint_files[0].clone());
    assert!(duplicate.validate().is_err());
    let mtp = Artifact {
        path: root_path.join("observed-mtp.gguf"),
        sha256: "b".repeat(64),
    };
    let compose = input.composition(mtp.clone());
    let MtpSource::SuppliedConverted { artifact } = compose.mtp else {
        panic!("wrong source");
    };
    assert!(artifact == mtp);
    assert!(compose.target_parts == input.target_parts);
    root.close().unwrap();
}
#[test]
fn raw_conversion_cancel_or_elapsed_deadline_refuses_before_identity_publication() {
    let root = tempfile::tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();
    let input = template(&root_path);
    let binary = Artifact {
        path: root_path.join("not-launched"),
        sha256: "b".repeat(64),
    };
    for cancel in [false, true] {
        let cancellation = Cancellation::default();
        if cancel {
            cancellation.cancel();
        }
        let mut evidence = json!({});
        let context = Context {
            binary: &binary,
            mesh_revision: &"c".repeat(40),
            deadline: if cancel {
                Instant::now() + Duration::from_secs(60)
            } else {
                Instant::now()
            },
            cancellation: &cancellation,
        };
        assert!(input.execute(&root_path, &context, &mut evidence).is_err());
        assert!(evidence.as_object().unwrap().is_empty());
        assert_eq!(std::fs::read_dir(&root_path).unwrap().count(), 0);
    }
    root.close().unwrap();
}
#[test]
fn raw_verification_binds_real_native_diagnostic_names_without_claiming_runtime() {
    let root = tempfile::tempdir().unwrap();
    let value = json!({"root":root.path(),"prefix":"","basename":"mtp","expected_splits":1,"completed_count":1,"first_missing":null,"last_present":1,"first_shard":"mtp-00001-of-00001.gguf","last_shard":"mtp-00001-of-00001.gguf","complete":true});
    correlate_verification(root.path(), &value).unwrap();
    for key in ["complete", "root", "first_shard", "completed_count"] {
        let mut wrong = value.clone();
        wrong[key] = Value::Null;
        assert!(correlate_verification(root.path(), &wrong).is_err());
    }
    root.close().unwrap();
}
