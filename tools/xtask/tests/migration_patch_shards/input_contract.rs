use super::{AFMOE, ShardError, diff, encode_family_shards, fixture_manifest, fixture_map};
use serde_json::{Value, json};

#[test]
fn migration_patch_shards_reject_invalid_family_map_records() {
    for families in [
        json!({"Bad": ["src/models/a.cpp"]}),
        json!({"": ["src/models/a.cpp"]}),
        json!({"a": []}),
        json!({"a": "src/models/a.cpp"}),
        json!({"a": ["src/models/a.cpp", "src/models/a.cpp"]}),
        json!({"a": ["src/models/nested/a.cpp"]}),
        json!({"a": ["src/models/.cpp"]}),
        json!({"a": ["src/models/a.cpp\n"]}),
        json!({"a": [null]}),
    ] {
        let map = json!({"schema_version": 1, "families": families});

        let result = encode_family_shards(diff(AFMOE), &map, &json!({"models": []}));

        assert!(
            matches!(
                result,
                Err(ShardError::FamilyMapping(_) | ShardError::SourceMapping(_))
            ),
            "{map}"
        );
    }
}

#[test]
fn migration_patch_shards_reject_wrong_map_schema_or_shape() {
    for map in [
        json!({}),
        json!([]),
        json!({"schema_version":2,"families":{}}),
        json!({"schema_version":"1","families":{}}),
        json!({"schema_version":1,"families":[]}),
    ] {
        let result = encode_family_shards(diff(AFMOE), &map, &json!({"models": []}));

        assert!(matches!(result, Err(ShardError::FamilyMapSchema)), "{map}");
    }
}

#[test]
fn migration_patch_shards_accept_empty_maps_and_python_schema_one_values() {
    for schema in [json!(1), json!(1.0), json!(true)] {
        let map = json!({"schema_version":schema,"families":{},"extra":42});

        let result =
            encode_family_shards(diff(AFMOE), &map, &json!({"schema_version":42,"models":[]}))
                .unwrap();

        assert_eq!(result.series, b"0001-family-unmapped.patch\n");
    }
}

#[test]
fn migration_patch_shards_accept_map_names_without_extra_path_policy() {
    let map = json!({"schema_version":1,"families":{
        "0a._-": ["src/models/..cpp", "src/models/A_B-1.cpp"]
    }});
    let input = b"diff --git a/src/models/..cpp b/elsewhere\n";

    let result = encode_family_shards(input, &map, &json!({"models":[]})).unwrap();

    assert_eq!(result.series, b"0001-family-0a._-.patch\n");
}

#[test]
fn migration_patch_shards_reject_missing_causal_coverage() {
    let map = json!({"schema_version":1,"families":{}});

    let result = encode_family_shards(diff(AFMOE), &map, &fixture_manifest());

    assert!(
        matches!(result, Err(ShardError::MissingCoverage(names)) if names == ["afmoe", "deepseek2", "mistral4", "rwkv6"])
    );
}

#[test]
fn migration_patch_shards_reject_missing_duplicate_or_unhashable_manifest_families() {
    for names in [
        vec![Value::Null],
        vec![json!([])],
        vec![json!({})],
        vec![json!("same"), json!("same")],
        vec![json!(true), json!(1.0)],
        vec![json!(0), json!(-0.0)],
    ] {
        let models: Vec<_> = names
            .iter()
            .map(|family| json!({"family":family,"class":"embedding","profile":"workload-smoke"}))
            .collect();

        let result = encode_family_shards(diff(AFMOE), &fixture_map(), &json!({"models":models}));

        assert!(
            matches!(result, Err(ShardError::ManifestFamily)),
            "{names:?}"
        );
    }
}

#[test]
fn migration_patch_shards_reject_absent_family_and_invalid_models_shape() {
    for manifest in [
        json!({}),
        json!({"models":{}}),
        json!({"models":[null]}),
        json!({"models":[{"class":"embedding","profile":"workload-smoke"}]}),
    ] {
        let result = encode_family_shards(diff(AFMOE), &fixture_map(), &manifest);

        assert!(
            matches!(
                result,
                Err(ShardError::ManifestModels | ShardError::ManifestFamily)
            ),
            "{manifest}"
        );
    }
}

#[test]
fn migration_patch_shards_reject_unknown_classes_and_cross_class_profiles() {
    for fields in [
        json!({}),
        json!({"class":"future"}),
        json!({"class":"embedding","profile":"full"}),
        json!({"class":"causal_generation","profile":"workload-oracle"}),
    ] {
        let mut model = fields;
        model["family"] = json!("afmoe");

        let result = encode_family_shards(diff(AFMOE), &fixture_map(), &json!({"models":[model]}));

        assert!(matches!(
            result,
            Err(ShardError::WorkloadClass
                | ShardError::CausalProfile
                | ShardError::WorkloadProfile)
        ));
    }
}

#[test]
fn migration_patch_shards_accept_all_legacy_workload_profile_pairs() {
    for class in [
        "embedding",
        "rerank",
        "encoder_decoder",
        "ocr",
        "speech_synthesis",
        "speech_recognition",
    ] {
        for profile in ["workload-smoke", "workload-oracle"] {
            let manifest = json!({"models":[{"family":"Not a decoder name!","class":class,"profile":profile}]});

            let result = encode_family_shards(diff(AFMOE), &fixture_map(), &manifest).unwrap();

            assert_eq!(result.shards[0].families, ["afmoe"]);
        }
    }
}

#[test]
fn migration_patch_shards_accept_all_causal_profiles() {
    for profile in ["full", "package-oracle", "graph-only"] {
        let manifest =
            json!({"models":[{"family":"afmoe","class":"causal_generation","profile":profile}]});

        let result = encode_family_shards(diff(AFMOE), &fixture_map(), &manifest).unwrap();

        assert_eq!(result.shards[0].bytes, AFMOE);
    }
}

#[test]
fn migration_patch_shards_preserve_nonchat_scalar_family_acceptance() {
    let models: Vec<_> = [json!(""), json!(false), json!(1), json!(1.5), json!("1")]
        .into_iter()
        .map(|family| json!({"family":family,"class":"ocr","profile":"workload-smoke"}))
        .collect();

    let result =
        encode_family_shards(diff(AFMOE), &fixture_map(), &json!({"models":models})).unwrap();

    assert_eq!(result.shards[0].bytes, AFMOE);
}
