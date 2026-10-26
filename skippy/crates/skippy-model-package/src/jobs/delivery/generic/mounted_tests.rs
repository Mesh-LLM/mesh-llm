use super::super::request_transport::{LOCATOR_ENVIRONMENT, MountedRequest};
use super::*;
#[test]
fn mounted_generic_request_carries_full_native_roster_without_large_provider_secret() {
    let (mut input, mut mounts, plan) = facade_fixture();
    input["operator"]["conversion"]["source_files"] = serde_json::json!((0..1024).map(|i| serde_json::json!({"path":format!("/models/target/source-{i:04}.safetensors"),"sha256":"1".repeat(64)})).collect::<Vec<_>>());
    mounts.push(ModelMount {
        repo: "fixture/request".into(),
        revision: "a".repeat(40),
        mount_path: "/models/request".into(),
    });
    let bytes = serde_json::to_vec(&input).unwrap();
    assert!(bytes.len() > 65536);
    assert!(PreparedConversionDelivery::prepare(&bytes, &mounts, &plan).is_err());
    let locator = MountedRequest {
        schema_version: 1,
        path: "/models/request/request.json".into(),
        sha256: admission::digest(&bytes),
        byte_size: bytes.len() as u64,
        repo: "fixture/request".into(),
        revision: "a".repeat(40),
    };
    let prepared =
        PreparedConversionDelivery::prepare_mounted(&bytes, &mounts, &plan, &locator).unwrap();
    assert_eq!(
        prepared.declaration().transport_input_sha256,
        locator.sha256
    );
    assert!(prepared.native.spec.secrets.is_empty());
    assert_eq!(
        prepared.native.spec.arguments[3],
        "--mounted-input-environment"
    );
    assert_eq!(prepared.native.spec.arguments[4], LOCATOR_ENVIRONMENT);
    assert!(prepared.native.spec.environment[LOCATOR_ENVIRONMENT].len() < 8192);
    assert!(
        prepared
            .native
            .spec
            .volumes
            .iter()
            .all(|v| v.read_only == Some(true))
    );
    assert!(!prepared.declaration().submitted);
    for changed in [
        MountedRequest {
            sha256: "2".repeat(64),
            ..locator.clone()
        },
        MountedRequest {
            byte_size: locator.byte_size + 1,
            ..locator.clone()
        },
        MountedRequest {
            revision: "b".repeat(40),
            ..locator.clone()
        },
        MountedRequest {
            path: "/models/other/request.json".into(),
            ..locator.clone()
        },
    ] {
        assert!(
            PreparedConversionDelivery::prepare_mounted(&bytes, &mounts, &plan, &changed).is_err()
        );
    }
    input["operator"]["conversion"]["source_files"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"path":"/models/target/extra","sha256":"1".repeat(64)}));
    let bytes = serde_json::to_vec(&input).unwrap();
    let too_many = MountedRequest {
        sha256: admission::digest(&bytes),
        byte_size: bytes.len() as u64,
        ..locator
    };
    assert!(
        PreparedConversionDelivery::prepare_mounted(&bytes, &mounts, &plan, &too_many).is_err()
    );
}
