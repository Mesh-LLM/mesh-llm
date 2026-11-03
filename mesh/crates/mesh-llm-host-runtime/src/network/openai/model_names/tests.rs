use super::*;
use crate::network::openai::request_parse::internal_model_for_public_id;

fn identity(source_kind: mesh::ModelSourceKind) -> mesh::ServedModelIdentity {
    mesh::ServedModelIdentity {
        model_name: "local-gguf/sha256-model-identity-test".into(),
        source_kind,
        repository: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF".into()),
        canonical_ref: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M".into()),
        artifact: Some("Falcon-H1-1.5B-Instruct-Q4_K_M.gguf".into()),
        ..Default::default()
    }
}

#[test]
fn public_ids_round_trip_to_the_same_runtime_identity_with_profiles() {
    for source in [
        mesh::ModelSourceKind::HuggingFace,
        mesh::ModelSourceKind::Catalog,
    ] {
        for revision in [None, Some("main"), Some("immutable-revision")] {
            let mut identity = identity(source);
            identity.revision = revision.map(str::to_owned);
            let internal = identity.model_name.clone();
            let descriptor = mesh::ServedModelDescriptor {
                identity,
                ..Default::default()
            };
            let expected = mesh::public_model_id_from_identity(&descriptor.identity).unwrap();
            for profile in ["", "small-context"] {
                let internal = if profile.is_empty() {
                    internal.clone()
                } else {
                    format!("{internal}#{profile}")
                };
                let public =
                    public_model_id(&descriptor.identity.model_name, Some(&descriptor), profile);
                let expected = if profile.is_empty() {
                    expected.clone()
                } else {
                    format!("{expected}#{profile}")
                };
                assert_eq!(public, expected);
                assert_eq!(
                    internal_model_for_public_id(
                        &public,
                        std::slice::from_ref(&internal),
                        std::slice::from_ref(&descriptor)
                    ),
                    Some(internal)
                );
            }
        }
    }
}

#[test]
fn incomplete_huggingface_metadata_does_not_discard_quantization_suffix() {
    let mut identity = identity(mesh::ModelSourceKind::HuggingFace);
    identity.model_name = "missing-test-model-identity-Q8_0".into();
    identity.artifact = None;
    let descriptor = mesh::ServedModelDescriptor {
        identity,
        ..Default::default()
    };
    assert_eq!(
        public_model_id(&descriptor.identity.model_name, Some(&descriptor), "small"),
        "missing-test-model-identity-Q8_0#small"
    );
}
