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

/// Startup registers every served ref against the GGUF it was loaded from
/// so by-name metadata lookups reach the file. That registration must not
/// let `/v1/models` re-derive a host-specific path-hash id from the file:
/// an explicit `--model` alias, a configured alias and a runtime-minted
/// content-addressed id all have to stay listed verbatim and keep routing
/// (the renamed id broke the Laya smoke; PR #2651 review found the alias
/// form of the same bug).
#[test]
fn served_refs_registered_against_a_real_gguf_keep_their_listed_id() {
    let temp = tempfile::tempdir().unwrap();
    let served = [
        ("my-model", "weights.gguf"),
        ("configured/alias:Q4_K_M", "configured.gguf"),
        (
            "local-gguf/sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "content-addressed.gguf",
        ),
    ];
    for (model_ref, file_name) in served {
        let gguf = temp.path().join(file_name);
        std::fs::write(&gguf, b"GGUF").unwrap();
        crate::models::remember_served_model_ref(model_ref, &gguf);

        // The file now resolves by name, which is what the metadata scan
        // needs, and would otherwise be re-derived into a path-hash ref.
        assert_eq!(crate::models::find_model_path(model_ref), gguf);
        assert!(crate::models::model_ref_for_path(&gguf).starts_with("local-gguf/sha256-"));

        let descriptor = mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_ref.into(),
                source_kind: mesh::ModelSourceKind::LocalGguf,
                local_file_name: Some(file_name.into()),
                ..Default::default()
            },
            ..Default::default()
        };
        for profile in ["", "low-ctx"] {
            let expected = if profile.is_empty() {
                model_ref.to_string()
            } else {
                format!("{model_ref}#{profile}")
            };
            let public = public_model_id(model_ref, Some(&descriptor), profile);
            assert_eq!(public, expected);
            // The listed id must parse back to the served ref the request
            // resolver routes on.
            let internal = if profile.is_empty() {
                model_ref.to_string()
            } else {
                format!("{model_ref}#{profile}")
            };
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

/// A configured alias can be re-registered against a different GGUF (model
/// swap under the same name, or a later startup registration picking the
/// file actually loaded). The alias must then describe the file served
/// now, not keep whichever path sorted first (CodeRabbit, PR #2651).
#[test]
fn re_registering_a_served_alias_replaces_its_file() {
    let temp = tempfile::tempdir().unwrap();
    // Alphabetically the first path wins the scanner's preference key, so
    // "keep the preferred" semantics would pin the alias to the stale file.
    let first = temp.path().join("a-first.gguf");
    let second = temp.path().join("b-second.gguf");
    std::fs::write(&first, b"GGUF").unwrap();
    std::fs::write(&second, b"GGUF").unwrap();
    let alias = "swapped/alias:Q4_K_M";

    crate::models::remember_served_model_ref(alias, &first);
    assert_eq!(crate::models::find_model_path(alias), first);

    crate::models::remember_served_model_ref(alias, &second);
    assert_eq!(crate::models::find_model_path(alias), second);

    let descriptor = mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: alias.into(),
            source_kind: mesh::ModelSourceKind::LocalGguf,
            local_file_name: Some("b-second.gguf".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(public_model_id(alias, Some(&descriptor), ""), alias);
}
