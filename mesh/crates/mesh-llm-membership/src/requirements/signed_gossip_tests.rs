use super::*;

#[test]
fn strict_gossip_is_opt_in_and_bound_to_policy_identity() {
    let legacy = MeshGenesisPolicy::new("owner", 123, MeshRequirements::default()).unwrap();
    // Fixed pre-extension v1 bytes: existing mesh IDs, signatures and
    // invitations must remain valid when the new requirement is absent.
    let legacy_bytes = b"mesh-llm-genesis-policy-v1:\
        \x01\0\0\0\
        \x05\0\0\0\0\0\0\0owner\
        \x7b\0\0\0\0\0\0\0\
        \0\0\0\0\0\
        \0\0\0\0\0\0\0\0";
    assert_eq!(legacy.canonical_bytes().unwrap(), legacy_bytes);
    let mut strict = legacy.clone();
    strict.requirements.require_signed_gossip = true;
    assert_ne!(
        legacy.canonical_hash().unwrap(),
        strict.canonical_hash().unwrap()
    );
    assert!(!strict.requirements.is_unrestricted());
    assert_eq!(
        MeshGenesisPolicy::from_proto(&strict.to_proto()).unwrap(),
        strict
    );
    assert_eq!(
        MeshGenesisPolicy::from_proto(&legacy.to_proto()).unwrap(),
        legacy
    );
    let serialized = serde_json::to_value(&legacy).unwrap();
    assert!(
        serialized["requirements"]
            .get("require_signed_gossip")
            .is_none()
    );
    assert_eq!(
        serde_json::from_value::<MeshGenesisPolicy>(serialized).unwrap(),
        legacy
    );
}

#[test]
fn dropping_the_strict_flag_cannot_downgrade_a_signed_policy() {
    let owner = mesh_llm_identity::OwnerKeypair::from_bytes(&[31; 32], &[32; 32]).unwrap();
    let policy = MeshGenesisPolicy::new(
        owner.owner_id(),
        123,
        MeshRequirements {
            require_signed_gossip: true,
            ..Default::default()
        },
    )
    .unwrap();
    let signed = SignedMeshGenesisPolicy::sign(policy, &owner).unwrap();
    signed.verify().unwrap();
    let mut wire = signed.to_proto();
    // An old decoder discards the new field. It must not authenticate a
    // weaker policy using the original signature or join the strict mesh.
    wire.policy
        .as_mut()
        .unwrap()
        .requirements
        .as_mut()
        .unwrap()
        .require_signed_gossip = false;
    let downgraded = SignedMeshGenesisPolicy::from_proto(&wire);
    assert!(downgraded.is_err() || downgraded.unwrap().verify().is_err());
}
