use super::*;
use crate::crypto::release_attestation::tests::{stamped_binary_bytes, test_release_signing_key};
use crate::crypto::{OwnerKeypair, ReleaseSignerTrustStore, release_signer_key_id};
use crate::runtime::release_attestation::{LoadedReleaseAttestation, load_for_binary_path};
use mesh_llm_membership::requirements::{
    MeshGenesisPolicy, MeshRequirements, ReleaseAttestationRequirement, SignedBootstrapToken,
    SignedMeshGenesisPolicy,
};

struct CertifiedMesh {
    seed: Node,
    accepting: tokio::task::JoinHandle<()>,
    directory: tempfile::TempDir,
    membership: ServingMembershipConfig,
    proof: LoadedReleaseAttestation,
}

impl CertifiedMesh {
    async fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let signing_key = test_release_signing_key(8);
        let proof = load_fixture(directory.path(), &stamped_binary_bytes(&signing_key));
        let seed = Node::new_for_tests(NodeRole::Client).await.unwrap();
        seed.set_release_attestation_report(proof.summary.clone(), proof.attestation.clone())
            .await;
        install_certified_policy(&seed, release_signer_key_id(&signing_key.verifying_key())).await;
        let token_file = directory.path().join("certified.token");
        std::fs::write(&token_file, seed.invite_token().await).unwrap();
        seed.start_accepting();
        let accepting_seed = seed.clone();
        let accepting = tokio::spawn(async move { accepting_seed.accept_loop().await });
        Self {
            seed,
            accepting,
            directory,
            membership: ServingMembershipConfig {
                name: "certified".into(),
                token_file,
                trust_store: None,
                owner_key: None,
            },
            proof,
        }
    }

    async fn attach(&self, proof: Result<LoadedReleaseAttestation>) -> Result<Membership> {
        Membership::attach(AttachmentInputs {
            config: &self.membership,
            state_dir: &self.directory.path().join("state"),
            backend: Arc::new(RecordingBackend::default()),
            descriptor: ServedModelDescriptor {
                identity: ServedModelIdentity {
                    model_name: "shared-test".into(),
                    source_kind: ModelSourceKind::LocalGguf,
                    ..Default::default()
                },
                ..Default::default()
            },
            admission: Admission::new(1, 2).unwrap(),
            queue_capacity: 2,
            lan_only: true,
            bind_ip: Some(std::net::Ipv4Addr::LOCALHOST.into()),
            ingress_slots: Arc::new(tokio::sync::Semaphore::new(4)),
            ingress_capacity: 4,
            release_attestation: proof,
        })
        .await
    }

    async fn close(self) {
        self.seed.close_endpoint().await;
        self.accepting.abort();
        let _ = self.accepting.await;
    }
}

async fn install_certified_policy(seed: &Node, signer: String) {
    let owner = OwnerKeypair::from_bytes(&[0xb1; 32], &[0xb2; 32]).unwrap();
    let policy = MeshGenesisPolicy::new(
        owner.owner_id(),
        1_717_171_717_000,
        MeshRequirements {
            release_attestation: ReleaseAttestationRequirement {
                required: true,
                allowed_signer_keys: vec![signer],
            },
            ..Default::default()
        },
    )
    .unwrap();
    let signed = SignedMeshGenesisPolicy::sign(policy.clone(), &owner).unwrap();
    let token = SignedBootstrapToken::sign(
        vec![serde_json::to_vec(&seed.endpoint_addr_for_advertisement()).unwrap()],
        &signed,
        Some(crate::mesh::current_time_unix_ms() + 60_000),
        &owner,
    )
    .unwrap();
    seed.install_requirement_aware_mesh_state(
        policy.policy_derived_mesh_id().unwrap(),
        policy.canonical_hash_hex().unwrap(),
        policy,
        Some(signed),
        Some(token),
    )
    .await
    .unwrap();
}

fn load_fixture(directory: &std::path::Path, bytes: &[u8]) -> LoadedReleaseAttestation {
    let path = directory.join("mesh-llm-test-binary");
    std::fs::write(&path, bytes).unwrap();
    load_for_binary_path(&path, &ReleaseSignerTrustStore::default()).unwrap()
}

#[tokio::test]
async fn shared_attachment_joins_certified_mesh_with_verified_binary_proof() {
    let fixture = CertifiedMesh::new().await;
    let attachment = fixture.attach(Ok(fixture.proof.clone())).await;
    let attachment = attachment.expect("certified shared attachment must pass QUIC admission");
    let endpoint = attachment.node.id();
    let peer = fixture
        .seed
        .peers()
        .await
        .into_iter()
        .find(|peer| peer.id == endpoint)
        .expect("certified attachment is admitted by the seed");
    assert!(peer.admitted);
    assert!(peer.release_attestation_summary.verified);
    assert_eq!(
        attachment.node.release_attestation_summary().await.status,
        crate::ReleaseAttestationStatus::Valid
    );
    attachment.advertise(true).await;
    let (mut send, mut recv) = fixture.seed.open_http_tunnel(endpoint).await.unwrap();
    send.write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    send.finish().unwrap();
    let response = tokio::time::timeout(Duration::from_secs(10), recv.read_to_end(32768))
        .await
        .unwrap()
        .unwrap();
    let response = String::from_utf8(response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("shared-test"), "{response}");
    attachment.close().await;

    let reattached = fixture.attach(Ok(fixture.proof.clone())).await.unwrap();
    assert_eq!(reattached.node.id(), endpoint);
    assert!(reattached.node.release_attestation_summary().await.verified);
    reattached.close().await;
    fixture.close().await;
}

#[tokio::test]
async fn shared_attachment_rejects_missing_tampered_and_untrusted_binary_proofs() {
    let mut tampered = stamped_binary_bytes(&test_release_signing_key(8));
    tampered[0] ^= 1;
    let candidates = [
        (
            b"unsigned-binary".to_vec(),
            crate::ReleaseAttestationStatus::Missing,
        ),
        (tampered, crate::ReleaseAttestationStatus::Invalid),
        (
            stamped_binary_bytes(&test_release_signing_key(9)),
            crate::ReleaseAttestationStatus::Valid,
        ),
    ];
    for (bytes, expected_status) in candidates {
        let fixture = CertifiedMesh::new().await;
        let proof = load_fixture(fixture.directory.path(), &bytes);
        assert_eq!(proof.summary.status, expected_status);
        let expected_rejection = if expected_status == crate::ReleaseAttestationStatus::Valid {
            crate::MeshRequirementRejectReason::ReleaseSignerUntrusted
        } else {
            crate::MeshRequirementRejectReason::CertifiedBinaryRequired
        };
        let result = fixture.attach(Ok(proof)).await;
        assert!(result.is_err(), "uncertified binary must not join");
        let error = result.err().unwrap();
        assert!(format!("{error:#}").contains("rejected"), "{error:#}");
        assert!(fixture.seed.peers().await.is_empty());
        let rejections = fixture.seed.recent_mesh_requirement_rejections().await;
        assert!(
            rejections
                .iter()
                .any(|event| event.reason == expected_rejection),
            "expected {expected_rejection:?}, got {rejections:?}"
        );
        fixture.close().await;
    }
    let fixture = CertifiedMesh::new().await;
    let unreadable = load_for_binary_path(
        &fixture.directory.path().join("missing-binary"),
        &ReleaseSignerTrustStore::default(),
    );
    assert!(unreadable.is_err());
    assert!(fixture.attach(unreadable).await.is_err());
    assert!(fixture.seed.peers().await.is_empty());
    fixture.close().await;
}
