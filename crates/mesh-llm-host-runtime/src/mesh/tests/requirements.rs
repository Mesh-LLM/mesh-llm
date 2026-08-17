#[test]
fn mesh_requirements_outbound_admits_compliant_peer_after_requirements_pass() {
    assert_mesh_requirements_outbound_admits_compliant_peer_after_requirements_pass();
}

#[test]
fn mesh_requirements_inbound_rejects_before_topology_announcement() {
    assert_mesh_requirements_inbound_rejects_before_topology_announcement();
}

#[test]
fn mesh_requirements_outbound_rejects_before_peer_promotion() {
    assert_mesh_requirements_outbound_rejects_before_peer_promotion();
}

#[test]
fn mesh_requirements_add_peer_rejects_missing_direct_admission_proof() {
    assert_mesh_requirements_add_peer_rejects_missing_direct_admission_proof();
}

#[test]
fn mesh_requirements_add_peer_rejects_invalid_direct_admission_proof() {
    assert_mesh_requirements_add_peer_rejects_invalid_direct_admission_proof();
}

#[test]
fn mesh_requirements_add_peer_rejects_stale_direct_admission_proof() {
    assert_mesh_requirements_add_peer_rejects_stale_direct_admission_proof();
}

#[test]
fn mesh_requirements_add_peer_rejects_direct_proof_sender_mismatch() {
    assert_mesh_requirements_add_peer_rejects_direct_proof_sender_mismatch();
}

#[test]
fn requirement_aware_mesh_without_attestation_rejects_missing_direct_proof() {
    assert_requirement_aware_mesh_without_attestation_rejects_missing_direct_proof();
}

#[test]
fn fast_join_apply_failure_closes_connection_and_propagates_err() {
    assert_fast_join_apply_failure_closes_connection_and_propagates_err();
}

#[test]
fn requirement_aware_mesh_without_attestation_rejects_invalid_direct_proof() {
    assert_requirement_aware_mesh_without_attestation_rejects_invalid_direct_proof();
}

#[test]
fn requirement_aware_mesh_without_attestation_rejects_stale_direct_proof() {
    assert_requirement_aware_mesh_without_attestation_rejects_stale_direct_proof();
}

#[test]
fn requirement_aware_mesh_without_attestation_rejects_sender_mismatch_direct_proof() {
    assert_requirement_aware_mesh_without_attestation_rejects_sender_mismatch_direct_proof();
}

#[test]
fn requirement_aware_mesh_without_attestation_accepts_valid_direct_proof() {
    assert_requirement_aware_mesh_without_attestation_accepts_valid_direct_proof();
}

#[test]
fn mesh_requirements_add_peer_rejects_untrusted_release_signer() {
    assert_mesh_requirements_add_peer_rejects_untrusted_release_signer();
}

#[test]
fn mesh_requirements_add_peer_rejects_invalid_release_attestation_signature() {
    assert_mesh_requirements_add_peer_rejects_invalid_release_attestation_signature();
}

#[test]
fn mesh_requirements_add_peer_rejects_wrong_mesh_id() {
    assert_mesh_requirements_add_peer_rejects_wrong_mesh_id();
}

#[test]
fn mesh_requirements_transitive_gossip_never_admits_peer_without_direct_proof() {
    assert_mesh_requirements_transitive_gossip_never_admits_peer_without_direct_proof();
}

#[test]
fn mesh_requirements_rejected_peer_messages_have_no_mesh_effect() {
    assert_mesh_requirements_rejected_peer_messages_have_no_mesh_effect();
}

#[test]
fn mesh_requirements_gossip_rejects_direct_sender_before_payload_effects() {
    assert_mesh_requirements_gossip_rejects_direct_sender_before_payload_effects();
}

#[test]
fn mesh_requirements_join_rejects_invalid_bootstrap_token() {
    assert_mesh_requirements_join_rejects_invalid_bootstrap_token();
}

#[test]
fn mesh_requirements_join_accepts_matching_bootstrap_before_policy_state_installed() {
    assert_mesh_requirements_join_accepts_matching_bootstrap_before_policy_state_installed();
}

#[test]
fn mesh_requirements_unrestricted_legacy_mesh_join_stays_compatible() {
    assert_mesh_requirements_unrestricted_legacy_mesh_join_stays_compatible();
}

#[test]
fn named_mesh_id_uses_documented_sha256_derivation() {
    assert_named_mesh_id_uses_documented_sha256_derivation();
}

#[test]
#[serial_test::serial]
fn persisted_random_mesh_id_is_preserved() {
    assert_persisted_random_mesh_id_is_preserved();
}

#[test]
fn requirement_invite_signing_failure_uses_cached_fallback() {
    assert_requirement_invite_signing_failure_uses_cached_fallback();
}

#[test]
#[serial_test::serial]
fn malformed_genesis_policy_fails_closed() {
    assert_malformed_genesis_policy_fails_closed();
}

#[test]
#[serial_test::serial]
fn unverified_genesis_policy_fails_closed() {
    assert_unverified_genesis_policy_fails_closed();
}

#[test]
#[serial_test::serial]
fn mismatched_genesis_policy_fails_closed() {
    assert_mismatched_genesis_policy_fails_closed();
}

#[test]
fn requirement_state_reads_coherent_snapshot() {
    assert_requirement_state_reads_coherent_snapshot();
}

#[test]
fn concurrent_requirement_state_installs_do_not_overwrite() {
    assert_concurrent_requirement_state_installs_do_not_overwrite();
}
