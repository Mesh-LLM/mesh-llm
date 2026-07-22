#[test]
fn early_tui_spawns_before_llama_ready_in_active_flow() {
    runtime::assert_active_serve_path_spawn_gate_behavior();
}

#[test]
fn passive_path_tui_still_starts_immediately() {
    runtime::assert_passive_path_immediate_spawn_behavior();
}

#[test]
fn interactive_handler_spawns_once_across_startup_callbacks() {
    runtime::assert_interactive_handler_spawns_once_across_startup_callbacks();
}

#[test]
fn startup_launch_plan_describes_planned_runtime_before_process_start() {
    runtime::assert_startup_launch_plan_describes_planned_runtime_before_process_start();
}

#[test]
fn quitting_during_startup_cancels_without_late_ready_render() {
    runtime::assert_quitting_during_startup_cancels_without_late_ready_render();
}

#[test]
fn mesh_requirements_policy_canonical_hash_is_stable() {
    mesh::requirements::tests::assert_mesh_requirements_policy_canonical_hash_is_stable();
}

#[test]
fn mesh_requirements_policy_change_changes_mesh_id() {
    mesh::requirements::tests::assert_mesh_requirements_policy_change_changes_mesh_id();
}

#[test]
fn mesh_requirements_bootstrap_token_validates_origin_signature() {
    mesh::requirements::tests::assert_mesh_requirements_bootstrap_token_validates_origin_signature(
    );
}

#[test]
fn mesh_requirements_bootstrap_rejects_expired_token() {
    mesh::requirements::tests::assert_mesh_requirements_bootstrap_rejects_expired_token();
}

#[test]
fn mesh_requirements_bootstrap_rejects_policy_hash_mismatch() {
    mesh::requirements::tests::assert_mesh_requirements_bootstrap_rejects_policy_hash_mismatch();
}

#[test]
fn mesh_requirements_policy_hash_derives_mesh_id() {
    mesh::requirements::tests::assert_mesh_requirements_policy_hash_derives_mesh_id();
}

#[test]
fn mesh_requirements_version_bounds_unset_min_only_max_only_and_exact() {
    mesh::requirements::tests::assert_mesh_requirements_version_bounds_unset_min_only_max_only_and_exact();
}

#[test]
fn mesh_requirements_protocol_bounds_reject_unknown_only_when_constrained() {
    mesh::requirements::tests::assert_mesh_requirements_protocol_bounds_reject_unknown_only_when_constrained();
}

#[test]
fn mesh_requirements_rejects_unsigned_when_attestation_required() {
    mesh::requirements::tests::assert_mesh_requirements_rejects_unsigned_when_attestation_required(
    );
}

#[test]
fn mesh_requirements_rejection_reasons_are_stable() {
    mesh::requirements::tests::assert_mesh_requirements_rejection_reasons_are_stable();
}

#[test]
fn mesh_requirements_cli_accepts_each_bound_independently() {
    runtime::assert_mesh_requirements_cli_accepts_each_bound_independently();
}

#[test]
fn mesh_requirements_config_accepts_unset_min_only_max_only_and_full_ranges() {
    plugin::assert_mesh_requirements_config_accepts_unset_min_only_max_only_and_full_ranges();
}

#[test]
fn mesh_requirements_config_rejects_required_attestation_without_signer_keys() {
    plugin::assert_mesh_requirements_config_rejects_required_attestation_without_signer_keys();
}

#[test]
fn mesh_requirements_config_rejects_non_ed25519_signer_key() {
    plugin::assert_mesh_requirements_config_rejects_non_ed25519_signer_key();
}

#[test]
fn mesh_requirements_survive_owner_control_config_round_trip() {
    protocol::tests::mesh_requirements_survive_owner_control_config_round_trip();
}

#[test]
fn mesh_requirements_cli_overrides_config_per_field_before_genesis() {
    runtime::assert_mesh_requirements_cli_overrides_config_per_field_before_genesis();
}

#[test]
fn mesh_requirements_config_rejects_min_greater_than_max_after_merge() {
    runtime::assert_mesh_requirements_config_rejects_min_greater_than_max_after_merge();
}

#[test]
fn mesh_requirements_rejects_local_policy_mutation_on_existing_mesh() {
    runtime::assert_mesh_requirements_rejects_local_policy_mutation_on_existing_mesh();
}

#[test]
fn mesh_requirements_direct_proof_rejects_stale_timestamp() {
    mesh::requirements::tests::assert_mesh_requirements_direct_proof_rejects_stale_timestamp();
}

#[test]
fn mesh_requirements_direct_proof_rejects_sender_id_mismatch() {
    mesh::requirements::tests::assert_mesh_requirements_direct_proof_rejects_sender_id_mismatch();
}

#[test]
fn mesh_requirements_status_excludes_rejected_peers_from_admitted_list() {
    api::tests::assert_mesh_requirements_status_excludes_rejected_peers_from_admitted_list();
}

#[test]
fn mesh_requirements_status_reports_policy_hash_read_only() {
    api::tests::assert_mesh_requirements_status_reports_policy_hash_read_only();
}

#[test]
fn mesh_requirements_certified_binary_required_event_text() {
    api::tests::assert_mesh_requirements_certified_binary_required_event_text();
}

#[test]
fn mesh_requirements_rejection_events_do_not_expose_tokens() {
    api::tests::assert_mesh_requirements_rejection_events_do_not_expose_tokens();
}

#[test]
fn release_attestation_status_surfaces_in_api_and_runtime_data() {
    runtime_data::tests::assert_release_attestation_status_surfaces_in_api_and_runtime_data();
}

#[test]
fn release_attestation_policy_accepts_trusted_signer() {
    mesh::tests::assert_mesh_requirements_outbound_admits_compliant_peer_after_requirements_pass();
}

#[test]
fn release_attestation_policy_accepts_trusted_signer_with_compatible_different_peer_version() {
    mesh::requirements::tests::assert_mesh_requirements_accept_trusted_signer_with_compatible_peer_version();
}

#[test]
fn release_attestation_policy_rejects_missing_status() {
    mesh::tests::assert_mesh_requirements_inbound_rejects_before_topology_announcement();
}

#[test]
fn release_attestation_policy_rejects_invalid_signature() {
    mesh::tests::assert_mesh_requirements_add_peer_rejects_invalid_release_attestation_signature();
}

#[test]
fn release_attestation_reports_missing_for_unstamped_binary() {
    runtime::assert_release_attestation_reports_missing_for_unstamped_binary();
}

#[test]
fn mixed_version_peer_ignores_missing_release_attestation() {
    protocol::tests::assert_mixed_version_peer_ignores_missing_release_attestation();
}
