#![recursion_limit = "256"]

mod api;
mod cli;
pub mod crypto;
mod inference;
mod mesh;
mod models;
mod network;
mod plugin;
mod plugins;
mod protocol;
mod runtime;
mod runtime_data;
mod system;

pub mod proto {
    pub mod node {
        include!(concat!(env!("OUT_DIR"), "/meshllm.node.v1.rs"));
    }
}

pub(crate) use plugins::blackboard;

use anyhow::Result;
use std::time::Duration;

pub const VERSION: &str = "0.65.0-rc2";

pub async fn run() -> Result<()> {
    runtime::run().await
}

pub async fn run_main() -> i32 {
    match run().await {
        Ok(()) => 0,
        Err(err) => {
            let _ = cli::output::emit_fatal_error(&err);
            tokio::time::sleep(Duration::from_millis(50)).await;
            1
        }
    }
}

#[cfg(test)]
#[test]
fn runtime_data_collector_shell_constructs_and_clones() {
    runtime_data::test_support::runtime_data_collector_shell_constructs_and_clones();
}

#[cfg(test)]
#[test]
fn runtime_data_collector_exposes_initial_snapshots() {
    runtime_data::test_support::runtime_data_collector_exposes_initial_snapshots();
}

#[cfg(test)]
#[test]
fn runtime_data_version_advances_and_marks_dirty_bits() {
    runtime_data::test_support::runtime_data_version_advances_and_marks_dirty_bits();
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_subscribe_notifies_once_per_update() {
    runtime_data::test_support::runtime_data_subscribe_notifies_once_per_update().await;
}

#[cfg(test)]
#[test]
fn runtime_data_process_snapshot_matches_existing_runtime_views() {
    runtime_data::test_support::runtime_data_process_snapshot_matches_existing_runtime_views();
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_inventory_single_flight_scan_coalesces() {
    runtime_data::test_support::runtime_data_inventory_single_flight_scan_coalesces().await;
}

#[cfg(test)]
#[test]
fn runtime_data_local_instance_snapshot_replaces_existing_scan_results() {
    runtime_data::test_support::runtime_data_local_instance_snapshot_replaces_existing_scan_results(
    );
}

#[cfg(test)]
#[tokio::test]
async fn api_runtime_reads_from_collector_snapshot() {
    api::test_support::api_runtime_reads_from_collector_snapshot().await;
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_collector_rule_guardrails_hold() {
    runtime_data::test_support::runtime_data_collector_rule_guardrails_hold().await;
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_routing_snapshot_reflects_proxy_attempts_and_inflight() {
    runtime_data::test_support::runtime_data_routing_snapshot_reflects_proxy_attempts_and_inflight(
    )
    .await;
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_request_updates_stay_non_blocking() {
    runtime_data::test_support::runtime_data_request_updates_stay_non_blocking().await;
}

#[cfg(test)]
#[test]
fn runtime_data_status_snapshot_matches_api_payloads() {
    runtime_data::test_support::runtime_data_status_snapshot_matches_api_payloads();
}

#[cfg(test)]
#[test]
fn runtime_data_model_snapshot_matches_api_payloads() {
    runtime_data::test_support::runtime_data_model_snapshot_matches_api_payloads();
}

#[cfg(test)]
#[test]
fn runtime_data_plugin_reports_are_scoped_by_name_and_endpoint() {
    runtime_data::test_support::runtime_data_plugin_reports_are_scoped_by_name_and_endpoint();
}

#[cfg(test)]
#[test]
fn runtime_data_plugin_clear_removes_only_target_plugin_reports() {
    runtime_data::test_support::runtime_data_plugin_clear_removes_only_target_plugin_reports();
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_public_payload_shapes_remain_unchanged() {
    api::test_support::runtime_data_public_payload_shapes_remain_unchanged().await;
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_inventory_and_plugin_contracts_hold() {
    runtime_data::test_support::runtime_data_inventory_and_plugin_contracts_hold().await;
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_llama_metrics_polling_records_success_and_samples() {
    runtime_data::test_support::runtime_data_llama_metrics_polling_records_success_and_samples()
        .await;
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_llama_metrics_polling_marks_unavailable_nonfatally() {
    runtime_data::test_support::runtime_data_llama_metrics_polling_marks_unavailable_nonfatally()
        .await;
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_llama_failed_refresh_preserves_previous_payloads() {
    runtime_data::test_support::runtime_data_llama_failed_refresh_preserves_previous_payloads()
        .await;
}

#[cfg(test)]
#[test]
fn runtime_data_llama_slots_json_parses_permissively() {
    runtime_data::test_support::runtime_data_llama_slots_json_parses_permissively();
}

#[cfg(test)]
#[test]
fn runtime_data_llama_slots_parsing_bounds_entries_and_large_json() {
    runtime_data::test_support::runtime_data_llama_slots_parsing_bounds_entries_and_large_json();
}

#[cfg(test)]
#[test]
fn runtime_data_llama_items_preserve_slot_index_and_busy_state() {
    runtime_data::test_support::runtime_data_llama_items_preserve_slot_index_and_busy_state();
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_api_routes_remain_payload_stable() {
    api::test_support::runtime_data_api_routes_remain_payload_stable().await;
}

#[cfg(test)]
#[tokio::test]
async fn runtime_data_sse_bridge_delivers_initial_and_incremental_updates() {
    api::test_support::runtime_data_sse_bridge_delivers_initial_and_incremental_updates().await;
}
