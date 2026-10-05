use serde_json::json;
use skippy_protocol::{StageConfig, binary::StageWireMessage};

use crate::{
    kv_integration::{KvStageIntegration, ResidentActivationRecord},
    telemetry::Telemetry,
};

use super::binary_kv::{BinaryKvRecordResult, binary_message_kv_attrs};

pub(super) fn add_binary_activation_records(
    result: &mut BinaryKvRecordResult,
    config: &StageConfig,
    kv: &KvStageIntegration,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    records: &[ResidentActivationRecord],
) {
    for record in records {
        accumulate_record(result, record);
        let mut attrs =
            binary_message_kv_attrs(config, kv, session_id, message, record.token_count);
        attrs.insert("skippy.kv.decision".to_string(), json!("activation_record"));
        attrs.insert(
            "skippy.activation_cache.payload_bytes".to_string(),
            json!(record.payload_bytes),
        );
        attrs.insert(
            "skippy.activation_cache.entries".to_string(),
            json!(record.entries),
        );
        attrs.insert(
            "skippy.activation_cache.resident_bytes".to_string(),
            json!(record.resident_bytes),
        );
        telemetry.emit("stage.binary_kv_record_decision", attrs);
    }
}

fn accumulate_record(result: &mut BinaryKvRecordResult, record: &ResidentActivationRecord) {
    result.recorded_activations = result.recorded_activations.saturating_add(1);
    result.recorded_activation_bytes = result
        .recorded_activation_bytes
        .saturating_add(record.payload_bytes as u64);
    result.evicted_activation_entries = result
        .evicted_activation_entries
        .saturating_add(record.evicted_entries);
    result.evicted_activation_bytes = result
        .evicted_activation_bytes
        .saturating_add(record.evicted_bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn activation_record(token_count: usize, payload_bytes: usize) -> ResidentActivationRecord {
        ResidentActivationRecord {
            page_id: format!("page-{token_count}"),
            token_count,
            payload_bytes,
            evicted_entries: 1,
            evicted_bytes: 4,
            entries: 2,
            resident_bytes: 16,
        }
    }

    #[test]
    fn binary_activation_stats_include_each_record_identity() {
        let mut result = BinaryKvRecordResult::default();

        accumulate_record(&mut result, &activation_record(2214, 8_856));
        accumulate_record(&mut result, &activation_record(2176, 8_704));

        assert_eq!(result.recorded_activations, 2);
        assert_eq!(result.recorded_activation_bytes, 17_560);
        assert_eq!(result.evicted_activation_entries, 2);
        assert_eq!(result.evicted_activation_bytes, 8);
    }
}
