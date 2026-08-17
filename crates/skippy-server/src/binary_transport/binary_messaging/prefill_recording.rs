use std::sync::{Arc, Mutex};

use skippy_protocol::{StageConfig, binary::StageWireMessage};
use skippy_runtime::ActivationFrame;

use crate::{
    binary_transport::{
        activation_cache::add_binary_activation_records,
        binary_kv::{
            BinaryKvRecordResult, maybe_record_binary_full_prefill, maybe_record_binary_prefill,
        },
        stage_execution::binary_message_base,
    },
    kv_integration::KvStageIntegration,
    runtime_state::RuntimeState,
    telemetry::Telemetry,
};

enum PrefillRecordPlan<'a> {
    Full(&'a [i32]),
    Incremental {
        token_ids: &'a [i32],
        restored_tokens: u64,
    },
}

fn prefill_record_plan<'a>(
    accumulated_tokens: Option<&'a [i32]>,
    token_start: usize,
    token_ids: &'a [i32],
    restored_tokens: u64,
) -> PrefillRecordPlan<'a> {
    let logical_end = token_start.checked_add(token_ids.len());
    let chain_compatible = accumulated_tokens.filter(|tokens| {
        logical_end == Some(tokens.len()) && tokens.get(token_start..) == Some(token_ids)
    });
    chain_compatible.map_or(
        PrefillRecordPlan::Incremental {
            token_ids,
            restored_tokens,
        },
        PrefillRecordPlan::Full,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_completed_prefill(
    config: &StageConfig,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    accumulated_tokens: Option<&[i32]>,
    token_ids: &[i32],
    restored_tokens: u64,
    activation_width: i32,
    output: &ActivationFrame,
) -> BinaryKvRecordResult {
    match prefill_record_plan(
        accumulated_tokens,
        message.pos_start.max(0) as usize,
        token_ids,
        restored_tokens,
    ) {
        PrefillRecordPlan::Full(tokens) => record_full_prefill_with_activations(
            config,
            runtime,
            kv,
            telemetry,
            session_id,
            message,
            tokens,
            activation_width,
            output,
        ),
        PrefillRecordPlan::Incremental {
            token_ids,
            restored_tokens,
        } => maybe_record_binary_prefill(
            config,
            runtime,
            kv,
            telemetry,
            session_id,
            message,
            token_ids,
            restored_tokens,
            activation_width,
            Some(output),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn record_full_prefill_with_activations(
    config: &StageConfig,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    tokens: &[i32],
    activation_width: i32,
    output: &ActivationFrame,
) -> BinaryKvRecordResult {
    let mut runtime = runtime.lock().expect("runtime lock poisoned");
    let mut record = maybe_record_binary_full_prefill(
        config,
        &mut runtime,
        kv,
        telemetry,
        session_id,
        message,
        tokens,
    );
    drop(runtime);
    if let Some(kv) = kv
        && config.downstream.is_some()
    {
        let base = binary_message_base(config, session_id, message);
        let activations =
            kv.record_resident_activation(config, &base, 0, tokens, activation_width, output);
        add_binary_activation_records(
            &mut record,
            config,
            kv,
            telemetry,
            session_id,
            message,
            &activations,
        );
    }
    record
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulated_prompt_takes_full_prefill_recording_path() {
        let accumulated = [1, 2, 3, 4];
        let message = [3, 4];

        let plan = prefill_record_plan(Some(&accumulated), 2, &message, 2);

        assert!(matches!(
            plan,
            PrefillRecordPlan::Full(tokens) if tokens == accumulated
        ));
    }

    #[test]
    fn message_tokens_take_incremental_recording_path_without_accumulation() {
        let message = [3, 4];

        let plan = prefill_record_plan(None, 2, &message, 2);

        assert!(matches!(
            plan,
            PrefillRecordPlan::Incremental {
                token_ids,
                restored_tokens: 2,
            } if token_ids == message
        ));
    }

    #[test]
    fn completed_chunked_prefill_uses_chain_compatible_logical_prompt() {
        let accumulated = (0..4905).collect::<Vec<i32>>();
        let remainder = &accumulated[4096..];

        let plan = prefill_record_plan(Some(&accumulated), 4096, remainder, 0);

        assert!(matches!(
            plan,
            PrefillRecordPlan::Full(tokens) if tokens.len() == 4905
        ));
    }

    #[test]
    fn mismatched_accumulation_cannot_be_used_for_full_prefill_recording() {
        let accumulated = (0..4905).collect::<Vec<i32>>();
        let mut remainder = accumulated[4096..].to_vec();
        remainder[0] = -1;

        let plan = prefill_record_plan(Some(&accumulated), 4096, &remainder, 0);

        assert!(matches!(plan, PrefillRecordPlan::Incremental { .. }));
    }
}
