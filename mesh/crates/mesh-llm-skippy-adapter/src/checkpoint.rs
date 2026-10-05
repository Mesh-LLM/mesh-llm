//! Mesh console rendering for prepared checkpoints.
use mesh_llm_events::{OutputEvent, emit_event};
use skippy_runtime::CheckpointQuantization;
use std::path::Path;

pub(super) fn emit_load_notice(
    model_path: &Path,
    quantization: CheckpointQuantization,
    has_imatrix: bool,
) {
    if !skippy_runtime::is_safetensors_checkpoint(model_path) {
        return;
    }
    let imatrix_status = if has_imatrix { "configured" } else { "none" };
    let message = format!(
        "SafeTensors native loader: quantization={} imatrix={imatrix_status}. Set with `mesh-llm serve --quant <RECIPE>`; see `mesh-llm serve --help-advanced` for valid recipes.",
        quantization.canonical_name()
    );
    let event = if quantization == CheckpointQuantization::Preserve {
        OutputEvent::Info {
            message,
            context: None,
        }
    } else {
        OutputEvent::Warning {
            message,
            context: None,
        }
    };
    let _ = emit_event(event);
}
