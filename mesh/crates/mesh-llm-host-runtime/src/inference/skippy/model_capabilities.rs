use anyhow::{Context, Result};

use super::SkippyModelHandle;
use skippy_inference_api::thinking::ThinkingControls;

impl SkippyModelHandle {
    /// Reuse buffer measurements only when the native workload preserves requested lanes.
    pub(crate) fn permits_memory_measurement_reuse(&self) -> bool {
        workload_preserves_memory_lanes(self.runtime.workload_info().ok())
    }

    /// Classify the loaded native runtime, including speech-capable projectors.
    pub(crate) fn workload_class(&self) -> Result<crate::mesh::ModelWorkloadClass> {
        if self.runtime.supports_speech_synthesis() {
            return Ok(crate::mesh::ModelWorkloadClass::SpeechSynthesis);
        }
        let workload = self
            .runtime
            .workload_info()
            .context("read loaded model workload contract")?;
        Ok(match workload.kind {
            skippy_runtime::ModelWorkload::CausalGeneration => {
                crate::mesh::ModelWorkloadClass::CausalGeneration
            }
            skippy_runtime::ModelWorkload::Embedding => crate::mesh::ModelWorkloadClass::Embedding,
            skippy_runtime::ModelWorkload::Rerank => crate::mesh::ModelWorkloadClass::Rerank,
            skippy_runtime::ModelWorkload::EncoderDecoder => {
                crate::mesh::ModelWorkloadClass::EncoderDecoder
            }
        })
    }

    /// Advertise System One only when this exact runtime can execute the endpoint.
    pub(crate) fn supports_system_one(&self) -> bool {
        self.runtime.supports_system_one()
    }

    /// Render-only reasoning-control observations Skippy produced for this model.
    ///
    /// The host never interprets the value; it republishes it on `/v1/models`.
    pub(crate) fn thinking(&self) -> Option<&ThinkingControls> {
        self.thinking.as_ref()
    }
}

fn workload_preserves_memory_lanes(workload: Option<skippy_runtime::WorkloadInfo>) -> bool {
    workload.is_some_and(|workload| {
        workload.kind != skippy_runtime::ModelWorkload::EncoderDecoder
            && !(workload.has_encoder && workload.has_decoder)
    })
}

#[cfg(test)]
mod tests {
    use super::workload_preserves_memory_lanes;
    use skippy_runtime::{ModelWorkload, PoolingType, WorkloadInfo};

    fn workload(kind: ModelWorkload, has_encoder: bool, has_decoder: bool) -> WorkloadInfo {
        WorkloadInfo {
            kind,
            pooling: PoolingType::None,
            output_dimensions: 0,
            classifier_outputs: 0,
            has_encoder,
            has_decoder,
            full_model_only: true,
        }
    }

    #[test]
    fn memory_measurement_reuse_rejects_unknown_and_encoder_decoder_workloads() {
        assert!(!workload_preserves_memory_lanes(None));
        for kind in [
            ModelWorkload::EncoderDecoder,
            ModelWorkload::CausalGeneration,
        ] {
            assert!(!workload_preserves_memory_lanes(Some(workload(
                kind, true, true
            ))));
        }
        assert!(!workload_preserves_memory_lanes(Some(workload(
            ModelWorkload::EncoderDecoder,
            false,
            false
        ))));
    }

    #[test]
    fn memory_measurement_reuse_preserves_causal_embedding_and_rerank_workloads() {
        for (kind, encoder, decoder) in [
            (ModelWorkload::CausalGeneration, false, true),
            (ModelWorkload::Embedding, true, false),
            (ModelWorkload::Rerank, true, false),
        ] {
            assert!(workload_preserves_memory_lanes(Some(workload(
                kind, encoder, decoder
            ))));
        }
    }
}
