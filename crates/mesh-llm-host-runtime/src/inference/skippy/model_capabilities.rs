use anyhow::{Context, Result};

use super::SkippyModelHandle;

impl SkippyModelHandle {
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
}
