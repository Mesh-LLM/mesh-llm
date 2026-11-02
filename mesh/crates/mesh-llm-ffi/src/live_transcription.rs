//! Native live-transcription sessions for Python, Swift, and Kotlin.

use crate::{FfiError, MeshNodeHandle, runtime_blocking::block_on};
use mesh_llm_sdk::live_transcription::{
    LiveTranscriptionOptions, session::LiveTranscriptionSession,
};
use std::sync::Arc;

#[derive(uniffi::Object)]
pub struct LiveTranscriptionNative {
    session: LiveTranscriptionSession,
}

#[uniffi::export]
impl MeshNodeHandle {
    /// Start a session with the Rust SDK's defaults and role checks.
    pub fn start_live_transcription_json(
        &self,
        options_json: String,
    ) -> Result<Arc<LiveTranscriptionNative>, FfiError> {
        let options: LiveTranscriptionOptions =
            serde_json::from_str(&options_json).map_err(stream_error)?;
        let node = self.node.lock().map_err(stream_error)?;
        let node = node
            .as_ref()
            .ok_or_else(|| FfiError::HostUnavailable("node is not running".into()))?;
        let session = block_on(async {
            node.start_live_transcription(options)
                .await
                .map(LiveTranscriptionSession::new)
        })
        .map_err(stream_error)?;
        Ok(Arc::new(LiveTranscriptionNative { session }))
    }
}

#[uniffi::export]
impl LiveTranscriptionNative {
    pub fn send_audio(&self, samples: Vec<i16>) -> Result<(), FfiError> {
        block_on(self.session.send_audio(samples)).map_err(stream_error)
    }

    pub fn send_audio_f32(&self, samples: Vec<f32>) -> Result<(), FfiError> {
        block_on(self.session.send_audio_f32(samples)).map_err(stream_error)
    }

    pub fn commit(&self) -> Result<(), FfiError> {
        block_on(self.session.commit()).map_err(stream_error)
    }

    pub fn next_event_json(&self) -> Result<Option<String>, FfiError> {
        block_on(self.session.next_event())
            .map(|event| serde_json::to_string(&event).map_err(stream_error))
            .transpose()
    }

    pub fn stop_json(&self) -> Result<String, FfiError> {
        let events = block_on(self.session.stop()).map_err(stream_error)?;
        serde_json::to_string(&events).map_err(stream_error)
    }

    pub fn close_audio(&self) {
        block_on(self.session.close());
    }
}

fn stream_error(error: impl ToString) -> FfiError {
    FfiError::StreamFailed(error.to_string())
}
