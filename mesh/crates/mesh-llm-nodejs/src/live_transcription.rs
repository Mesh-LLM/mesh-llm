//! Native live-transcription sessions for Node.js.

use super::{Node, to_napi_error};
use mesh_llm_sdk::live_transcription::{
    LiveTranscriptionOptions, session::LiveTranscriptionSession,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub struct LiveTranscriptionNative {
    session: LiveTranscriptionSession,
}

#[napi]
impl Node {
    #[napi(js_name = "startLiveTranscriptionJson")]
    pub async fn start_live_transcription_json(
        &self,
        options_json: String,
    ) -> Result<LiveTranscriptionNative> {
        let options: LiveTranscriptionOptions =
            serde_json::from_str(&options_json).map_err(to_napi_error)?;
        let node = self.node.lock().await;
        let node = node
            .as_ref()
            .ok_or_else(|| Error::from_reason("node is not running"))?;
        let live = node
            .start_live_transcription(options)
            .await
            .map_err(to_napi_error)?;
        Ok(LiveTranscriptionNative {
            session: LiveTranscriptionSession::new(live),
        })
    }
}

#[napi]
impl LiveTranscriptionNative {
    #[napi(js_name = "sendAudio")]
    pub async fn send_audio(&self, samples: Vec<i16>) -> Result<()> {
        self.session
            .send_audio(samples)
            .await
            .map_err(to_napi_error)
    }

    #[napi(js_name = "sendAudioF32")]
    pub async fn send_audio_f32(&self, samples: Vec<f64>) -> Result<()> {
        // N-API numbers are doubles. Round to the same f32 PCM accepted by the
        // Rust SDK; its shared encoder then clips out-of-range audio values.
        let samples = samples.into_iter().map(|sample| sample as f32).collect();
        self.session
            .send_audio_f32(samples)
            .await
            .map_err(to_napi_error)
    }

    #[napi]
    pub async fn commit(&self) -> Result<()> {
        self.session.commit().await.map_err(to_napi_error)
    }

    #[napi(js_name = "nextEventJson")]
    pub async fn next_event_json(&self) -> Result<Option<String>> {
        self.session
            .next_event()
            .await
            .map(|event| serde_json::to_string(&event).map_err(to_napi_error))
            .transpose()
    }

    #[napi(js_name = "stopJson")]
    pub async fn stop_json(&self) -> Result<String> {
        serde_json::to_string(&self.session.stop().await.map_err(to_napi_error)?)
            .map_err(to_napi_error)
    }

    #[napi]
    pub async fn close(&self) {
        self.session.close().await;
    }
}
