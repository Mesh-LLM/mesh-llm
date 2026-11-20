use super::*;

pub(crate) fn load_laya_model(
    path: &Path,
    device: Option<&str>,
) -> Result<Arc<skippy_runtime::LayaModel>> {
    crate::system::native_runtime_requirement::ensure_native_runtime_available()?;
    let threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4);
    Ok(Arc::new(skippy_runtime::LayaModel::open(
        path, threads, device,
    )?))
}

pub(crate) fn start_laya_http_on(
    model_id: &str,
    model: Arc<skippy_runtime::LayaModel>,
    bind_addr: std::net::SocketAddr,
) -> SkippyHttpHandle {
    let lifecycle_observer = crate::plugin::inference_observation::adapter::observe_frontend(
        crate::network::openai::runtime_events::compose_lifecycle_observer(
            crate::logging_runtime_state().and_then(|state| state.openai_lifecycle_observer()),
        ),
        model_id,
    );
    let server = skippy_serving::start_openai_backend_with_lifecycle_observer(
        bind_addr,
        Arc::new(skippy_serving::LayaSystemOneBackend::new(model_id, model)),
        lifecycle_observer,
    );
    SkippyHttpHandle {
        port: bind_addr.port(),
        native_backend: None,
        server,
    }
}

pub(crate) fn load_speech_recognition_model(
    path: &Path,
    device: Option<&str>,
) -> Result<Arc<skippy_runtime::SpeechRecognitionModel>> {
    crate::system::native_runtime_requirement::ensure_native_runtime_available()?;
    Ok(Arc::new(skippy_runtime::SpeechRecognitionModel::open(
        path, device,
    )?))
}

pub(crate) fn start_speech_recognition_http_on(
    model_id: &str,
    model: Arc<skippy_runtime::SpeechRecognitionModel>,
    concurrency: usize,
    bind_addr: std::net::SocketAddr,
) -> SkippyHttpHandle {
    let lifecycle_observer = crate::plugin::inference_observation::adapter::observe_frontend(
        crate::network::openai::runtime_events::compose_lifecycle_observer(
            crate::logging_runtime_state().and_then(|state| state.openai_lifecycle_observer()),
        ),
        model_id,
    );
    let backend: Arc<dyn InferenceBackend> = Arc::new(
        skippy_serving::SpeechRecognitionBackend::new(model_id, model, concurrency),
    );
    let server = skippy_serving::start_openai_backend_with_lifecycle_observer(
        bind_addr,
        backend.clone(),
        lifecycle_observer,
    );
    SkippyHttpHandle {
        port: bind_addr.port(),
        native_backend: Some(backend),
        server,
    }
}
