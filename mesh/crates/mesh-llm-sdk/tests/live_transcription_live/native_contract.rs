//! Opt-in public Rust SDK speech qualification, never an internal SpeechStream test.
//!
//! Required environment: MESH_SDK_LIVE_SPEECH=1, MESH_LLM_NO_RUNTIME_DOWNLOAD=1,
//! MESH_SPEECH_INPUTS, MESH_SPEECH_EVIDENCE, MESH_SDK_STATE_ROOT, three explicit
//! MESH_SDK_{API,CONSOLE,BIND}_PORT values, MESH_SDK_TEST_ARTIFACT/SHA256,
//! MESH_SDK_RUNTIME_MANIFEST/SHA256 and the isolated state variables below.
//! The enclosing foreground test process needs its own operator-owned deadline.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::future::Future;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use mesh_llm_sdk::embedded_node::EmbeddedMeshNodeConfig;
use mesh_llm_sdk::live_transcription::session::LiveTranscriptionSession;
use mesh_llm_sdk::live_transcription::{LiveTranscriptionEvent, LiveTranscriptionOptions};
use mesh_llm_sdk::models::{
    MeshModel, MeshModelAvailability, MeshModelKind, MeshModelQuery, ModelWorkloadClass,
};
use mesh_llm_sdk::native_runtime::{self, NativeRuntimeManifest, RuntimeSelection};
use mesh_llm_sdk::{MeshNode, MeshNodeBuilder};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::task::JoinHandle;

const PCM_SHA256: &str = "a29462b8ebd467318000e683b9117ade46230d3255ed2024e7db894abd9b38c9";
const MODEL_SHA256: &str = "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21";
const ISOLATION_KEYS: &[&str] = &[
    "MESH_LLM_CONFIG",
    "MESH_LLM_RUNTIME_ROOT",
    "MESH_LLM_DATA_DIR",
    "MESH_LLM_PLUGIN_DIR",
    "MESH_LLM_NODE_KEY_PATH",
    "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
];

struct Inputs {
    model: PathBuf,
    pcm: Vec<i16>,
    evidence: PathBuf,
    ports: [u16; 3],
    manifest_path: PathBuf,
    manifest: NativeRuntimeManifest,
    executable: PathBuf,
}

impl Inputs {
    /// Reads and validates the admitted inputs from the environment.
    fn read() -> Result<Self> {
        ensure!(
            required("MESH_SDK_LIVE_SPEECH")? == "1",
            "Explicit speech opt-in is required"
        );
        ensure!(
            required("MESH_LLM_NO_RUNTIME_DOWNLOAD")? == "1",
            "Runtime startup must be offline"
        );
        ensure!(
            std::env::var_os("MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL").is_none(),
            "Do not configure a remote runtime manifest"
        );
        validate_isolation()?;
        let source = existing_path("MESH_SPEECH_INPUTS")?;
        let audio = fs::read(source.join("jfk.s16le"))?;
        ensure!(
            audio.len() == 352_000,
            "Expected exactly 11 seconds of PCM16LE"
        );
        ensure!(
            digest_hex(&Sha256::digest(&audio))? == PCM_SHA256,
            "PCM digest mismatch"
        );
        let model = source.join("ggml-tiny.bin");
        ensure!(
            hash_file(&model)? == MODEL_SHA256,
            "Tiny model digest mismatch"
        );
        let executable = std::env::current_exe()?.canonicalize()?;
        ensure!(
            executable == existing_path("MESH_SDK_TEST_ARTIFACT")?,
            "Unexpected running Rust test artifact"
        );
        ensure!(
            hash_file(&executable)? == required("MESH_SDK_TEST_SHA256")?,
            "Rust test artifact digest mismatch"
        );
        let manifest_path = existing_path("MESH_SDK_RUNTIME_MANIFEST")?;
        ensure!(
            manifest_path.starts_with(existing_path("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR")?),
            "Manifest must belong to the selected bundle"
        );
        ensure!(
            hash_file(&manifest_path)? == required("MESH_SDK_RUNTIME_MANIFEST_SHA256")?,
            "Runtime manifest digest mismatch"
        );
        let manifest: NativeRuntimeManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        ensure!(
            manifest.runtime.skippy_abi == native_runtime::current_skippy_abi_version(),
            "Runtime ABI differs from the compiled SDK"
        );
        ensure!(
            !manifest.runtime.files.is_empty(),
            "Runtime must declare file digests"
        );
        manifest.verify_payload(manifest_path.parent().context("runtime manifest parent")?)?;
        let ports = [
            port("MESH_SDK_API_PORT")?,
            port("MESH_SDK_CONSOLE_PORT")?,
            port("MESH_SDK_BIND_PORT")?,
        ];
        ensure!(
            ports.iter().collect::<BTreeSet<_>>().len() == 3,
            "Fixture ports must be distinct"
        );
        let evidence = absolute_path("MESH_SPEECH_EVIDENCE")?;
        require_fresh(&evidence)?;
        fs::create_dir_all(&evidence)?;
        Ok(Self {
            model,
            evidence,
            ports,
            manifest_path,
            manifest,
            executable,
            pcm: audio
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| i16::from_le_bytes(*pair))
                .collect(),
        })
    }

    /// Builds the embedded node config from the admitted inputs.
    fn config(&self) -> Result<EmbeddedMeshNodeConfig> {
        Ok(serde_json::from_value(json!({
            "mode": "serve",
            "http": {"api_port": self.ports[0], "console_port": self.ports[1], "console_ui": false},
            "serving": {"models": [self.model]},
            "storage": {"isolated_config": true},
            "network": {"auto_join": false, "publish": false, "enumerate_host": true,
                "bind_ip": "127.0.0.1", "bind_port": self.ports[2], "disable_iroh_relays": true,
                "discovery_mode": "mdns"},
            "startup_timeout": {"secs": 60, "nanos": 0}
        }))?)
    }

    /// Writes one evidence record under the fresh evidence directory.
    fn write(&self, name: &str, value: &Value) -> Result<()> {
        fs::write(self.evidence.join(name), serde_json::to_vec_pretty(value)?)?;
        Ok(())
    }

    /// Confirms the fixture ports are free before the node binds them.
    fn ports_free(&self) -> Result<()> {
        for port in &self.ports[..2] {
            let socket = tokio::net::TcpSocket::new_v4()?;
            socket.set_reuseaddr(true)?;
            socket
                .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, *port)))
                .with_context(|| format!("owned TCP port {port} is not free"))?;
            let listener = socket.listen(1)?;
            drop(listener);
        }
        let socket = UdpSocket::bind((std::net::Ipv4Addr::LOCALHOST, self.ports[2]))
            .context("owned UDP port is not free")?;
        drop(socket);
        Ok(())
    }
}

/// Reads a non-blank required environment variable.
fn required(key: &str) -> Result<String> {
    let value = std::env::var(key).with_context(|| format!("Missing {key}"))?;
    ensure!(!value.is_empty(), "Missing {key}");
    Ok(value)
}

/// Reads a required environment variable as an absolute path.
fn absolute_path(key: &str) -> Result<PathBuf> {
    let path = PathBuf::from(required(key)?);
    ensure!(
        path.is_absolute()
            && !path
                .components()
                .any(|part| matches!(part, Component::ParentDir)),
        "{key} must be an absolute path without parent traversal"
    );
    Ok(path)
}

/// Reads a required environment variable as an existing absolute path.
fn existing_path(key: &str) -> Result<PathBuf> {
    Ok(absolute_path(key)?.canonicalize()?)
}

/// Requires a path to be fresh: absent or an empty directory.
fn require_fresh(path: &Path) -> Result<()> {
    if path.exists() {
        ensure!(
            path.is_dir() && fs::read_dir(path)?.next().is_none(),
            "Fixture state/evidence must be absent or an empty directory"
        );
    }
    Ok(())
}

/// Validates the isolated state variables against the state root.
fn validate_isolation() -> Result<()> {
    let state = existing_path("MESH_SDK_STATE_ROOT")?;
    ensure!(state.parent().is_some(), "State root must be isolated");
    if let Some(home) = std::env::var_os("HOME") {
        ensure!(
            state != PathBuf::from(home).canonicalize()?,
            "Do not use HOME as fixture state"
        );
    }
    for key in ISOLATION_KEYS {
        let path = absolute_path(key)?;
        let existing = path
            .ancestors()
            .find(|ancestor| ancestor.exists())
            .context("existing state ancestor")?;
        // Canonicalize the existing prefix even when the leaf is fresh. On
        // macOS /tmp is /private/tmp; a lexical comparison would reject it.
        let resolved = existing.canonicalize()?.join(path.strip_prefix(existing)?);
        ensure!(
            resolved.starts_with(&state) && resolved != state,
            "{key} escapes the state root through a symlink"
        );
        require_fresh(&resolved)?;
    }
    Ok(())
}

/// Reads a required environment variable as a fixture port.
fn port(key: &str) -> Result<u16> {
    let port = required(key)?.parse()?;
    ensure!(port >= 1024, "{key} must be an unprivileged port");
    Ok(port)
}

/// Returns the SHA-256 of a file as lowercase hexadecimal.
fn hash_file(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    let mut file = File::open(path)?;
    let mut buffer = [0_u8; 65_536];
    loop {
        let size = file.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        hash.update(&buffer[..size]);
    }
    digest_hex(&hash.finalize())
}

/// Renders a byte slice as lowercase hexadecimal.
fn digest_hex(bytes: &[u8]) -> Result<String> {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        std::fmt::Write::write_fmt(&mut encoded, format_args!("{byte:02x}"))?;
    }
    Ok(encoded)
}

/// Awaits a future under a deadline, naming the timeout failure.
async fn bounded<T>(seconds: u64, label: &str, future: impl Future<Output = T>) -> Result<T> {
    tokio::time::timeout(Duration::from_secs(seconds), future)
        .await
        .with_context(|| format!("{label} timed out; fallback cleanup is not a pass"))
}

pub(super) async fn run() -> Result<()> {
    let inputs = Inputs::read()?;
    inputs.ports_free()?;
    prepare_runtime(&inputs).await?;
    bounded(
        30,
        "native initialization",
        mesh_llm_sdk::initialize_host_runtime(),
    )
    .await??;
    let node = bounded(
        75,
        "combined node startup",
        MeshNodeBuilder::from_config(inputs.config()?).start(),
    )
    .await??;
    let outcome = exercise(&node, &inputs).await;
    // Public Rust stop consumes the node. Successful shutdown and free owned
    // ports are its cleanup evidence; a post-stop status handle does not exist.
    let stopped = bounded(30, "node shutdown", node.stop())
        .await
        .and_then(std::convert::identity);
    let ports = inputs.ports_free();
    inputs.write("cleanup.json", &json!({"shutdown_completed": stopped.is_ok(), "owned_ports_free": ports.is_ok(), "ports": inputs.ports}))?;
    inputs.write(
        "outcome.json",
        &json!({"passed": outcome.is_ok() && stopped.is_ok() && ports.is_ok()}),
    )?;
    outcome?;
    stopped?;
    ports
}

/// Installs the admitted runtime into the isolated state root.
async fn prepare_runtime(inputs: &Inputs) -> Result<()> {
    let root = inputs.manifest_path.parent().context("runtime root")?;
    let release = inputs
        .manifest
        .runtime
        .release_version
        .clone()
        .unwrap_or_else(|| native_runtime::current_runtime_release().to_owned());
    let mut options = native_runtime::NativeRuntimeInstallOptions::new(
        release,
        native_runtime::mesh_native_runtime_catalog(),
    );
    options.selection = RuntimeSelection::Id(inputs.manifest.runtime.id.clone());
    options.skippy_abi_version = Some(native_runtime::current_skippy_abi_version());
    options.bundle_dirs = vec![root.to_path_buf()];
    options.cache_dir = Some(absolute_path("MESH_LLM_NATIVE_RUNTIME_CACHE_DIR")?);
    options.allow_download = false;
    let outcome = bounded(
        30,
        "offline runtime admission",
        native_runtime::install_native_runtime_explicit(options),
    )
    .await??;
    ensure!(
        outcome.runtime.path.canonicalize()? == root.canonicalize()?,
        "Resolver selected a different runtime root"
    );
    inputs.write("identity.json", &json!({
        "pid": std::process::id(), "version": native_runtime::CURRENT_MESH_VERSION,
        "test_executable": inputs.executable, "test_sha256": hash_file(&inputs.executable)?,
        "runtime_manifest": inputs.manifest_path, "runtime_manifest_sha256": hash_file(&inputs.manifest_path)?,
        "runtime_id": outcome.runtime.native_runtime_id, "runtime_release": outcome.runtime.release_version,
        "source_identity": "Requires the separate exact-source test-build/link receipt", "ports": inputs.ports
    }))
}

/// Runs the full SDK exercise against the started node.
async fn exercise(node: &MeshNode, inputs: &Inputs) -> Result<()> {
    let status = bounded(5, "node status", node.status()).await??;
    ensure!(
        status.api_base_url == node.api_base_url(),
        "Live node status reports a different API endpoint"
    );
    let model = bounded(45, "native speech discovery", discover(node, inputs)).await??;
    record_mapped_runtime(inputs)?;
    for step in [199, 5001] {
        match bounded(
            35,
            "invalid cadence",
            node.start_live_transcription(options(&model, step)),
        )
        .await?
        {
            Ok(live) => {
                bounded(
                    5,
                    "invalid session cleanup",
                    LiveTranscriptionSession::new(live).close(),
                )
                .await?;
                bail!("Invalid step {step} created a session");
            }
            Err(error) => {
                let message = error.to_string();
                ensure!(
                    message.contains("200") && message.contains("5000"),
                    "Invalid cadence did not report the supported range"
                );
                inputs.write(
                    &format!("invalid-{step}.json"),
                    &json!({"step_ms": step, "error_type": "anyhow::Error", "message": message}),
                )?;
            }
        }
    }
    transcribe(node, inputs, &model).await?;
    cancel_blocked_read(node, inputs, &model).await
}

/// Builds the live transcription options for the fixture model.
fn options(model: &str, step: u32) -> LiveTranscriptionOptions {
    LiveTranscriptionOptions {
        model: Some(model.to_owned()),
        language: Some("en".into()),
        sample_rate: 16000,
        turn_detection: None,
        step_ms: Some(step),
        tentative: true,
        ..Default::default()
    }
}

/// Reports whether the model is ready to serve the speech workload.
fn ready(model: &MeshModel) -> bool {
    model.kind == MeshModelKind::Model
        && model.workload_class == Some(ModelWorkloadClass::SpeechRecognition)
        && model.availability == MeshModelAvailability::Available
        && model
            .operations
            .iter()
            .any(|operation| operation == "live_transcription")
        && model.providers.iter().any(|provider| {
            provider.local
                && provider.available
                && provider
                    .runtime
                    .as_ref()
                    .is_some_and(|runtime| runtime.ready)
        })
}

/// Waits for the speech model to appear on the node.
async fn discover(node: &MeshNode, inputs: &Inputs) -> Result<String> {
    // Subscribe to runtime status changes before the first list, so a model
    // becoming ready between a list and the wait still wakes the next list.
    let mut changes = reqwest::Client::builder()
        .no_proxy()
        .build()?
        .get(format!("{}/api/events", node.console_url()))
        .send()
        .await?
        .error_for_status()?;
    loop {
        let page = bounded(
            10,
            "native model list",
            node.models().list(MeshModelQuery {
                workload_class: Some(ModelWorkloadClass::SpeechRecognition),
                available: Some(true),
                ..Default::default()
            }),
        )
        .await??;
        if let Some(model) = page.data.into_iter().find(ready) {
            let exact = bounded(10, "native model get", node.models().get(&model.id)).await??;
            ensure!(
                exact.id == model.id && ready(&exact),
                "Native get must preserve the ready speech identity"
            );
            ensure!(
                exact.operations == model.operations
                    && exact.workload_class == model.workload_class,
                "Native list/get capability mismatch"
            );
            inputs.write(
                "native-discovery.json",
                &json!({"id": exact.id, "kind": exact.kind,
                "workload_class": exact.workload_class, "availability": exact.availability,
                "operations": exact.operations, "local_available_runtime_ready": true}),
            )?;
            return Ok(exact.id);
        }
        // Each status update or server keepalive triggers another list.
        changes
            .chunk()
            .await?
            .context("Runtime event stream ended before speech became ready")?;
    }
}

/// Records the mapped runtime files for the provenance evidence.
fn record_mapped_runtime(inputs: &Inputs) -> Result<()> {
    let map = process_map()?;
    let root = inputs.manifest_path.parent().context("runtime root")?;
    let expected = inputs
        .manifest
        .runtime
        .files
        .iter()
        .map(|(name, hash)| Ok((root.join(name).canonicalize()?, hash.clone())))
        .collect::<std::io::Result<BTreeMap<_, _>>>()?;
    let mut mapped = BTreeMap::new();
    for line in map.lines().filter(|line| {
        ["libggml", "libllama.", "libmtmd"]
            .iter()
            .any(|name| line.contains(name))
    }) {
        let (path, hash) = expected
            .iter()
            .find(|(path, _)| line.trim_end().ends_with(path.to_string_lossy().as_ref()))
            .context("Mapped runtime image is outside the admitted runtime")?;
        mapped.insert(path.clone(), hash.clone());
    }
    ensure!(
        mapped.keys().any(|path| path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("libllama."))),
        "No actual mapped llama runtime found"
    );
    let images = mapped
        .into_iter()
        .map(|(path, expected)| {
            let actual = hash_file(&path)?;
            ensure!(actual == expected, "Mapped runtime image digest mismatch");
            Ok(json!({"sha256": actual, "path": path}))
        })
        .collect::<Result<Vec<_>>>()?;
    inputs.write(
        "model-loaded-identity.json",
        &json!({"mapped_runtime_images": images,
        "runtime_manifest_sha256": hash_file(&inputs.manifest_path)?}),
    )
}

/// Reads the process memory map under its platform name.
#[cfg(target_os = "linux")]
fn process_map() -> Result<String> {
    Ok(fs::read_to_string("/proc/self/maps")?)
}

/// Reads the process memory map under its platform name.
#[cfg(target_os = "macos")]
fn process_map() -> Result<String> {
    use std::ffi::{CStr, OsStr, c_char};
    use std::os::unix::ffi::OsStrExt;

    unsafe extern "C" {
        fn _dyld_image_count() -> u32;
        fn _dyld_get_image_name(index: u32) -> *const c_char;
    }
    // SAFETY: dyld's no-argument query reads this process's loaded-image count.
    let count = unsafe { _dyld_image_count() };
    let mut paths = Vec::new();
    for index in 0..count {
        // SAFETY: dyld accepts any index and returns null if it is out of range.
        let name = unsafe { _dyld_get_image_name(index) };
        if name.is_null() {
            continue;
        }
        // SAFETY: the non-null name is a dyld-owned NUL-terminated string. Copy
        // it immediately; this fixture keeps the host/runtime loaded until
        // enumeration finishes and never calls dlclose.
        let path = PathBuf::from(OsStr::from_bytes(
            unsafe { CStr::from_ptr(name) }.to_bytes(),
        ));
        if path.file_name().is_some_and(|file| {
            ["libggml", "libllama.", "libmtmd"]
                .iter()
                .any(|prefix| file.to_string_lossy().starts_with(prefix))
        }) {
            paths.push(path.canonicalize()?.to_string_lossy().into_owned());
        }
    }
    Ok(paths.join("\n"))
}

/// Reads the process memory map under its platform name.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn process_map() -> Result<String> {
    bail!("This native fixture requires macOS or Linux mapped-image inspection")
}

/// Appends one session event projection to the trace file.
fn record_event(
    file: &mut File,
    start: Instant,
    event: &LiveTranscriptionEvent,
    source: &str,
) -> Result<()> {
    serde_json::to_writer(
        &mut *file,
        &json!({"elapsed_ms": start.elapsed().as_millis(), "source": source, "event": event}),
    )?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

type SpeechReader = Option<JoinHandle<Result<Vec<LiveTranscriptionEvent>>>>;

/// Reads the session events until end of stream.
async fn join_reader(reader: &mut SpeechReader) -> Result<Vec<LiveTranscriptionEvent>> {
    let Some(mut task) = reader.take() else {
        return Ok(Vec::new());
    };
    match bounded(5, "speech reader EOF", &mut task).await {
        Ok(result) => result.context("speech reader task")?,
        Err(error) => {
            *reader = Some(task);
            Err(error)
        }
    }
}

/// Streams the fixture PCM and checks the counted transcript phrases.
async fn transcribe(node: &MeshNode, inputs: &Inputs, model: &str) -> Result<()> {
    let live = bounded(
        35,
        "speech start",
        node.start_live_transcription(options(model, 400)),
    )
    .await??;
    let session = Arc::new(LiveTranscriptionSession::new(live));
    let event_path = inputs.evidence.join("speech.jsonl");
    let inbound = Arc::clone(&session);
    let started = Instant::now();
    let mut reader = Some(tokio::spawn(async move {
        let mut file = OpenOptions::new()
            .append(true)
            .create_new(true)
            .open(event_path)?;
        let mut events = Vec::new();
        while let Some(event) = inbound.next_event().await {
            record_event(&mut file, started, &event, "reader")?;
            events.push(event);
        }
        Ok(events)
    }));
    let outcome = stream_clip(&session, &mut reader, inputs, started).await;
    let closed = bounded(5, "speech close", session.close()).await;
    let joined = join_reader(&mut reader).await;
    if let Some(task) = reader {
        task.abort();
        let _ = task.await;
    }
    let repeated_close = bounded(5, "repeat speech close", session.close()).await;
    outcome?;
    closed?;
    joined?;
    repeated_close
}

/// Streams a clip through the SDK and collects its events.
async fn stream_clip(
    session: &LiveTranscriptionSession,
    reader: &mut SpeechReader,
    inputs: &Inputs,
    started: Instant,
) -> Result<()> {
    for chunk in inputs.pcm.as_chunks::<1600>().0 {
        // Each send waits for the session to accept the chunk. Cadence runs on
        // the audio clock, so unpaced sends must give the same transcript.
        bounded(10, "audio send", session.send_audio(chunk.to_vec())).await??;
    }
    let tail = bounded(20, "speech stop", session.stop()).await??;
    let mut events = join_reader(reader).await?;
    let mut file = OpenOptions::new()
        .append(true)
        .open(inputs.evidence.join("speech.jsonl"))?;
    for event in tail {
        record_event(&mut file, started, &event, "stop")?;
        events.push(event);
    }
    inputs.write(
        "sent-audio.json",
        &json!({"samples": inputs.pcm.len(), "sends": 110,
        "sample_rate": 16000, "step_ms": 400, "tentative": true, "pcm_sha256": PCM_SHA256}),
    )?;
    assert_transcript(inputs, &events)?;
    let closed = bounded(2, "repeat speech stop", session.stop())
        .await?
        .err()
        .context("Repeated stop succeeded")?;
    ensure!(
        closed.to_string().contains("closed"),
        "Repeated stop must report a closed session"
    );
    ensure!(
        bounded(2, "ended speech read", session.next_event())
            .await?
            .is_none(),
        "Stopped reader is not at EOF"
    );
    Ok(())
}

/// Asserts the four counted phrase repetitions in the completed transcript.
fn assert_transcript(inputs: &Inputs, events: &[LiveTranscriptionEvent]) -> Result<()> {
    let completed = events
        .iter()
        .filter_map(|event| match event {
            LiveTranscriptionEvent::Completed { transcript, .. } => Some(transcript),
            _ => None,
        })
        .collect::<Vec<_>>();
    let text = completed.first().map_or("", |text| text.as_str());
    inputs.write("transcript.json", &json!({"transcript": text}))?;
    ensure!(
        completed.len() == 1,
        "Expected exactly one completed utterance"
    );
    let deltas = events
        .iter()
        .filter_map(|event| match event {
            LiveTranscriptionEvent::Delta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    ensure!(deltas == text, "Delta/completed mismatch");
    ensure!(events.iter().any(|event| matches!(event, LiveTranscriptionEvent::Tentative { text, .. } if !text.trim().is_empty())), "Missing tentative text");
    // The clip was sent at the 400 ms cadence `options` requested: about
    // 27 passes read 11 s of audio, versus about 12 at the 1000 ms default.
    // Each pass that changes the unconfirmed tail reports a tentative event,
    // so a healthy 400 ms run cannot stay at the default's reading count.
    let tentative = events
        .iter()
        .filter(|event| matches!(event, LiveTranscriptionEvent::Tentative { .. }))
        .count();
    ensure!(
        tentative >= 15,
        "Expected the 400 ms cadence to out-read the 1000 ms default; saw {tentative} tentative readings"
    );
    inputs.write(
        "cadence.json",
        &json!({"step_ms": 400, "tentative_events": tentative}),
    )?;
    ensure!(
        !events
            .iter()
            .any(|event| matches!(event, LiveTranscriptionEvent::Error { .. })),
        "Native speech emitted an error"
    );
    let complete_at = events
        .iter()
        .position(|event| matches!(event, LiveTranscriptionEvent::Completed { .. }));
    let delta_at = events
        .iter()
        .rposition(|event| matches!(event, LiveTranscriptionEvent::Delta { .. }));
    ensure!(
        complete_at > delta_at,
        "Completed must follow the final delta"
    );
    let words = text
        .to_lowercase()
        .split(|letter: char| !letter.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let phrases = [
        "fellow americans",
        "ask not",
        "can do for you",
        "ask what you can do for your country",
        "what you can do",
        "do for your country",
    ];
    let mut previous = 0;
    for phrase in phrases {
        let expected = phrase.split_whitespace().collect::<Vec<_>>();
        let positions = words
            .windows(expected.len())
            .enumerate()
            .filter_map(|(index, window)| {
                window
                    .iter()
                    .map(String::as_str)
                    .eq(expected.iter().copied())
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        ensure!(
            positions.len() == 1 && positions[0] >= previous,
            "Expected one ordered occurrence of {phrase}"
        );
        previous = positions[0];
    }
    Ok(())
}

/// A no-op waker for observing one poll of a read that must report
/// `Poll::Pending` without registering a real wake-up.
fn noop_raw_waker() -> std::task::RawWaker {
    const VTABLE: std::task::RawWakerVTable = std::task::RawWakerVTable::new(
        |data| std::task::RawWaker::new(data, &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );
    std::task::RawWaker::new(std::ptr::null(), &VTABLE)
}

/// Aborts the reader task and joins it so no read outlives a failed deadline.
async fn settle_reader<T>(reader: &mut JoinHandle<T>) {
    reader.abort();
    let _ = reader.await;
}

/// Cancels a pending read and confirms close wakes the reader.
async fn cancel_blocked_read(node: &MeshNode, inputs: &Inputs, model: &str) -> Result<()> {
    let live = bounded(
        35,
        "empty session start",
        node.start_live_transcription(options(model, 400)),
    )
    .await??;
    let session = Arc::new(LiveTranscriptionSession::new(live));
    let (polled_tx, polled_rx) = tokio::sync::oneshot::channel();
    // No audio was sent, so the read blocks in the session's event channel.
    // It lives in a waiting task: one poll there observes the pending state
    // before the close, and the `read.await` below it re-registers with the
    // task's own waker, so `close()` - not a repoll from this test - wakes
    // the read at EOF.
    let mut reader = {
        let session = Arc::clone(&session);
        tokio::spawn(async move {
            let mut read = Box::pin(session.next_event());
            let pending = {
                let waker = unsafe { std::task::Waker::from_raw(noop_raw_waker()) };
                let mut context = std::task::Context::from_waker(&waker);
                matches!(read.as_mut().poll(&mut context), Poll::Pending)
            };
            let _ = polled_tx.send(pending);
            read.await
        })
    };
    let pending = bounded(2, "reader polled pending", polled_rx).await??;
    // Rust cancels this session by close, not by aborting a recv future. The
    // reader's handle stays owned here across both deadlines: if one expires,
    // the reader is aborted and joined before the error is returned, while
    // the successful path is unchanged.
    match bounded(3, "cancel pending read", session.close()).await {
        Ok(()) => {}
        Err(error) => {
            settle_reader(&mut reader).await;
            return Err(error);
        }
    }
    let event = match tokio::time::timeout(Duration::from_secs(5), &mut reader).await {
        Ok(join) => join.context("cancelled reader EOF")?,
        Err(_elapsed) => {
            settle_reader(&mut reader).await;
            bail!("cancelled reader EOF deadline expired");
        }
    };
    ensure!(
        pending,
        "The native read must be pending before cancellation, not return an early event"
    );
    ensure!(
        event.is_none(),
        "Close must unblock the pending read at EOF"
    );
    inputs.write(
        "cancellation.json",
        &json!({"pending_observed": "Poll::Pending reported by the waiting reader task",
        "cancellation_api": "LiveTranscriptionSession::close",
        "close_woke_reader": true,
        "pending_reader_returned_eof": true}),
    )?;
    match bounded(3, "second cancellation close", session.close()).await {
        Ok(()) => {}
        Err(error) => {
            settle_reader(&mut reader).await;
            return Err(error);
        }
    }
    ensure!(
        bounded(2, "repeat cancelled read", session.next_event())
            .await?
            .is_none(),
        "Cancelled reader must stay at EOF"
    );
    Ok(())
}
