//! Actual CLI and finite HTTP exchanges, without model/native product execution.
#![cfg(unix)]
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
#[derive(Clone)]
struct Request {
    headers: String,
    body: Vec<u8>,
}
struct Server {
    url: String,
    requests: Arc<Mutex<Vec<Request>>>,
    stop: Arc<AtomicBool>,
    worker: Mutex<Option<thread::JoinHandle<Result<(), String>>>>,
}
fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    stream
        .set_nonblocking(false)
        .map_err(|error| error.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .map_err(|error| error.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut raw = Vec::new();
    let mut buffer = [0; 4096];
    loop {
        if Instant::now() >= deadline {
            return Err(format!(
                "request read deadline; received {} bytes; headers: {}",
                raw.len(),
                String::from_utf8_lossy(&raw[..raw.len().min(4096)])
            ));
        }
        let n = match stream.read(&mut buffer) {
            Ok(0) => return Err("request EOF before complete body".into()),
            Ok(n) => n,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::Interrupted
                ) =>
            {
                continue;
            }
            Err(error) => return Err(format!("request read failed: {error}")),
        };
        raw.extend_from_slice(&buffer[..n]);
        if raw.len() > 1024 * 1024 {
            return Err("fixture request exceeds 1 MiB".into());
        }
        let Some(split) = raw.windows(4).position(|b| b == b"\r\n\r\n") else {
            continue;
        };
        let headers = std::str::from_utf8(&raw[..split]).map_err(|error| error.to_string())?;
        let length = headers
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|s| s.trim().parse::<usize>())
            })
            .ok_or_else(|| format!("missing Content-Length in fixture request: {headers}"))?
            .map_err(|error| error.to_string())?;
        if length > 1024 * 1024 - split - 4 {
            return Err("declared fixture body exceeds bound".into());
        }
        if raw.len() >= split + 4 + length {
            return Ok(Request {
                headers: headers.into(),
                body: raw[split + 4..split + 4 + length].to_vec(),
            });
        }
    }
}
impl Server {
    fn new(status: u16, mime: &'static str, reply: Json) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let halted = stop.clone();
        let worker = thread::spawn(move || {
            let end = Instant::now() + Duration::from_secs(12);
            while !halted.load(Ordering::SeqCst) && Instant::now() < end {
                let (mut stream, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => return Err(error.to_string()),
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .map_err(|error| error.to_string())?;
                stream
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .map_err(|error| error.to_string())?;
                let request = read_request(&mut stream)?;
                captured.lock().unwrap().push(request);
                let body = reply.to_string();
                write!(stream,"HTTP/1.1 {status} Fixture\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).map_err(|error| error.to_string())?;
            }
            Ok(())
        });
        Self {
            url,
            requests,
            stop,
            worker: Mutex::new(Some(worker)),
        }
    }
    fn finish(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.lock().unwrap().take() {
            let result = worker.join().expect("HTTP fixture worker panicked");
            assert!(result.is_ok(), "HTTP fixture worker failed: {result:?}");
        }
    }
    fn request(&self) -> Request {
        let requests = self.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        requests[0].clone()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Ok(slot) = self.worker.get_mut()
            && let Some(worker) = slot.take()
        {
            let _ = worker.join();
        }
    }
}
fn invoke(
    root: &Path,
    candidate: &Server,
    oracle: &Server,
    class: &str,
    media: &Path,
    extra: &[&str],
) -> process::ProcessReport {
    let args = [
        "automation",
        "workload-media-oracle",
        "--candidate-url",
        &candidate.url,
        "--oracle-url",
        &oracle.url,
        "--model",
        "model \"quoted\"",
        "--class",
        class,
        "--media-path",
        media.to_str().unwrap(),
    ];
    let spec = ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        arguments: args
            .into_iter()
            .chain(extra.iter().copied())
            .map(|s| Value::Public(s.into()))
            .collect(),
        cwd: root.into(),
        environment: BTreeMap::new(),
    };
    let limits = Limits {
        execution: Duration::from_secs(8),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let output = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(output.cleanup.complete, "{output:?}");
    candidate.finish();
    oracle.finish();
    output
}
fn choice(text: &str) -> Json {
    json!({"choices":[{"message":{"content":text}}]})
}
fn image(root: &Path) -> std::path::PathBuf {
    let path = root.join("known-label.png");
    fs::write(
        &path,
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../ci/fixtures/ocr-mesh-42.png"
        )),
    )
    .unwrap();
    path
}
#[test]
fn ocr_actual_requests_preserve_known_label_and_media() {
    let root = tempfile::tempdir().unwrap();
    let media = image(root.path());
    let candidate = Server::new(200, "application/json", choice("ＭＥＳＨ—42"));
    let oracle = Server::new(200, "application/json", choice("mesh 42"));
    let output = invoke(root.path(), &candidate, &oracle, "ocr", &media, &[]);
    assert!(output.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout.bytes_retained)
            .contains("ocr local-monolithic oracle passed")
    );
    let request = candidate.request();
    assert!(request.headers.starts_with("POST /v1/chat/completions "));
    assert_eq!(request.body, oracle.request().body);
    let payload: Json = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(payload["max_tokens"], 64);
    assert_eq!(payload["temperature"], 0);
    assert_eq!(payload["seed"], 1);
    assert_eq!(payload["model"], "model \"quoted\"");
    let content = &payload["messages"][0]["content"];
    assert_eq!(
        content[0]["text"],
        "Read all visible text. Return only the transcription."
    );
    let url = content[1]["image_url"]["url"].as_str().unwrap();
    assert!(url.starts_with("data:image/png;base64,iVBORw0KGgo"));
}
#[test]
fn asr_actual_multipart_and_aligned_chat_are_parity_only_unlabeled() {
    let root = tempfile::tempdir().unwrap();
    let media = root.path().join("audio.wav");
    let audio = b"RIFF\0fixture\xffWAVE";
    fs::write(&media, audio).unwrap();
    let candidate = Server::new(
        200,
        "application/json",
        json!({"text":"The text is: Straße!"}),
    );
    let oracle = Server::new(200, "application/json", choice("The audio is STRASSE"));
    let output = invoke(
        root.path(),
        &candidate,
        &oracle,
        "speech_recognition",
        &media,
        &[],
    );
    assert!(output.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout.bytes_retained)
            .contains("unlabeled fixture, no accuracy claim")
    );
    let actual = candidate.request();
    assert!(actual.headers.starts_with("POST /v1/audio/transcriptions "));
    assert!(actual.headers.contains("boundary=mesh-llm-ocr-asr-oracle"));
    assert!(actual.body.windows(audio.len()).any(|part| part == audio));
    let fields = String::from_utf8_lossy(&actual.body);
    assert!(fields.contains("name=\"response_format\"\r\n\r\njson\r\n"));
    assert!(fields.contains("name=\"temperature\"\r\n\r\n0\r\n"));
    assert!(fields.contains("filename=\"oracle.wav\""));
    let request = oracle.request();
    assert!(request.headers.starts_with("POST /v1/chat/completions "));
    let payload: Json = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(
        payload["messages"][0]["content"][0]["text"],
        "Transcribe audio to text\n"
    );
    assert_eq!(
        payload["messages"][0]["content"][1]["input_audio"]["format"],
        "wav"
    );
    assert_eq!(payload["max_tokens"], 128);
    assert_eq!(payload["temperature"], 0);
    assert_eq!(
        payload["messages"][0]["content"][1]["input_audio"]["data"],
        "UklGRgBmaXh0dXJl/1dBVkU="
    );
}
#[test]
fn ocr_mismatch_and_matching_wrong_label_fail() {
    let root = tempfile::tempdir().unwrap();
    let media = image(root.path());
    for (actual, reference, diagnostic) in [
        ("MESH 42", "MESH 43", "text differs"),
        ("MESH 43", "MESH 43", "independently known fixture text"),
        (
            "MESH 42 extra",
            "MESH 42 extra",
            "independently known fixture text",
        ),
        (
            "MESH 42 MESH 42",
            "MESH 42 MESH 42",
            "independently known fixture text",
        ),
        ("", "MESH 42", "empty text"),
    ] {
        let candidate = Server::new(200, "application/json", choice(actual));
        let oracle = Server::new(200, "application/json", choice(reference));
        let output = invoke(root.path(), &candidate, &oracle, "ocr", &media, &[]);
        assert!(!output.success());
        assert!(
            String::from_utf8_lossy(&output.stderr.bytes_retained).contains(diagnostic),
            "{output:?}"
        );
        candidate.request();
        oracle.request();
    }
}
#[test]
fn asr_refusal_empty_and_label_mismatch_fail_after_actual_requests() {
    let root = tempfile::tempdir().unwrap();
    let media = root.path().join("audio.wav");
    fs::write(&media, b"RIFF-fixture-WAVE").unwrap();
    for (actual, expected, diagnostic) in [
        ("I cannot fulfill that", None, "refusal"),
        ("The audio is:", None, "empty transcript"),
        ("Ready", Some("Other"), "independently known fixture text"),
    ] {
        let candidate = Server::new(200, "application/json", json!({"text":actual}));
        let oracle = Server::new(200, "application/json", choice(actual));
        let extra = expected.map_or(Vec::new(), |text| vec!["--expected-text", text]);
        let output = invoke(
            root.path(),
            &candidate,
            &oracle,
            "speech_recognition",
            &media,
            &extra,
        );
        assert!(!output.success());
        assert!(
            String::from_utf8_lossy(&output.stderr.bytes_retained).contains(diagnostic),
            "{output:?}"
        );
        candidate.request();
        oracle.request();
    }
}
#[test]
fn collision_empty_media_and_bad_http_fail_closed() {
    let root = tempfile::tempdir().unwrap();
    let media = root.path().join("audio.wav");
    let candidate = Server::new(200, "application/json", json!({"text":"Ready"}));
    let oracle = Server::new(200, "application/json", choice("Ready"));
    for bytes in [b"mesh-llm-ocr-asr-oracle".as_slice(), b""] {
        fs::write(&media, bytes).unwrap();
        let output = invoke(
            root.path(),
            &candidate,
            &oracle,
            "speech_recognition",
            &media,
            &[],
        );
        assert!(!output.success());
        assert!(candidate.requests.lock().unwrap().is_empty());
        assert!(oracle.requests.lock().unwrap().is_empty());
    }
    fs::File::create(&media)
        .unwrap()
        .set_len(64 * 1024 * 1024 + 1)
        .unwrap();
    let output = invoke(
        root.path(),
        &candidate,
        &oracle,
        "speech_recognition",
        &media,
        &[],
    );
    assert!(!output.success());
    assert!(String::from_utf8_lossy(&output.stderr.bytes_retained).contains("64 MiB"));
    assert!(candidate.requests.lock().unwrap().is_empty());
    assert!(oracle.requests.lock().unwrap().is_empty());
    let media = image(root.path());
    for (status, mime, body, diagnostic) in [
        (
            503,
            "application/json",
            choice("MESH 42"),
            "HTTP request failed",
        ),
        (200, "text/plain", choice("MESH 42"), "content type"),
        (200, "application/json", json!([]), "JSON object"),
        (
            200,
            "application/json",
            json!({"choices":[]}),
            "exactly one",
        ),
    ] {
        let candidate = Server::new(status, mime, body);
        let oracle = Server::new(200, "application/json", choice("MESH 42"));
        let output = invoke(root.path(), &candidate, &oracle, "ocr", &media, &[]);
        assert!(!output.success());
        assert!(
            String::from_utf8_lossy(&output.stderr.bytes_retained).contains(diagnostic),
            "{output:?}"
        );
        candidate.request();
    }
}

#[test]
fn unknown_asr_prefix_and_repeated_content_are_not_silently_removed() {
    let root = tempfile::tempdir().unwrap();
    let media = root.path().join("audio.wav");
    fs::write(&media, b"RIFF-fixture-WAVE").unwrap();
    for (actual, reference, success) in [
        ("Transcribed text: ready", "The text is ready", false),
        ("ready ready", "The audio is ready", false),
        // Unknown prefix is ordinary content; matching both sides is valid parity.
        ("Transcribed text: ready", "Transcribed text: ready", true),
    ] {
        let candidate = Server::new(200, "application/json", json!({"text":actual}));
        let oracle = Server::new(200, "application/json", choice(reference));
        let output = invoke(
            root.path(),
            &candidate,
            &oracle,
            "speech_recognition",
            &media,
            &[],
        );
        assert_eq!(output.success(), success, "{output:?}");
        if !success {
            assert!(
                String::from_utf8_lossy(&output.stderr.bytes_retained).contains("text differs")
            );
        }
        candidate.request();
        oracle.request();
    }
}
#[test]
fn reviewed_static_ocr_asset_has_frozen_identity_and_rgb_dimensions() {
    let root = tempfile::tempdir().unwrap();
    let bytes = fs::read(image(root.path())).unwrap();
    assert_eq!(
        hex::encode(Sha256::digest(&bytes)),
        "ff5f6c29aa9b06d1c577561c6cddcfce8b06f40a8c316542ec64fbc2fdd9c009"
    );
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(&bytes[8..16], b"\x00\x00\x00\x0dIHDR");
    assert_eq!(u32::from_be_bytes(bytes[16..20].try_into().unwrap()), 564);
    assert_eq!(u32::from_be_bytes(bytes[20..24].try_into().unwrap()), 156);
    assert_eq!(&bytes[24..29], &[8, 2, 0, 0, 0]);
}

#[test]
fn special_media_is_ordinary_failure_without_http() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt as _};
    let root = tempfile::tempdir().unwrap();
    let fifo = root.path().join("unopened writer fifo");
    let name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // SAFETY: CString is NUL terminated and remains valid for the mkfifo call.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let directory = root.path().join("directory");
    fs::create_dir(&directory).unwrap();
    let original = image(root.path());
    let symlink = root.path().join("media symlink");
    std::os::unix::fs::symlink(original, &symlink).unwrap();
    let candidate = Server::new(200, "application/json", choice("MESH 42"));
    let oracle = Server::new(200, "application/json", choice("MESH 42"));
    for media in [fifo, directory, symlink] {
        let output = invoke(root.path(), &candidate, &oracle, "ocr", &media, &[]);
        assert_eq!(output.outcome, process::Outcome::Exited, "{output:?}");
        assert!(!output.success(), "{output:?}");
        assert!(output.elapsed < Duration::from_secs(4), "{output:?}");
        assert!(String::from_utf8_lossy(&output.stderr.bytes_retained).contains("regular file"));
        assert!(candidate.requests.lock().unwrap().is_empty());
        assert!(oracle.requests.lock().unwrap().is_empty());
    }
}

fn candidate_server_helper() -> String {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/skippy-workload-certify.sh"),
    )
    .unwrap();
    let start = source
        .find("start_candidate_server() {\n")
        .expect("owning startup helper");
    let tail = &source[start..];
    let end = tail
        .find("\nstart_candidate_server\n")
        .expect("complete helper boundary");
    tail[..end].to_owned()
}

fn server_caller(root: &Path, helper: &str) -> process::ProcessReport {
    let script = format!(
        "set -euo pipefail\nwait_for_workload_server() {{ wait \"$1\"; }}\n{helper}\nstart_candidate_server\n"
    );
    let mut environment = BTreeMap::new();
    for (name, value) in [
        ("CANDIDATE_BIN_DIR", root.display().to_string()),
        ("FIXTURE", root.display().to_string()),
        (
            "CONFIG_PATH",
            root.join("admitted config.json").display().to_string(),
        ),
        ("SERVER_LOG", root.join("server log").display().to_string()),
        ("PORT", "43123".into()),
        ("PORT_START_ATTEMPTS", "1".into()),
        ("BACKEND", "cpu".into()),
        ("MODEL_CLASS", "speech_recognition".into()),
    ] {
        environment.insert(name.into(), Value::Public(value.into()));
    }
    let spec = ProcessSpec {
        executable: "/bin/bash".into(),
        arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
        cwd: root.into(),
        environment,
    };
    let limits = Limits {
        execution: Duration::from_secs(8),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let output = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        output.cleanup.complete && !output.stdout.truncated && !output.stderr.truncated,
        "{output:?}"
    );
    output
}

#[test]
fn actual_asr_server_caller_binds_admitted_config_backend_and_128_token_default() {
    use std::os::unix::fs::PermissionsExt as _;
    let root = tempfile::Builder::new()
        .prefix("asr caller space ")
        .tempdir()
        .unwrap();
    let executable = root.path().join("skippy-server");
    fs::write(&executable, "#!/bin/sh\nset -eu\nprintf '%s\\n' \"$@\" > \"$FIXTURE/argv\"\nprintf '%s\\n' \"$LLAMA_STAGE_BACKEND\" > \"$FIXTURE/backend\"\nprintf 'called\\n' >> \"$FIXTURE/calls\"\n").unwrap();
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
    let config = root.path().join("admitted config.json");
    fs::write(&config, b"immutable admitted fixture config").unwrap();
    let output = server_caller(root.path(), &candidate_server_helper());
    assert!(output.success(), "{output:?}");
    assert_eq!(
        fs::read_to_string(root.path().join("argv")).unwrap(),
        format!(
            "serve-openai\n--config\n{}\n--bind-addr\n127.0.0.1:43123\n--default-max-tokens\n128\n--telemetry-level\noff\n",
            config.display()
        )
    );
    assert_eq!(
        fs::read_to_string(root.path().join("backend")).unwrap(),
        "cpu\n"
    );
    assert_eq!(
        fs::read_to_string(root.path().join("calls")).unwrap(),
        "called\n"
    );
    assert_eq!(
        fs::read(&config).unwrap(),
        b"immutable admitted fixture config"
    );
    assert!(root.path().join("server log").is_file());
}
