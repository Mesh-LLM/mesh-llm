//! A node with the default settings sends nothing to any peer on its own,
//! idle or serving: the compiled `capsules` binary, driven over the real
//! plugin wire protocol by a fake host, is handed a completed exchange with a
//! known counterparty and must not open a mesh stream, send a channel message
//! or a bulk transfer, or ask for a peer block. The control: the same run with
//! `share_record_at_completion = counterparty` does open a mesh stream to the
//! counterparty, so the quiet result is not just a test that cannot see a
//! push.
//!
//! Besides a 5-second look while idle, no fixed window decides a result. The plugin handles channel
//! messages strictly one at a time, in order, so once it has recorded a
//! second exchange, its handling of the first (any push included) has
//! finished: that is when the quiet test looks. The control waits for the
//! push itself. A push first cuts a checkpoint covering the record, padded to
//! its bucket, which a slow machine running a debug build can take many
//! seconds to do. Every wait is bounded ([`SETTLE`], [`STARTUP`]) and fails
//! loudly.

use mesh_llm_plugin::proto::{self, envelope::Payload};
use mesh_llm_plugin::{LocalListener, PROTOCOL_VERSION};
use prost::Message;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::time::timeout;

/// The two halves of the plugin's socket, as `LocalStream::into_split` gives
/// them.
type LocalReadHalf = Box<dyn AsyncRead + Send + Unpin>;
type LocalWriteHalf = Box<dyn AsyncWrite + Send + Unpin>;

const PLUGIN_BIN: &str = env!("CARGO_BIN_EXE_capsules");
const SELF_PEER: &str = "aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11";
const SERVING_PEER: &str = "bb22bb22bb22bb22bb22bb22bb22bb22bb22bb22bb22bb22bb22bb22bb22bb22";
/// The most a test waits for the plugin to finish with an exchange.
const SETTLE: Duration = Duration::from_secs(180);
/// The most a test waits for the plugin to connect and answer initialize.
const STARTUP: Duration = Duration::from_secs(60);

/// Unique within the process too: the tests run in parallel, and two clock
/// reads can return the same nanosecond.
fn nonce() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

struct Host {
    child: tokio::process::Child,
    /// The write half of the plugin's socket.
    writer: LocalWriteHalf,
    /// Every envelope the plugin sends, in order, from [`read_frames`].
    inbox: mpsc::UnboundedReceiver<proto::Envelope>,
    next: u64,
    data_dir: std::path::PathBuf,
    /// Everything peer-bound the plugin has sent so far.
    sent: Vec<String>,
}

impl Host {
    async fn spawn(extra_env: &[(&str, &str)]) -> Self {
        let socket = std::env::temp_dir().join(format!("capsules-quiet-{}.sock", nonce()));
        let _ = std::fs::remove_file(&socket);
        let listener = UnixListener::bind(&socket).unwrap();
        let data_dir = std::env::temp_dir().join(format!("capsules-quiet-data-{}", nonce()));
        let mut cmd = Command::new(PLUGIN_BIN);
        cmd.env("MESH_LLM_PLUGIN_ENDPOINT", &socket)
            .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
            .env("CAPSULES_DATA_DIR", &data_dir)
            .env("CAPSULES_SELF_PEER_ID", SELF_PEER)
            // Never the developer's own mesh config or settings.
            .env("MESH_LLM_CONFIG", "/nonexistent/config.toml")
            .env_remove("CAPSULES_SHARE_RECORD_AT_COMPLETION")
            .env_remove("CAPSULES_SHARE_ADJUDICATIONS")
            .env_remove("CAPSULES_ADJUDICATE_DIFFERING_TWINS")
            .env_remove("CAPSULES_CHECKPOINT_WITNESS_URLS");
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
        let child = cmd.spawn().unwrap();
        let stream = timeout(STARTUP, LocalListener::Unix(listener, socket).accept())
            .await
            .unwrap_or_else(|_| panic!("the plugin did not connect within {STARTUP:?}"))
            .unwrap();
        let (reader, writer) = stream.into_split();
        let (frames, inbox) = mpsc::unbounded_channel();
        tokio::spawn(read_frames(reader, frames));
        let mut host = Self {
            child,
            writer,
            inbox,
            next: 1,
            data_dir,
            sent: Vec::new(),
        };
        host.send(Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "quiet-default-test".into(),
            host_info_json: "{}".into(),
            mesh_visibility: proto::MeshVisibility::Private as i32,
            ..Default::default()
        }))
        .await;
        // The plugin registers its handlers before it connects, so its
        // initialize reply is the readiness signal: an exchange sent after it
        // is queued and handled, never dropped.
        let reply = host
            .recv(STARTUP)
            .await
            .unwrap_or_else(|| panic!("no initialize reply within {STARTUP:?}"));
        assert!(matches!(
            reply.payload,
            Some(Payload::InitializeResponse(_))
        ));
        host
    }

    async fn send(&mut self, payload: Payload) {
        let request_id = self.next;
        self.next += 1;
        let body = proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "capsules".into(),
            request_id,
            payload: Some(payload),
        }
        .encode_to_vec();
        self.writer
            .write_all(&(body.len() as u32).to_le_bytes())
            .await
            .unwrap();
        self.writer.write_all(&body).await.unwrap();
    }

    /// The plugin's next envelope, if one arrives within `wait`. Waiting on
    /// the channel is safe to cut short; the socket itself is read only by
    /// [`read_frames`], which is never cut short.
    async fn recv(&mut self, wait: Duration) -> Option<proto::Envelope> {
        timeout(wait, self.inbox.recv()).await.ok()?
    }

    /// Read what the plugin sends for up to `window`, keeping everything
    /// that would reach a peer (or the routing plane) in `sent`. Rpc replies
    /// to the host are not kept. Returns early once `done` holds.
    async fn watch(&mut self, window: Duration, done: impl Fn(&Self) -> bool) {
        let deadline = tokio::time::Instant::now() + window;
        while !done(self) {
            let Some(left) = deadline.checked_duration_since(tokio::time::Instant::now()) else {
                return;
            };
            // Short reads, so `done` is checked often even when nothing comes.
            let Some(envelope) = self.recv(left.min(Duration::from_millis(250))).await else {
                continue;
            };
            match envelope.payload {
                Some(Payload::OpenMeshStreamRequest(r)) => {
                    self.sent.push(format!("open_mesh_stream {r:?}"))
                }
                Some(Payload::ChannelMessage(m)) => {
                    self.sent.push(format!("channel_message {}", m.channel))
                }
                Some(Payload::BulkTransferMessage(_)) => self.sent.push("bulk_transfer".into()),
                Some(Payload::PeerBlockRequest(r)) => {
                    self.sent.push(format!("peer_block {}", r.peer_id))
                }
                _ => {}
            }
        }
    }

    /// Whether the plugin has recorded the exchange `exchange_id` in its
    /// lifecycle log, which it does while handling that exchange.
    fn has_recorded(&self, exchange_id: &str) -> bool {
        std::fs::read_to_string(self.data_dir.join("lifecycle-events.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .any(|event| event["exchange_id"] == exchange_id)
    }

    /// Send a completed exchange, then a second one, and wait until the
    /// plugin has recorded the second: channel messages are handled one at
    /// a time, in order, so by then it has finished with the first, push or
    /// no push. Fails if that does not happen within [`SETTLE`].
    async fn exchange_and_settle(&mut self) {
        self.completed_exchange().await;
        let barrier = self.completed_exchange().await;
        let started = std::time::Instant::now();
        self.watch(SETTLE, |host| host.has_recorded(&barrier)).await;
        assert!(
            self.has_recorded(&barrier),
            "the plugin did not finish with the exchange within {SETTLE:?}"
        );
        eprintln!("exchange settled after {:?}", started.elapsed());
    }

    /// A completed exchange this node asked for and `SERVING_PEER` served;
    /// returns its exchange id.
    async fn completed_exchange(&mut self) -> String {
        let exchange_id = format!("quiet-{}", nonce());
        let event = serde_json::json!({
            "dispatch_path": "remote_mesh",
            "phase": "terminal",
            "model": "quiet-test-model",
            "status": 200,
            "exchange_id": exchange_id,
            "capsule_id": null,
            "nonce": null,
            "request_digest": "11".repeat(32),
            "response_digest": "22".repeat(32),
            "serving_provenance": {"served_by_node_id": SERVING_PEER},
        });
        self.send(Payload::ChannelMessage(proto::ChannelMessage {
            channel: "openai.exchange.v1".into(),
            source_peer_id: String::new(),
            target_peer_id: "capsules".into(),
            content_type: "application/json".into(),
            body: serde_json::to_vec(&event).unwrap(),
            ..Default::default()
        }))
        .await;
        exchange_id
    }

    async fn stop(mut self) {
        let _ = self.child.kill().await;
    }
}

#[tokio::test]
async fn a_default_node_sends_nothing_to_peers_idle_or_serving() {
    let mut host = Host::spawn(&[]).await;
    host.watch(Duration::from_secs(5), |_| false).await;
    assert!(host.sent.is_empty(), "idle: {:?}", host.sent);
    host.exchange_and_settle().await;
    assert!(
        host.sent.is_empty(),
        "serving, default settings: {:?}",
        host.sent
    );
    host.stop().await;
}

/// The control: with the push turned on, the same exchange does reach the
/// counterparty, so the test above can see a push when one happens.
#[tokio::test]
async fn with_the_push_turned_on_the_counterparty_is_contacted() {
    let mut host = Host::spawn(&[("CAPSULES_SHARE_RECORD_AT_COMPLETION", "counterparty")]).await;
    host.completed_exchange().await;
    let pushed = |host: &Host| {
        host.sent
            .iter()
            .any(|s| s.starts_with("open_mesh_stream") && s.contains(SERVING_PEER))
    };
    let started = std::time::Instant::now();
    host.watch(SETTLE, pushed).await;
    assert!(
        pushed(&host),
        "expected a push to the counterparty within {SETTLE:?}, saw {:?}",
        host.sent
    );
    eprintln!("pushed after {:?}", started.elapsed());
    host.stop().await;
}

/// Reads the plugin's envelopes (a little-endian u32 length, then the body)
/// and passes each on, until the socket closes or a frame does not decode.
///
/// This task owns the read half and is never cancelled. Reading a frame is
/// two `read_exact` calls, and neither may be abandoned partway: a timeout
/// around them would drop the bytes already read, and every later read would
/// start mid-frame. The waits with timeouts are on the channel instead.
async fn read_frames(mut reader: LocalReadHalf, frames: mpsc::UnboundedSender<proto::Envelope>) {
    loop {
        let mut len = [0u8; 4];
        if reader.read_exact(&mut len).await.is_err() {
            return;
        }
        let mut body = vec![0u8; u32::from_le_bytes(len) as usize];
        if reader.read_exact(&mut body).await.is_err() {
            return;
        }
        let Ok(envelope) = proto::Envelope::decode(body.as_slice()) else {
            return;
        };
        if frames.send(envelope).is_err() {
            return;
        }
    }
}

/// Each request a [`recording_witness`] saw: its request line and `Host`.
type Seen = std::sync::Arc<std::sync::Mutex<Vec<(String, String)>>>;

/// A local witness that answers every request with 503 and keeps the request
/// line and `Host` header of each, so a test can see exactly where the plugin
/// went.
fn recording_witness() -> (String, Seen) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let keep = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut buf = vec![0u8; 16 * 1024];
            let n = stream.read(&mut buf).unwrap_or(0);
            let text = String::from_utf8_lossy(&buf[..n]).to_string();
            let line = text.lines().next().unwrap_or_default().to_string();
            let host = text
                .lines()
                .find_map(|l| {
                    l.split_once(':')
                        .filter(|(k, _)| k.eq_ignore_ascii_case("host"))
                        .map(|(_, v)| v.trim().to_string())
                })
                .unwrap_or_default();
            keep.lock().unwrap().push((line, host));
            let _ = stream.write_all(
                b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
            );
        }
    });
    (addr, seen)
}

/// A witness URL is dialled exactly as the operator gave it: the same host and
/// port, under the same path, for the plugin's own key fetch and for the
/// checkpoint registration alike. No URL is mapped to another service.
#[tokio::test]
async fn a_configured_witness_is_dialled_exactly_as_given() {
    let (addr, seen) = recording_witness();
    let url = format!("http://{addr}/given/prefix");
    let mut host = Host::spawn(&[
        ("CAPSULES_CHECKPOINT_WITNESS_URLS", url.as_str()),
        ("CAPSULES_CHECKPOINT_CADENCE_SECONDS", "1"),
        ("CAPSULES_CHECKPOINT_CADENCE_ENTRIES", "1"),
    ])
    .await;
    host.completed_exchange().await;
    let both = |seen: &Seen| {
        let requests = seen.lock().unwrap();
        requests
            .iter()
            .any(|(line, _)| line.starts_with("GET /given/prefix/anchor/authority-pubkey "))
            && requests
                .iter()
                .any(|(line, _)| line.starts_with("POST /given/prefix/"))
    };
    // Keeps the host side drained while the plugin works; returns as soon as
    // the witness has seen both requests.
    host.watch(SETTLE, |_| both(&seen)).await;
    assert!(
        both(&seen),
        "expected the key fetch and a checkpoint registration within {SETTLE:?}, saw {:?}",
        seen.lock().unwrap()
    );
    for (line, host_header) in seen.lock().unwrap().iter() {
        assert_eq!(host_header, &addr, "{line}");
        let path = line.split(' ').nth(1).unwrap_or_default();
        assert!(path.starts_with("/given/prefix/"), "{line}");
    }
    host.stop().await;
}
