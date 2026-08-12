use std::collections::VecDeque;
use std::sync::Mutex;

use mesh_llm_log_store::{Clock as StoreClock, WebhookDeliveryState};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, watch};

use super::*;
use crate::logging::operator_audit::OperatorAuditWriter;

const NOW: &str = "2026-08-04T12:00:00Z";

#[derive(Clone)]
struct FixedClock;

impl WebhookWorkerClock for FixedClock {
    fn now(&self) -> String {
        NOW.to_string()
    }
}

impl StoreClock for FixedClock {
    fn now(&self) -> String {
        NOW.to_string()
    }
}

struct FixedJitter(u64);

impl WebhookJitter for FixedJitter {
    fn millis_up_to(&self, inclusive_maximum: u64) -> u64 {
        self.0.min(inclusive_maximum)
    }
}

#[derive(Clone)]
struct AdjustableClock {
    value: Arc<Mutex<String>>,
}

impl AdjustableClock {
    fn new(value: &str) -> Self {
        Self {
            value: Arc::new(Mutex::new(value.to_owned())),
        }
    }

    fn set(&self, value: &str) {
        *self.value.lock().expect("clock lock") = value.to_owned();
    }
}

impl WebhookWorkerClock for AdjustableClock {
    fn now(&self) -> String {
        self.value.lock().expect("clock lock").clone()
    }
}

impl StoreClock for AdjustableClock {
    fn now(&self) -> String {
        WebhookWorkerClock::now(self)
    }
}

#[derive(Clone, Copy)]
enum LocalHttpReply {
    Status(u16),
    Stall,
}

struct LocalFakeHttpServer {
    endpoint: String,
    requests: Arc<Mutex<Vec<String>>>,
    received: Arc<Notify>,
    shutdown_tx: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl LocalFakeHttpServer {
    async fn start(replies: impl IntoIterator<Item = LocalHttpReply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind local fake webhook server");
        let endpoint = format!(
            "http://{}/webhook",
            listener.local_addr().expect("local server address")
        );
        let requests = Arc::new(Mutex::new(Vec::new()));
        let received = Arc::new(Notify::new());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let task = tokio::spawn(run_local_fake_http_server(
            listener,
            replies.into_iter().collect(),
            Arc::clone(&requests),
            Arc::clone(&received),
            shutdown_rx,
        ));
        Self {
            endpoint,
            requests,
            received,
            shutdown_tx,
            task,
        }
    }

    async fn wait_for_requests(&self, expected: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if self.requests.lock().expect("request lock").len() >= expected {
                    return;
                }
                self.received.notified().await;
            }
        })
        .await
        .expect("fake server received expected requests");
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().expect("request lock").clone()
    }

    async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        self.task.abort();
        let _ = self.task.await;
    }
}

async fn run_local_fake_http_server(
    listener: TcpListener,
    mut replies: VecDeque<LocalHttpReply>,
    requests: Arc<Mutex<Vec<String>>>,
    received: Arc<Notify>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    loop {
        let accepted = tokio::select! {
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    return;
                }
                continue;
            }
            accepted = listener.accept() => accepted,
        };
        let (mut stream, _) = accepted.expect("accept fake webhook request");
        let request = read_fake_http_request(&mut stream).await;
        requests.lock().expect("request lock").push(request);
        received.notify_one();
        match replies.pop_front().unwrap_or(LocalHttpReply::Status(500)) {
            LocalHttpReply::Status(status) => write_fake_http_response(&mut stream, status).await,
            LocalHttpReply::Stall => {
                let _ = shutdown_rx.changed().await;
                return;
            }
        }
    }
}

async fn read_fake_http_request(stream: &mut TcpStream) -> String {
    const MAX_REQUEST_BYTES: usize = 16 * 1024;
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 1024];
    loop {
        let read = stream.read(&mut chunk).await.expect("read webhook request");
        assert!(read > 0, "webhook client closed before sending a request");
        bytes.extend_from_slice(&chunk[..read]);
        assert!(
            bytes.len() <= MAX_REQUEST_BYTES,
            "fake request exceeded bound"
        );
        let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let headers = std::str::from_utf8(&bytes[..header_end]).expect("request headers utf-8");
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.strip_prefix("content-length: ")
                    .or_else(|| line.strip_prefix("Content-Length: "))
            })
            .and_then(|value| value.parse::<usize>().ok())
            .expect("webhook request content length");
        if bytes.len() >= header_end + 4 + content_length {
            return String::from_utf8(bytes).expect("request utf-8");
        }
    }
}

async fn write_fake_http_response(stream: &mut TcpStream, status: u16) {
    stream
        .write_all(
            format!("HTTP/1.1 {status} Test\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        )
        .await
        .expect("write fake webhook response");
}

#[derive(Clone, Copy)]
enum FakeReply {
    Status(u16),
    Timeout,
}

struct FakeTransport {
    replies: Mutex<VecDeque<FakeReply>>,
    payloads: Mutex<Vec<WebhookTerminalPayload>>,
}

impl FakeTransport {
    fn new(replies: impl IntoIterator<Item = FakeReply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            payloads: Mutex::new(Vec::new()),
        }
    }

    fn payloads(&self) -> Vec<WebhookTerminalPayload> {
        self.payloads.lock().expect("payload lock").clone()
    }
}

#[async_trait]
impl WebhookTransport for FakeTransport {
    async fn post_terminal(
        &self,
        _endpoint: &Url,
        payload: &WebhookTerminalPayload,
        _timeout: Duration,
    ) -> Result<u16, WebhookTransportError> {
        self.payloads
            .lock()
            .expect("payload lock")
            .push(payload.clone());
        match self.replies.lock().expect("reply lock").pop_front() {
            Some(FakeReply::Status(status)) => Ok(status),
            Some(FakeReply::Timeout) => Err(WebhookTransportError::Timeout),
            None => Err(WebhookTransportError::Transport),
        }
    }
}

struct GatedTransport {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl WebhookTransport for GatedTransport {
    async fn post_terminal(
        &self,
        _endpoint: &Url,
        _payload: &WebhookTerminalPayload,
        _timeout: Duration,
    ) -> Result<u16, WebhookTransportError> {
        self.started.notify_one();
        self.release.notified().await;
        Ok(204)
    }
}

fn open_store() -> (Arc<LogStore>, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("test root");
    let store = LogStore::open(root.path(), Arc::new(FixedClock)).expect("open store");
    (Arc::new(store), root)
}

fn seed_terminal_delivery(store: &LogStore, delivery_id: &str, max_attempts: u32) {
    let request_id = format!("request-{delivery_id}");
    store
        .insert_summary(&request_id, None, None, None, None, NOW, None, None, None)
        .expect("summary");
    store
        .write_terminal_event(
            &request_id,
            &format!("event-{delivery_id}"),
            r#"{"type":"completed","prompt":"prompt-secret","path":"/private/secret","artifact_url":"https://artifact.invalid/value"}"#,
            "completed",
            None,
            NOW,
        )
        .expect("terminal event");
    store
        .enqueue_webhook_delivery(delivery_id, &request_id, NOW, max_attempts)
        .expect("delivery");
}

#[test]
fn terminal_payload_preserves_distinct_bounded_outcomes() {
    let (store, _root) = open_store();
    for (delivery_id, terminal_outcome) in [
        ("outcome-completed", "completed"),
        ("outcome-failed", "failed"),
    ] {
        let request_id = format!("request-{delivery_id}");
        store
            .insert_summary(&request_id, None, None, None, None, NOW, None, None, None)
            .expect("summary");
        store
            .write_terminal_event(
                &request_id,
                &format!("event-{delivery_id}"),
                &format!(r#"{{"type":"{terminal_outcome}"}}"#),
                terminal_outcome,
                None,
                NOW,
            )
            .expect("terminal event");
        store
            .enqueue_webhook_delivery(delivery_id, &request_id, NOW, 2)
            .expect("delivery");
    }

    let completed = WebhookTerminalPayload::from_record(
        &store
            .webhook_delivery("outcome-completed")
            .expect("completed delivery query")
            .expect("completed delivery"),
    )
    .expect("completed payload");
    let failed = WebhookTerminalPayload::from_record(
        &store
            .webhook_delivery("outcome-failed")
            .expect("failed delivery query")
            .expect("failed delivery"),
    )
    .expect("failed payload");

    assert_eq!(completed.outcome, WebhookTerminalOutcome::Completed);
    assert_eq!(failed.outcome, WebhookTerminalOutcome::Failed);
    assert_ne!(completed.outcome, failed.outcome);
    assert_eq!(
        serde_json::to_value(&completed).expect("completed payload JSON"),
        serde_json::json!({
            "request_id": "request-outcome-completed",
            "outcome": "completed",
        })
    );
    assert_eq!(
        serde_json::to_value(&failed).expect("failed payload JSON"),
        serde_json::json!({
            "request_id": "request-outcome-failed",
            "outcome": "failed",
        })
    );
}

#[tokio::test]
async fn terminal_request_status_is_immutable_webhook_intent_not_receiver_status() {
    let (store, _root) = open_store();
    let request_id = "request-terminal-status";
    let delivery_id = "delivery-terminal-status";
    store
        .insert_summary(request_id, None, None, None, None, NOW, None, None, None)
        .expect("summary");
    store
        .write_terminal_event_with_webhook(
            request_id,
            "event-terminal-status",
            r#"{"type":"completed","status_code":201}"#,
            "completed",
            Some(201),
            NOW,
            delivery_id,
            2,
        )
        .expect("terminal event and webhook intent");
    let transport = Arc::new(FakeTransport::new([FakeReply::Status(204)]));

    let outcome = worker(Arc::clone(&store), transport.clone())
        .process_next()
        .await
        .expect("worker step");

    assert_eq!(
        outcome,
        WebhookWorkerOutcome::Delivered {
            delivery_id: delivery_id.to_owned(),
            status_code: 204,
        }
    );
    let sent = transport.payloads();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        serde_json::to_value(&sent[0]).expect("sent webhook JSON"),
        serde_json::json!({
            "request_id": request_id,
            "outcome": "completed",
            "status_code": 201,
        })
    );

    let delivered = store
        .webhook_delivery(delivery_id)
        .expect("delivery query")
        .expect("delivery record");
    assert_eq!(delivered.terminal_status_code, Some(201));
    assert_eq!(delivered.response_status_code, Some(204));
    assert_eq!(
        serde_json::to_value(
            WebhookTerminalPayload::from_record(&delivered).expect("restart payload")
        )
        .expect("restart webhook JSON"),
        serde_json::json!({
            "request_id": request_id,
            "outcome": "completed",
            "status_code": 201,
        })
    );
}

fn worker(store: Arc<LogStore>, transport: Arc<dyn WebhookTransport>) -> WebhookDeliveryWorker {
    let config = LoggingWebhookConfig {
        enabled: true,
        url: Some("http://127.0.0.1:9444/webhook".to_string()),
        max_attempts: 3,
        timeout_secs: 1,
        dead_letter_retention_secs: 3_600,
    };
    WebhookDeliveryWorker::from_config(
        store,
        &config,
        transport,
        Arc::new(FixedClock),
        Arc::new(FixedJitter(250)),
    )
    .expect("worker config")
}

fn real_http_worker(
    store: Arc<LogStore>,
    endpoint: String,
    clock: AdjustableClock,
) -> WebhookDeliveryWorker {
    let transport: Arc<dyn WebhookTransport> =
        Arc::new(ReqwestWebhookTransport::new().expect("real webhook transport"));
    WebhookDeliveryWorker::from_config(
        store,
        &LoggingWebhookConfig {
            enabled: true,
            url: Some(endpoint),
            max_attempts: 3,
            timeout_secs: 1,
            dead_letter_retention_secs: 3_600,
        },
        transport,
        Arc::new(clock),
        Arc::new(FixedJitter(0)),
    )
    .expect("real worker config")
}

fn open_adjustable_store(clock: &AdjustableClock) -> (Arc<LogStore>, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("test root");
    let store = LogStore::open(root.path(), Arc::new(clock.clone())).expect("open store");
    (Arc::new(store), root)
}

fn seed_terminal_delivery_with_private_event(
    store: &LogStore,
    delivery_id: &str,
    occurred_at: &str,
    max_attempts: u32,
) -> String {
    let request_id = format!("request-{delivery_id}");
    store
        .insert_summary(
            &request_id,
            Some("safe-model"),
            Some("management"),
            None,
            None,
            occurred_at,
            None,
            None,
            None,
        )
        .expect("summary");
    store
        .write_terminal_event(
            &request_id,
            &format!("event-{delivery_id}"),
            r#"{"type":"completed","prompt":"prompt-secret","completion":"completion-secret","artifact":"/private/secret","credential":"webhook-secret"}"#,
            "completed",
            None,
            occurred_at,
        )
        .expect("terminal event");
    store
        .enqueue_webhook_delivery(delivery_id, &request_id, occurred_at, max_attempts)
        .expect("delivery");
    request_id
}

fn assert_private_delivery_storage(store: &LogStore, delivery_id: &str) {
    let (target_url, response_body, error_msg): (String, Option<String>, Option<String>) = store
        .conn()
        .query_row(
            "SELECT target_url, response_body, error_msg FROM webhook_deliveries WHERE delivery_id = ?1",
            [delivery_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("private delivery storage");
    for value in [target_url]
        .into_iter()
        .chain(response_body)
        .chain(error_msg)
    {
        for secret in [
            "prompt-secret",
            "completion-secret",
            "/private/secret",
            "webhook-secret",
            "127.0.0.1",
        ] {
            assert!(!value.contains(secret), "delivery storage leaked {secret}");
        }
    }
}

#[tokio::test]
async fn real_http_worker_delivers_a_private_terminal_payload() {
    let clock = AdjustableClock::new(NOW);
    let (store, _root) = open_adjustable_store(&clock);
    let server = LocalFakeHttpServer::start([LocalHttpReply::Status(204)]).await;
    let delivery_id = "real-success";
    let request_id = seed_terminal_delivery_with_private_event(&store, delivery_id, NOW, 2);

    let outcome = real_http_worker(Arc::clone(&store), server.endpoint.clone(), clock)
        .process_next()
        .await
        .expect("successful local HTTP delivery");

    assert_eq!(
        outcome,
        WebhookWorkerOutcome::Delivered {
            delivery_id: delivery_id.to_owned(),
            status_code: 204,
        }
    );
    assert_eq!(
        store.webhook_delivery(delivery_id).unwrap().unwrap().state,
        WebhookDeliveryState::Succeeded
    );
    server.wait_for_requests(1).await;
    let request = server.requests().pop().expect("captured webhook request");
    assert!(request.starts_with("POST /webhook HTTP/1.1\r\n"));
    let body = request.split("\r\n\r\n").nth(1).expect("request body");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(body).expect("terminal payload JSON"),
        serde_json::json!({ "request_id": request_id, "outcome": "completed" })
    );
    for secret in [
        "prompt-secret",
        "completion-secret",
        "/private/secret",
        "webhook-secret",
        "credential",
    ] {
        assert!(!request.contains(secret), "webhook request leaked {secret}");
    }
    assert_private_delivery_storage(&store, delivery_id);
    server.shutdown().await;
}

#[tokio::test]
async fn real_http_worker_retries_5xx_and_dead_letters_at_the_attempt_bound() {
    let clock = AdjustableClock::new(NOW);
    let (store, _root) = open_adjustable_store(&clock);
    let server = LocalFakeHttpServer::start([
        LocalHttpReply::Status(503),
        LocalHttpReply::Status(204),
        LocalHttpReply::Status(502),
    ])
    .await;
    let retry_id = "real-retry";
    seed_terminal_delivery_with_private_event(&store, retry_id, NOW, 2);
    let worker = real_http_worker(Arc::clone(&store), server.endpoint.clone(), clock.clone());

    assert_eq!(
        worker.process_next().await.expect("5xx worker step"),
        WebhookWorkerOutcome::RetryScheduled {
            delivery_id: retry_id.to_owned(),
        }
    );
    let retry = store.webhook_delivery(retry_id).unwrap().unwrap();
    assert_eq!(retry.state, WebhookDeliveryState::Retry);
    assert_eq!(
        retry.last_error_code,
        Some(WebhookDeliveryErrorCode::Http5xx)
    );
    assert_eq!(
        retry.next_attempt_at.as_deref(),
        Some("2026-08-04T12:00:01.000000000Z")
    );

    clock.set("2026-08-04T12:00:01Z");
    assert_eq!(
        worker.process_next().await.expect("retry worker step"),
        WebhookWorkerOutcome::Delivered {
            delivery_id: retry_id.to_owned(),
            status_code: 204,
        }
    );
    assert_eq!(
        store.webhook_delivery(retry_id).unwrap().unwrap().state,
        WebhookDeliveryState::Succeeded
    );

    let dead_letter_id = "real-dead-letter";
    seed_terminal_delivery_with_private_event(&store, dead_letter_id, "2026-08-04T12:00:01Z", 1);
    assert_eq!(
        worker
            .process_next()
            .await
            .expect("dead letter worker step"),
        WebhookWorkerOutcome::DeadLettered {
            delivery_id: dead_letter_id.to_owned(),
        }
    );
    let dead_letter = store.webhook_delivery(dead_letter_id).unwrap().unwrap();
    assert_eq!(dead_letter.state, WebhookDeliveryState::DeadLetter);
    assert_eq!(dead_letter.attempt_number, 1);
    assert_eq!(
        dead_letter.last_error_code,
        Some(WebhookDeliveryErrorCode::Http5xx)
    );
    server.wait_for_requests(3).await;
    assert_private_delivery_storage(&store, retry_id);
    assert_private_delivery_storage(&store, dead_letter_id);
    server.shutdown().await;
}

#[tokio::test]
async fn real_http_timeout_keeps_terminal_persistence_off_the_delivery_path() {
    let clock = AdjustableClock::new(NOW);
    let (store, _root) = open_adjustable_store(&clock);
    let server = LocalFakeHttpServer::start([LocalHttpReply::Stall]).await;
    seed_terminal_delivery_with_private_event(&store, "real-timeout", NOW, 2);
    let worker = real_http_worker(Arc::clone(&store), server.endpoint.clone(), clock.clone());
    let worker_task = tokio::spawn(async move { worker.process_next().await });

    server.wait_for_requests(1).await;
    tokio::time::timeout(Duration::from_millis(250), {
        let store = Arc::clone(&store);
        tokio::task::spawn_blocking(move || {
            seed_terminal_delivery_with_private_event(&store, "terminal-while-http-stalls", NOW, 1)
        })
    })
    .await
    .expect("terminal persistence is not delayed by HTTP")
    .expect("terminal persistence task");

    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), worker_task)
            .await
            .expect("HTTP timeout is bounded")
            .expect("worker task")
            .expect("timeout worker result"),
        WebhookWorkerOutcome::RetryScheduled {
            delivery_id: "real-timeout".to_owned(),
        }
    );
    let timeout_record = store.webhook_delivery("real-timeout").unwrap().unwrap();
    assert_eq!(timeout_record.state, WebhookDeliveryState::Retry);
    assert_eq!(
        timeout_record.last_error_code,
        Some(WebhookDeliveryErrorCode::Timeout)
    );
    assert!(
        store
            .webhook_delivery("terminal-while-http-stalls")
            .unwrap()
            .is_some(),
        "terminal persistence remains durable while HTTP is pending"
    );
    assert_private_delivery_storage(&store, "real-timeout");
    server.shutdown().await;
}

#[tokio::test]
async fn real_http_worker_resumes_after_restart_and_completes_audited_manual_retry() {
    let clock = AdjustableClock::new(NOW);
    let root = tempfile::tempdir().expect("restart test root");
    let first_store =
        Arc::new(LogStore::open(root.path(), Arc::new(clock.clone())).expect("initial store"));
    seed_terminal_delivery_with_private_event(&first_store, "real-restart", NOW, 2);
    first_store
        .claim_next_webhook_delivery("2026-08-04T12:00:00Z", "2026-08-04T12:00:01Z")
        .expect("claim before restart")
        .expect("initial claim");
    drop(first_store);
    clock.set("2026-08-04T12:01:00Z");

    let store = Arc::new(
        LogStore::reopen_at(root.path(), Arc::new(clock.clone())).expect("reopened store"),
    );
    let server = LocalFakeHttpServer::start([
        LocalHttpReply::Status(204),
        LocalHttpReply::Status(503),
        LocalHttpReply::Status(204),
    ])
    .await;
    let worker = real_http_worker(Arc::clone(&store), server.endpoint.clone(), clock.clone());
    assert_eq!(
        worker.process_next().await.expect("restart worker step"),
        WebhookWorkerOutcome::Delivered {
            delivery_id: "real-restart".to_owned(),
            status_code: 204,
        }
    );

    seed_terminal_delivery_with_private_event(&store, "real-manual-retry", NOW, 1);
    assert_eq!(
        worker
            .process_next()
            .await
            .expect("manual retry initial step"),
        WebhookWorkerOutcome::DeadLettered {
            delivery_id: "real-manual-retry".to_owned(),
        }
    );
    assert_eq!(
        store
            .manually_retry_webhook_delivery("real-manual-retry", &WebhookWorkerClock::now(&clock),)
            .expect("manual retry transition"),
        mesh_llm_log_store::WebhookManualRetryOutcome::Scheduled
    );
    OperatorAuditWriter::new()
        .write(
            Arc::clone(&store),
            "log_webhook_manual_retry",
            "operator webhook retry".to_owned(),
            "succeeded",
        )
        .expect("manual retry audit");
    assert_eq!(
        worker
            .process_next()
            .await
            .expect("manual retry worker step"),
        WebhookWorkerOutcome::Delivered {
            delivery_id: "real-manual-retry".to_owned(),
            status_code: 204,
        }
    );
    assert_eq!(
        store
            .webhook_delivery("real-manual-retry")
            .unwrap()
            .unwrap()
            .state,
        WebhookDeliveryState::Succeeded
    );
    let detail: String = store
        .conn()
        .query_row(
            "SELECT detail_json FROM audit_entries WHERE action = 'log_webhook_manual_retry'",
            [],
            |row| row.get(0),
        )
        .expect("manual retry audit detail");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&detail).expect("audit detail JSON"),
        serde_json::json!({
            "actor": "trusted_local_operator",
            "source": "logs_api",
            "result": "succeeded",
            "reason": "operator webhook retry",
        })
    );
    for secret in [
        "real-manual-retry",
        "prompt-secret",
        "completion-secret",
        "/private/secret",
        "webhook-secret",
        "127.0.0.1",
    ] {
        assert!(!detail.contains(secret), "audit leaked {secret}");
    }
    server.wait_for_requests(3).await;
    assert_private_delivery_storage(&store, "real-restart");
    assert_private_delivery_storage(&store, "real-manual-retry");
    server.shutdown().await;
}

#[tokio::test]
async fn worker_delivers_a_redacted_terminal_payload_on_success() {
    let (store, _root) = open_store();
    seed_terminal_delivery(&store, "success", 2);
    let transport = Arc::new(FakeTransport::new([FakeReply::Status(204)]));

    let outcome = worker(Arc::clone(&store), transport.clone())
        .process_next()
        .await
        .expect("worker step");

    assert_eq!(
        outcome,
        WebhookWorkerOutcome::Delivered {
            delivery_id: "success".to_string(),
            status_code: 204,
        }
    );
    assert_eq!(
        store.webhook_delivery("success").unwrap().unwrap().state,
        WebhookDeliveryState::Succeeded
    );
    let payloads = transport.payloads();
    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0].request_id, "request-success");
    assert_eq!(payloads[0].outcome, WebhookTerminalOutcome::Completed);
    assert_eq!(payloads[0].status_code, None);
    let payload_json = serde_json::to_string(&payloads[0]).expect("payload json");
    for forbidden in [
        "prompt-secret",
        "/private/secret",
        "artifact.invalid",
        "webhook",
    ] {
        assert!(
            !payload_json.contains(forbidden),
            "payload leaked {forbidden}"
        );
    }
}

#[tokio::test]
async fn worker_schedules_deterministic_retry_for_5xx() {
    let (store, _root) = open_store();
    seed_terminal_delivery(&store, "five-xx", 2);

    let outcome = worker(
        Arc::clone(&store),
        Arc::new(FakeTransport::new([FakeReply::Status(503)])),
    )
    .process_next()
    .await
    .expect("worker step");

    assert_eq!(
        outcome,
        WebhookWorkerOutcome::RetryScheduled {
            delivery_id: "five-xx".to_string(),
        }
    );
    let record = store.webhook_delivery("five-xx").unwrap().unwrap();
    assert_eq!(record.state, WebhookDeliveryState::Retry);
    assert_eq!(record.terminal_status_code, None);
    assert_eq!(record.response_status_code, Some(503));
    assert_eq!(
        record.last_error_code,
        Some(WebhookDeliveryErrorCode::Http5xx)
    );
    assert_eq!(
        record.next_attempt_at.as_deref(),
        Some("2026-08-04T12:00:01.250000000Z")
    );
}

#[tokio::test]
async fn worker_schedules_retry_for_timeout_without_persisting_error_text() {
    let (store, _root) = open_store();
    seed_terminal_delivery(&store, "timeout", 2);

    let outcome = worker(
        Arc::clone(&store),
        Arc::new(FakeTransport::new([FakeReply::Timeout])),
    )
    .process_next()
    .await
    .expect("worker step");

    assert_eq!(
        outcome,
        WebhookWorkerOutcome::RetryScheduled {
            delivery_id: "timeout".to_string(),
        }
    );
    let record = store.webhook_delivery("timeout").unwrap().unwrap();
    assert_eq!(record.terminal_status_code, None);
    assert_eq!(record.response_status_code, None);
    assert_eq!(
        record.last_error_code,
        Some(WebhookDeliveryErrorCode::Timeout)
    );
    let raw_error: Option<String> = store
        .conn()
        .query_row(
            "SELECT error_msg FROM webhook_deliveries WHERE delivery_id = 'timeout'",
            [],
            |row| row.get(0),
        )
        .expect("stored error");
    assert!(raw_error.is_none());
}

#[tokio::test]
async fn worker_dead_letters_after_the_configured_attempt_bound() {
    let (store, _root) = open_store();
    seed_terminal_delivery(&store, "dead-letter", 1);

    let outcome = worker(
        Arc::clone(&store),
        Arc::new(FakeTransport::new([FakeReply::Status(503)])),
    )
    .process_next()
    .await
    .expect("worker step");

    assert_eq!(
        outcome,
        WebhookWorkerOutcome::DeadLettered {
            delivery_id: "dead-letter".to_string(),
        }
    );
    let record = store.webhook_delivery("dead-letter").unwrap().unwrap();
    assert_eq!(record.state, WebhookDeliveryState::DeadLetter);
    assert_eq!(record.attempt_number, 1);
}

#[tokio::test]
async fn worker_transport_wait_does_not_block_the_serving_runtime() {
    let (store, _root) = open_store();
    seed_terminal_delivery(&store, "nonblocking", 1);
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let transport = Arc::new(GatedTransport {
        started: Arc::clone(&started),
        release: Arc::clone(&release),
    });
    let worker = worker(store, transport);
    let task = tokio::spawn(async move { worker.process_next().await });

    started.notified().await;
    let served = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let served_on_runtime = Arc::clone(&served);
    tokio::spawn(async move {
        served_on_runtime.store(true, std::sync::atomic::Ordering::Release);
    })
    .await
    .expect("serving task");
    assert!(served.load(std::sync::atomic::Ordering::Acquire));

    release.notify_one();
    assert!(matches!(
        task.await.expect("worker task").expect("worker result"),
        WebhookWorkerOutcome::Delivered { .. }
    ));
}

#[test]
fn worker_rejects_endpoints_that_configuration_validation_forbids() {
    let (store, _root) = open_store();
    let config = LoggingWebhookConfig {
        enabled: true,
        url: Some("https://operator:secret@example.invalid/hook?token=secret".to_string()),
        max_attempts: 3,
        timeout_secs: 1,
        dead_letter_retention_secs: 3_600,
    };

    let result = WebhookDeliveryWorker::from_config(
        store,
        &config,
        Arc::new(FakeTransport::new([])),
        Arc::new(FixedClock),
        Arc::new(FixedJitter(0)),
    );
    let Err(error) = result else {
        panic!("unsafe endpoint must be rejected");
    };

    assert_eq!(error, WebhookWorkerConfigError::InvalidEndpoint);
    assert!(!error.to_string().contains("secret"));
}
