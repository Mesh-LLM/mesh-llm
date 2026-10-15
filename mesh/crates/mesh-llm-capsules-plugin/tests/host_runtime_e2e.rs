//! Real-host end-to-end test: drives the compiled `capsules`
//! binary through an ACTUAL running `mesh-llm-host-runtime` process (the
//! `mesh-llm serve` binary), not the faithful-but-fake-host stand-in in
//! `tests/interop.rs`. This closes the gap `tests/interop.rs` documents at
//! its top: it proves the plugin's `inference::provider()` registration is
//! dispatched real OpenAI-compatible HTTP traffic by mesh-llm's own routing
//! (`mesh-llm-host-runtime::network::openai::ingress::route_missing_local_model`
//! / `try_route_plugin_model`), not just by a test harness that reimplements
//! the host side of the wire protocol.
//!
//! Ignored by default and gated on `MESH_LLM_HOST_BIN` because
//! `mesh-llm-host-runtime` is not published to crates.io and pulls in
//! mesh-llm's native GPU/runtime toolchain (iroh, skippy-runtime, etc.) —
//! exactly the reason `tests/interop.rs` gives for not depending on it
//! directly. This repo's CI cannot build that workspace, so this test cannot
//! run there. See `docs/REAL-HOST-VERIFICATION.md` for how to run it.
//!
//! Run with:
//!   MESH_LLM_HOST_BIN=/path/to/mesh-llm/target/debug/mesh-llm \
//!     cargo test --test host_runtime_e2e -- --ignored --test-threads=1
//!
//! Run against the deliberately-broken mutant (proves this test actually
//! discriminates correct behavior from the exact regression the admission
//! policy exists to catch — same shape as `tests/interop.rs`'s mutant proof,
//! but exercised through the real host this time):
//!   MESH_LLM_HOST_BIN=/path/to/mesh-llm/target/debug/mesh-llm \
//!     cargo test --test host_runtime_e2e --features mutant-blocked-model-served \
//!     -- --ignored --test-threads=1
//! (expected: `denies_blocked_model_end_to_end_through_real_host` FAILS)

use std::io::Write;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

const PLUGIN_BIN: &str = env!("CARGO_BIN_EXE_capsules");
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

struct RealHost {
    child: Child,
    api_port: u16,
    home_dir: PathBuf,
    capsule_data_dir: PathBuf,
}

impl RealHost {
    async fn spawn() -> Self {
        let host_bin = std::env::var("MESH_LLM_HOST_BIN").expect(
            "MESH_LLM_HOST_BIN must point to a real `mesh-llm` binary \
             (built from mesh-llm's own workspace: `cargo build -p mesh-llm --bin mesh-llm`). \
             See docs/REAL-HOST-VERIFICATION.md.",
        );
        assert!(
            std::path::Path::new(&host_bin).is_file(),
            "MESH_LLM_HOST_BIN '{host_bin}' does not exist"
        );

        // Deliberately NOT `std::env::temp_dir()`: on macOS that resolves to
        // a long per-process `/var/folders/.../T/` path, and the real host
        // derives its plugin Unix-domain-socket path from `$HOME`
        // (`{HOME}/.mesh-llm/run/plugins/p<pid>-<nonce>-<plugin>.sock`) —
        // long enough to exceed `sockaddr_un`'s ~104-byte `sun_path` limit
        // and fail the plugin's bind with no further detail. `/tmp` directly
        // keeps the whole path well under that limit.
        let home_dir = PathBuf::from("/tmp").join(format!("ape2e-{}", nonce()));
        std::fs::create_dir_all(&home_dir).expect("create isolated HOME for real host");

        let api_port = pick_port();
        let console_port = pick_port();

        let config_path = home_dir.join("config.toml");
        let mut config_file = std::fs::File::create(&config_path).expect("create config.toml");
        writeln!(
            config_file,
            r#"version = 1

[[plugin]]
name = "capsules"
enabled = true
command = "{PLUGIN_BIN}"
args = []
"#
        )
        .expect("write config.toml");

        let log_path = home_dir.join("run.log");
        let log_file = std::fs::File::create(&log_path).expect("create run.log");

        // Isolated per-run plugin state (signing key + durable
        // ledger + observed-lifecycle-events log) -- inherited by the
        // plugin subprocess the host spawns, same as
        // `CAPSULES_BLOCKED_MODELS` already is below.
        let capsule_data_dir = home_dir.join("capsule-data");

        let child = Command::new(&host_bin)
            .args([
                "serve",
                "--config",
                config_path.to_str().unwrap(),
                "--port",
                &api_port.to_string(),
                "--console",
                &console_port.to_string(),
                "--headless",
                "--disable-iroh-relays",
                "--log-format",
                "json",
            ])
            .env("HOME", &home_dir)
            .env(
                "CAPSULES_BLOCKED_MODELS",
                "blocked-test-model,allowed-test-model",
            )
            .env("CAPSULES_DATA_DIR", &capsule_data_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_file.try_clone().expect("clone log fd")))
            .stderr(Stdio::from(log_file))
            .kill_on_drop(true)
            .spawn()
            .expect("spawn real mesh-llm host");

        let host = Self {
            child,
            api_port,
            home_dir,
            capsule_data_dir,
        };
        host.wait_for_plugin_model("allowed-test-model").await;
        host
    }

    fn capsule_data_dir(&self) -> &std::path::Path {
        &self.capsule_data_dir
    }

    /// Poll the host's own `/v1/models` until the plugin's health-probe
    /// cycle (`mesh-llm-host-runtime::plugin::health`) has picked up the
    /// declared model — proves the real Initialize handshake, the real HTTP
    /// health probe, and the real routing table are all live, not just that
    /// the process started.
    async fn wait_for_plugin_model(&self, model: &str) {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        let url = format!("http://127.0.0.1:{}/v1/models", self.api_port);
        let deadline = tokio::time::Instant::now() + STARTUP_TIMEOUT;
        loop {
            if tokio::time::Instant::now() > deadline {
                let log = std::fs::read_to_string(self.home_dir.join("run.log"))
                    .unwrap_or_else(|e| format!("<could not read run.log: {e}>"));
                panic!(
                    "real host never advertised '{model}' via /v1/models within {STARTUP_TIMEOUT:?}. \
                     Host log:\n{log}"
                );
            }
            if let Ok(resp) = client.get(&url).send().await {
                if let Ok(body) = resp.text().await {
                    if body.contains(model) {
                        return;
                    }
                }
            }
            sleep(Duration::from_millis(500)).await;
        }
    }

    async fn post_chat_completions(&self, body: &serde_json::Value) -> reqwest::Response {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap();
        timeout(
            REQUEST_TIMEOUT,
            client
                .post(format!(
                    "http://127.0.0.1:{}/v1/chat/completions",
                    self.api_port
                ))
                .json(body)
                .send(),
        )
        .await
        .expect("request to real host did not time out")
        .expect("POST /v1/chat/completions to real host")
    }

    async fn post_chat_completions_raw(&self, body: &str) -> reqwest::Response {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap();
        timeout(
            REQUEST_TIMEOUT,
            client
                .post(format!(
                    "http://127.0.0.1:{}/v1/chat/completions",
                    self.api_port
                ))
                .header("content-type", "application/json")
                .body(body.to_string())
                .send(),
        )
        .await
        .expect("request to real host did not time out")
        .expect("POST /v1/chat/completions to real host")
    }
}

impl Drop for RealHost {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
        let _ = std::fs::remove_dir_all(&self.home_dir);
    }
}

fn nonce() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn pick_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .map(|l| l.local_addr().unwrap().port())
        .expect("bind ephemeral port")
}

/// (A) Deny, dispatched by the real host's real routing table — not a direct
/// call to the plugin's own HTTP port. Proves `inference::provider()`
/// registration + exact-name model dispatch
/// (`mesh-llm-host-runtime::network::openai::ingress::try_route_plugin_model`)
/// works against the actual host, matching `tests/interop.rs`'s test (C) but
/// through the real process instead of the fake-host stand-in.
///
/// This is also the mutant-discriminating check: built with
/// `--features mutant-blocked-model-served`, the real host still dispatches the
/// request to the (broken) plugin, which now answers 200 instead of 403 —
/// this assertion fails, proving the check actually exercises the plugin's
/// decision logic end-to-end rather than passing regardless of it.
#[tokio::test]
#[ignore = "requires a real mesh-llm host binary; see MESH_LLM_HOST_BIN and docs/REAL-HOST-VERIFICATION.md"]
async fn denies_blocked_model_end_to_end_through_real_host() {
    let host = RealHost::spawn().await;
    let resp = host
        .post_chat_completions(&serde_json::json!({
            "model": "blocked-test-model",
            "messages": []
        }))
        .await;
    assert_eq!(resp.status(), reqwest::StatusCode::FORBIDDEN);
    let body: serde_json::Value = resp.json().await.expect("valid JSON body");
    assert_eq!(body["error"]["code"], "blocked_model_prefix");
}

/// (B) Allow, dispatched by the real host. `allowed-test-model` is advertised
/// by the plugin (so the host's exact-name routing sends it there) but does
/// not match the blocked prefix, so the plugin's own decision logic allows
/// it. Demonstrates the full allow/deny matrix end-to-end, not just deny.
#[tokio::test]
#[ignore = "requires a real mesh-llm host binary; see MESH_LLM_HOST_BIN and docs/REAL-HOST-VERIFICATION.md"]
async fn allows_unblocked_advertised_model_end_to_end_through_real_host() {
    let host = RealHost::spawn().await;
    let resp = host
        .post_chat_completions(&serde_json::json!({
            "model": "allowed-test-model",
            "messages": []
        }))
        .await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.expect("valid JSON body");
    assert_eq!(body["admission_policy"]["decision"], "allow");
}

/// (C) Malformed input through the real host's standard ingress — with an
/// important, verified architectural difference from `tests/interop.rs`'s
/// (D): on the real host, a syntactically-unparseable body never reaches the
/// plugin at all. `network::openai::ingress::api_proxy` parses the "model"
/// field from the request body BEFORE any routing decision (including the
/// plugin fallback) can be made; when that parse fails outright, the host
/// itself returns a non-200 error (observed: 503, "single target None
/// unavailable") without ever opening a connection to the plugin's HTTP
/// endpoint. So the plugin's own `body_parse_failure` fail-safe branch
/// (`decision::malformed`, and the `mutant-malformed-body-served` feature that
/// breaks it) is real and correctly wired — proven directly in
/// `tests/interop.rs` and `src/decision.rs`'s unit tests, which talk to the
/// plugin's HTTP endpoint directly — but is NOT reachable through this real
/// host's `/v1/chat/completions` ingress in this single-plugin topology, so
/// it cannot be re-verified at this layer. What IS verified here: the real
/// host's OWN behavior on unparseable input is fail-safe (never a silent
/// 200), independent of the plugin.
///
/// (A second, more surprising case — valid JSON with a missing/non-string
/// "model" field — is NOT malformed from the real host's point of view: its
/// model-resolution fallback rewrites such requests to name the sole
/// available target ("blocked-test-model" in this topology) before
/// forwarding, so the plugin correctly denies them for the
/// `blocked_model_prefix` reason, not `body_parse_failure`. Verified
/// manually; see docs/REAL-HOST-VERIFICATION.md. Only a body that fails to parse
/// as JSON at all defeats the host's routing outright.)
#[tokio::test]
#[ignore = "requires a real mesh-llm host binary; see MESH_LLM_HOST_BIN and docs/REAL-HOST-VERIFICATION.md"]
async fn malformed_body_fails_safe_at_the_real_host_before_reaching_the_plugin() {
    let host = RealHost::spawn().await;
    let resp = host.post_chat_completions_raw("{not valid json").await;
    assert_ne!(
        resp.status(),
        reqwest::StatusCode::OK,
        "unparseable body must never be silently allowed, at the host or the plugin"
    );
}

/// (D) The #1332 integration proof: a REAL chat-completion inference event,
/// dispatched by the real `mesh-llm` host to this plugin exactly like test
/// (B), must produce a signed, hash-chained, ledgered AAC -- not just an
/// admit/deny decision. Two independent things are checked, corresponding to
/// the task's two required wiring points:
///
/// 1. **Sealing wired to the plugin's own handler**: the plugin's own
///    `/v1/chat/completions` handler calls `capsule_emit::CapsuleState`
///    directly with the real request/response bytes it exchanged; the
///    resulting capsule must be durably ledgered and verify offline against
///    capsule-emit's independent `verify::verify_offline` (COSE
///    signature, capsule_id recomputation, COSE-payload-matches-capsule).
/// 2. **Sealing wired to the lifecycle hook**: the real host
///    (a mesh-llm build with the #1331 lifecycle hooks)
///    independently publishes a `RawProxy`/`Terminal` envelope on the
///    `openai.exchange.v1` mesh channel for this SAME exchange
///    (`network/openai/ingress.rs::try_route_plugin_model`, wired into
///    production, not just tested) -- the plugin's own `on_channel_message`
///    handler must have received and logged it, proving the plugin observes
///    the host's lifecycle broadcast independently of its own HTTP view.
///
/// Then adversarially mutates the observed exchange two ways and confirms
/// both are caught: (a) flipping a signature byte on the ledgered COSE
/// statement must fail COSE verification; (b) recomputing `capsule_id` after
/// mutating the captured `agent_output_digest` (as an attacker forging a
/// different response would need to) must produce a DIFFERENT id, proving
/// the digest is bound to what was actually observed.
#[tokio::test]
#[ignore = "requires a real mesh-llm host binary; see MESH_LLM_HOST_BIN and docs/REAL-HOST-VERIFICATION.md"]
async fn allowed_exchange_emits_a_signed_chained_ledgered_capsule_and_publishes_lifecycle_event() {
    let host = RealHost::spawn().await;
    let resp = host
        .post_chat_completions(&serde_json::json!({
            "model": "allowed-test-model",
            "messages": [{"role": "user", "content": "hello from the real host e2e test"}],
            "client_nonce": "e2e-real-inference-nonce-001",
        }))
        .await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.expect("valid JSON body");
    assert_eq!(body["admission_policy"]["decision"], "allow");
    let capsule_id = body["admission_policy"]["capsule_id"]
        .as_str()
        .expect("allow response carries a minted capsule_id")
        .to_string();

    // Give the plugin's async on_channel_message handler a moment to persist
    // the host's lifecycle broadcast (delivery is fire-and-forget on the
    // host side, same as production).
    tokio::time::sleep(Duration::from_millis(750)).await;

    // --- (1) sealing <-> the plugin's handler: real ledger, real COSE ---
    let keys_dir = host.capsule_data_dir().join("keys");
    let pubkey_pem = std::fs::read_to_string(keys_dir.join("node-key.pub.pem"))
        .expect("plugin persisted its public key");
    let verifying_key = capsule_emit_lib::keys::load_verifying_key_pem(&pubkey_pem)
        .expect("parse persisted public key");

    let ledger_dir = host.capsule_data_dir().join("ledger");
    let (ledger, report) =
        capsule_emit_lib::ledger::Ledger::open(&ledger_dir).expect("reopen plugin's ledger");
    assert!(
        report.valid_entries >= 1,
        "ledger must contain the emitted capsule"
    );
    let entry = ledger
        .lookup(&capsule_id)
        .expect("ledger lookup")
        .expect("capsule_id from the HTTP response must be in the ledger");

    let verify_report = capsule_emit_lib::verify::verify_offline(
        &entry.capsule,
        &entry.signed_statement,
        &verifying_key,
        None,
    );
    assert!(
        verify_report.ok(),
        "real-host-produced capsule must verify offline: {:?}",
        verify_report.findings
    );
    assert_eq!(entry.capsule["capsule_id"], capsule_id);
    assert_eq!(
        entry.capsule["model_attestation"]["model_id"],
        "allowed-test-model"
    );
    assert_eq!(entry.capsule["assurance"]["effect_mode"], "confirmed");

    // --- (2) sealing <-> the lifecycle hook: the host's own ---
    //         terminal broadcast for this same exchange was received.
    let lifecycle_log_path = host.capsule_data_dir().join("lifecycle-events.jsonl");
    let lifecycle_log = std::fs::read_to_string(&lifecycle_log_path).unwrap_or_default();
    let events: Vec<serde_json::Value> = lifecycle_log
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    assert!(
        events.iter().any(|e| {
            e["phase"] == "terminal"
                && e["dispatch_path"] == "raw_proxy"
                && e["model"] == "allowed-test-model"
                && e["status"] == 200
        }),
        "expected the host's real openai.exchange.v1 terminal broadcast for \
         allowed-test-model/200 to have been received by the plugin; got: {events:#?}"
    );

    // --- (3) adversarial mutation: tampered signature must fail verification ---
    let mut tampered_statement = entry.signed_statement.clone();
    let last = tampered_statement.len() - 1;
    tampered_statement[last] ^= 0xFF;
    let tampered_report = capsule_emit_lib::verify::verify_offline(
        &entry.capsule,
        &tampered_statement,
        &verifying_key,
        None,
    );
    assert!(
        !tampered_report.ok(),
        "a bit-flipped signed statement must NOT verify"
    );

    // --- (4) adversarial mutation: forging a different observed response ---
    //         must change capsule_id, not silently keep the original one.
    let mut mutated_capsule = entry.capsule.clone();
    mutated_capsule["model_attestation"]["compute_attestation"]["agent_output_digest"] =
        serde_json::json!("f".repeat(64));
    let mutated_id = capsule_emit_lib::jcs::compute_capsule_id(&mutated_capsule)
        .expect("recompute digest over mutated capsule");
    assert_ne!(
        mutated_id, capsule_id,
        "mutating the observed response digest must change capsule_id"
    );
}
