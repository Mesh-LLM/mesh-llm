//! The plugin HTTP routes the Evidence page reads (`web-ui/DATA-ROUTES.md`).
//!
//! The console serves a plugin's declared GET bindings at
//! `/api/plugins/<plugin>/http/<path>`, and the page reaches them through its
//! host's `fetchPlugin("http/<path>")`. Every route reads this plugin's own
//! ledger directory, so the installed page needs nothing but the plugin:
//!
//! | Route | Answers |
//! | --- | --- |
//! | `ledger` | `{records, node_pub_key_pem}`: this node's records, padding left out |
//! | `ledger/signed-statement?capsule_id=` | `{signed_statement_b64}`, or null |
//! | `ledger/disclosure?capsule_id=` | `{disclosure}`, the kept request/response text, or null |
//! | `panes/pane-a`, `panes/pane-b` | the pane JSON ([`crate::evidence_panes`]) |
//! | `panes/pane-c[?exchange_id=]` | the Exchanges list, or one exchange's drilldown |
//! | `peer-key?peer=` | `{announced_key_id}`: the key the operator says that peer signs with, or null |
//! | `lookup?exchange_id=` or `?client_nonce=` | `{found, exchange_key, capsule_id, role}`: this node's own record of that exchange, for the console's chat and Logs contributions |
//!
//! The host turns query parameters into typed JSON (numbers stay numbers),
//! so a capsule id that is not a string is refused rather than guessed back.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use mesh_llm_plugin::{http, DeclarativePluginBuilder, PluginError};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::evidence_panes::{
    attach_asked_of_you, build_pane_json, read_capsule_records, read_received_log,
};
use crate::peer_keys::announced_key_in;

/// The directory this node's log of requests made of it
/// (`received_log.jsonl`, written by `received_log`) lives in: the drill's
/// "Asked of you".
pub const ENV_RECEIVED_LOG_DIR: &str = "CAPSULES_RECEIVED_LOG_DIR";
/// Where that log is read from when [`ENV_RECEIVED_LOG_DIR`] is unset:
/// `<data dir>/received-log`.
pub const DEFAULT_RECEIVED_LOG_SUBDIR: &str = "received-log";

/// A route that takes no arguments. Unknown members (the host may add query
/// parameters) are ignored.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct NoArgs {}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CapsuleIdArgs {
    /// The record's id: 64 lowercase hex.
    pub capsule_id: Value,
}

/// The host's ids for one exchange: either is enough.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct LookupArgs {
    /// The host's exchange id (the Logs request's).
    #[serde(default)]
    pub exchange_id: Option<Value>,
    /// The client nonce the host forwarded (the chat message's).
    #[serde(default)]
    pub client_nonce: Option<Value>,
}

/// A query value as text: the host types query values, so a nonce of digits
/// arrives as a number.
fn query_text(value: &Option<Value>) -> Result<Option<String>, PluginError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(Value::Number(n)) => Ok(Some(n.to_string())),
        Some(_) => Err(PluginError::invalid_params("must be a string")),
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PeerKeyArgs {
    /// The peer's endpoint id: 64 hex.
    pub peer: Value,
}

/// `peer-key`: what "Ask them for their record" judges a reply under. The
/// page never takes the key a reply names for itself. The key is read by the
/// same rule the record-push receiver uses (`peer_keys::announced_key_in`),
/// from the same registry (`peer_keys::registry`: the host's announced keys
/// and the operator's map).
pub fn peer_key_json(registry: Option<&str>, peer: &Value) -> Result<Value, PluginError> {
    let Some(peer) = peer
        .as_str()
        .map(str::trim)
        .filter(|p| p.len() == 64 && p.bytes().all(|b| b.is_ascii_hexdigit()))
    else {
        return Err(PluginError::invalid_params(
            "peer must be a 64-hex endpoint id",
        ));
    };
    Ok(json!({ "announced_key_id": announced_key_in(registry, peer) }))
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct PaneCArgs {
    /// One exchange's drilldown instead of the list.
    #[serde(default)]
    pub exchange_id: Option<Value>,
}

/// The ledger directory and the node's public key, shared by every route.
#[derive(Clone)]
pub struct EvidenceSource {
    pub ledger_dir: PathBuf,
    pub node_pub_key_pem: Option<String>,
    /// Where this node logs requests made of it, if anywhere.
    pub received_log_dir: Option<PathBuf>,
}

impl EvidenceSource {
    /// `received_log_dir`: [`ENV_RECEIVED_LOG_DIR`] when set, else
    /// `<data_dir>/`[`DEFAULT_RECEIVED_LOG_SUBDIR`]. Either way only a
    /// directory that exists counts, so a node that keeps no log shows
    /// "not shown", never an empty "0 requests".
    pub fn received_log_dir(data_dir: &Path) -> Option<PathBuf> {
        received_log_dir_for(crate::settings::var_os(ENV_RECEIVED_LOG_DIR), data_dir)
    }
}

fn received_log_dir_for(env: Option<std::ffi::OsString>, data_dir: &Path) -> Option<PathBuf> {
    let dir = env
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join(DEFAULT_RECEIVED_LOG_SUBDIR));
    dir.is_dir().then_some(dir)
}

fn is_record_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn record_id(value: &Value) -> Result<&str, PluginError> {
    value
        .as_str()
        .filter(|id| is_record_id(id))
        .ok_or_else(|| PluginError::invalid_params("capsule_id must be 64 lowercase hex"))
}

/// `{records, node_pub_key_pem}`.
pub fn ledger_json(source: &EvidenceSource) -> Value {
    json!({
        "records": read_capsule_records(&source.ledger_dir),
        "node_pub_key_pem": source.node_pub_key_pem,
    })
}

/// `{signed_statement_b64}`: the record's detached COSE_Sign1, or null.
pub fn signed_statement_json(ledger_dir: &Path, capsule_id: &str) -> Value {
    let path = ledger_dir
        .join("signed-statements")
        .join(format!("{capsule_id}.cose"));
    let b64 = std::fs::read(path)
        .ok()
        .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes));
    json!({ "signed_statement_b64": b64 })
}

/// `{disclosure}`: the request/response text kept beside the record, or null.
pub fn disclosure_json(ledger_dir: &Path, capsule_id: &str) -> Value {
    let path = ledger_dir
        .join("disclosures")
        .join(format!("{capsule_id}.json"));
    let disclosure = std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .filter(Value::is_object);
    json!({ "disclosure": disclosure })
}

/// One pane, or an error for a pane this plugin does not build.
pub fn pane_json(
    ledger_dir: &Path,
    pane: &str,
    exchange_id: Option<&str>,
) -> Result<Value, PluginError> {
    build_pane_json(pane, ledger_dir, exchange_id)
        .ok_or_else(|| PluginError::invalid_params(format!("no pane {pane:?}")))
}

/// [`pane_json`] from `source`: Pane B also carries this node's log of
/// requests made of it, when it keeps one.
pub fn source_pane_json(
    source: &EvidenceSource,
    pane: &str,
    exchange_id: Option<&str>,
) -> Result<Value, PluginError> {
    let mut payload = pane_json(&source.ledger_dir, pane, exchange_id)?;
    if let (Some(dir), "pane-b") = (&source.received_log_dir, pane) {
        attach_asked_of_you(&mut payload, &read_received_log(dir));
    }
    Ok(payload)
}

fn exchange_id(args: &PaneCArgs) -> Result<Option<String>, PluginError> {
    match &args.exchange_id {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) => Ok(Some(id.clone())),
        Some(_) => Err(PluginError::invalid_params("exchange_id must be a string")),
    }
}

/// Run a file read off the async worker.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, PluginError> + Send + 'static,
) -> Result<T, PluginError> {
    tokio::task::spawn_blocking(work).await.map_err(|error| {
        PluginError::internal(format!("evidence route task did not complete: {error}"))
    })?
}

/// `builder` with the page's routes declared.
pub fn with_routes(
    builder: DeclarativePluginBuilder,
    source: EvidenceSource,
) -> DeclarativePluginBuilder {
    let source = Arc::new(source);
    let mut builder = builder;

    let s = source.clone();
    builder = builder.http_item(
        http::get("/ledger")
            .binding_id("evidence_ledger")
            .description(
                "This node's records (padding left out) and its public key, for the Evidence page.",
            )
            .input::<NoArgs>()
            .handle(move |_args, _context| {
                let s = s.clone();
                Box::pin(async move { blocking(move || Ok(ledger_json(&s))).await })
            }),
    );

    let s = source.clone();
    builder = builder.http_item(
        http::get("/ledger/signed-statement")
            .binding_id("evidence_signed_statement")
            .description("One record's detached signed statement, base64, or null.")
            .input::<CapsuleIdArgs>()
            .handle(move |args, _context| {
                let s = s.clone();
                Box::pin(async move {
                    let id = record_id(&args.capsule_id)?.to_string();
                    blocking(move || Ok(signed_statement_json(&s.ledger_dir, &id))).await
                })
            }),
    );

    let s = source.clone();
    builder = builder.http_item(
        http::get("/ledger/verdict")
            .binding_id("evidence_verdict")
            .description(
                "One referee-signed twin verdict this node holds, with this plugin's check of its \
                 signature and of this node's own record of it, or a null capsule.",
            )
            .input::<CapsuleIdArgs>()
            .handle(move |args, _context| {
                let s = s.clone();
                Box::pin(async move {
                    let id = record_id(&args.capsule_id)?.to_string();
                    blocking(move || {
                        Ok(crate::adjudication_records::verdict_json(
                            &s.ledger_dir,
                            &id,
                        ))
                    })
                    .await
                })
            }),
    );

    let s = source.clone();
    builder = builder.http_item(
        http::get("/ledger/disclosure")
            .binding_id("evidence_disclosure")
            .description("The request/response text kept beside one record, or null.")
            .input::<CapsuleIdArgs>()
            .handle(move |args, _context| {
                let s = s.clone();
                Box::pin(async move {
                    let id = record_id(&args.capsule_id)?.to_string();
                    blocking(move || Ok(disclosure_json(&s.ledger_dir, &id))).await
                })
            }),
    );

    for pane in ["pane-a", "pane-b"] {
        let s = source.clone();
        builder = builder.http_item(
            http::get(format!("/panes/{pane}"))
                .binding_id(format!("evidence_{}", pane.replace('-', "_")))
                .description("One of the Evidence page's panes, built from this node's ledger.")
                .input::<NoArgs>()
                .handle(move |_args, _context| {
                    let s = s.clone();
                    Box::pin(
                        async move { blocking(move || source_pane_json(&s, pane, None)).await },
                    )
                }),
        );
    }

    builder = builder.http_item(
        http::get("/peer-key")
            .binding_id("evidence_peer_key")
            .description(
                "The key one peer is known to sign with (announced by its host, or set in \
                 CAPSULES_PEER_KEYS), or null.",
            )
            .input::<PeerKeyArgs>()
            .handle(move |args, _context| {
                Box::pin(async move {
                    let registry = crate::peer_keys::registry();
                    peer_key_json(registry.as_deref(), &args.peer)
                })
            }),
    );

    let s = source.clone();
    builder = builder.http_item(
        http::get("/lookup")
            .binding_id("evidence_lookup")
            .description(
                "This node's own record of one exchange, by the host's exchange id or client nonce.",
            )
            .input::<LookupArgs>()
            .handle(move |args, _context| {
                let s = s.clone();
                Box::pin(async move {
                    let exchange_id = query_text(&args.exchange_id)?;
                    let client_nonce = query_text(&args.client_nonce)?;
                    if exchange_id.is_none() && client_nonce.is_none() {
                        return Err(PluginError::invalid_params("give exchange_id or client_nonce"));
                    }
                    blocking(move || {
                        let records = read_capsule_records(&s.ledger_dir);
                        Ok(crate::evidence_panes::find_own_record(
                            &records,
                            exchange_id.as_deref(),
                            client_nonce.as_deref(),
                        ))
                    })
                    .await
                })
            }),
    );

    let s = source;
    builder = builder.http_item(
        http::get("/panes/pane-c")
            .binding_id("evidence_pane_c")
            .description("The Exchanges list, or one exchange's drilldown with ?exchange_id=.")
            .input::<PaneCArgs>()
            .handle(move |args, _context| {
                let s = s.clone();
                Box::pin(async move {
                    let exchange_id = exchange_id(&args)?;
                    blocking(move || source_pane_json(&s, "pane-c", exchange_id.as_deref())).await
                })
            }),
    );

    builder
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn record(id: &str) -> Value {
        json!({
            "capsule_id": id,
            "timestamp": "2026-09-27T10:00:00Z",
            "model_attestation": {"compute_attestation": {"x-mesh-poc-v1": {"role": "served"}}},
            "effect": {"request_digest": "a".repeat(64), "response_digest": "b".repeat(64)},
        })
    }

    fn ledger(lines: &[Value]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let mut file = std::fs::File::create(dir.path().join("capsules.jsonl")).unwrap();
        for line in lines {
            writeln!(file, "{line}").unwrap();
        }
        dir
    }

    #[test]
    fn the_ledger_route_lists_records_and_never_padding() {
        let padding = json!({"capsule_id": "c".repeat(64), "record_type": "padding", "store_nonce": "d".repeat(64)});
        let dir = ledger(&[record(&"1".repeat(64)), padding]);
        let source = EvidenceSource {
            ledger_dir: dir.path().into(),
            node_pub_key_pem: Some("PEM".into()),
            received_log_dir: None,
        };
        let body = ledger_json(&source);
        assert_eq!(body["records"].as_array().unwrap().len(), 1);
        assert_eq!(body["node_pub_key_pem"], json!("PEM"));
    }

    #[test]
    fn a_missing_ledger_is_an_empty_list_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let source = EvidenceSource {
            ledger_dir: dir.path().into(),
            node_pub_key_pem: None,
            received_log_dir: None,
        };
        assert_eq!(ledger_json(&source)["records"], json!([]));
        assert_eq!(
            pane_json(dir.path(), "pane-c", None).unwrap()["rows"],
            json!([])
        );
    }

    #[test]
    fn signed_statements_and_disclosures_are_read_by_id_or_null() {
        let dir = ledger(&[]);
        let id = "e".repeat(64);
        std::fs::create_dir_all(dir.path().join("signed-statements")).unwrap();
        std::fs::write(
            dir.path()
                .join("signed-statements")
                .join(format!("{id}.cose")),
            [1u8, 2, 3],
        )
        .unwrap();
        std::fs::create_dir_all(dir.path().join("disclosures")).unwrap();
        std::fs::write(
            dir.path().join("disclosures").join(format!("{id}.json")),
            br#"{"request":"hi"}"#,
        )
        .unwrap();
        assert_eq!(
            signed_statement_json(dir.path(), &id)["signed_statement_b64"],
            json!("AQID")
        );
        assert_eq!(
            disclosure_json(dir.path(), &id)["disclosure"]["request"],
            json!("hi")
        );
        let other = "f".repeat(64);
        assert_eq!(
            signed_statement_json(dir.path(), &other)["signed_statement_b64"],
            Value::Null
        );
        assert_eq!(
            disclosure_json(dir.path(), &other)["disclosure"],
            Value::Null
        );
    }

    #[test]
    fn only_a_64_hex_string_is_a_record_id() {
        assert!(record_id(&json!("a".repeat(64))).is_ok());
        for bad in [
            json!("../keys/node-key.pem"),
            json!("A".repeat(64)),
            json!(1234),
            json!("a".repeat(63)),
        ] {
            assert!(record_id(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn pane_c_answers_a_drilldown_for_a_string_exchange_id_only() {
        let dir = ledger(&[record(&"1".repeat(64))]);
        let args = PaneCArgs {
            exchange_id: Some(json!(format!("digest:{}", "a".repeat(64)))),
        };
        let id = exchange_id(&args).unwrap();
        assert_eq!(
            pane_json(dir.path(), "pane-c", id.as_deref()).unwrap()["found"],
            json!(true)
        );
        assert!(exchange_id(&PaneCArgs {
            exchange_id: Some(json!(5))
        })
        .is_err());
        assert!(pane_json(dir.path(), "pane-z", None).is_err());
    }

    /// The page reaches exactly these, at `/api/plugins/<plugin>/http/<path>`.
    #[test]
    fn the_manifest_declares_every_route_the_page_reads_as_a_get() {
        use mesh_llm_plugin::{plugin_server_info, Plugin, PluginMetadata};
        let dir = tempfile::tempdir().unwrap();
        let builder = DeclarativePluginBuilder::new(PluginMetadata::new(
            "capsules",
            "0.0.0",
            plugin_server_info("capsules", "0.0.0", "t", "t", None::<String>),
        ));
        let plugin = with_routes(
            builder,
            EvidenceSource {
                ledger_dir: dir.path().into(),
                node_pub_key_pem: None,
                received_log_dir: None,
            },
        )
        .build();
        let manifest = plugin
            .manifest()
            .expect("a declarative plugin publishes a manifest");
        let mut routes: Vec<(i32, String)> = manifest
            .http_bindings
            .iter()
            .map(|b| (b.method, b.path.clone()))
            .collect();
        routes.sort();
        let get = mesh_llm_plugin::proto::HttpMethod::Get as i32;
        assert_eq!(
            routes,
            [
                "/ledger",
                "/ledger/disclosure",
                "/ledger/signed-statement",
                "/ledger/verdict",
                "/lookup",
                "/panes/pane-a",
                "/panes/pane-b",
                "/panes/pane-c",
                "/peer-key",
            ]
            .iter()
            .map(|p| (get, p.to_string()))
            .collect::<Vec<_>>()
        );
        for binding in &manifest.http_bindings {
            assert!(
                binding.operation_name.is_some(),
                "{} is served by an operation",
                binding.path
            );
        }
    }

    #[test]
    fn a_peer_key_is_the_announced_one_or_null_never_a_guess() {
        let peer = "ab".repeat(32);
        let registry = format!("{{\"{peer}\": \"CDef\"}}");
        // As written, for exactly that id (the receiver's rule).
        assert_eq!(
            peer_key_json(Some(&registry), &json!(peer)).unwrap(),
            json!({"announced_key_id": "CDef"})
        );
        assert_eq!(
            peer_key_json(Some(&registry), &json!(peer.to_uppercase())).unwrap(),
            json!({"announced_key_id": null})
        );
        for registry in [
            None,
            Some(""),
            Some("not json"),
            Some("[]"),
            Some("{\"other\": \"k\"}"),
        ] {
            assert_eq!(
                peer_key_json(registry, &json!(peer)).unwrap(),
                json!({"announced_key_id": null})
            );
        }
        assert_eq!(
            announced_key_in(Some(&format!("{{\"{peer}\": \"\"}}")), &peer),
            None
        );
        assert_eq!(
            announced_key_in(Some(&format!("{{\"{peer}\": 7}}")), &peer),
            None
        );
        for bad in [json!("ab"), json!(7), json!(null)] {
            assert!(peer_key_json(Some(&registry), &bad).is_err());
        }
    }

    #[test]
    fn pane_b_carries_asked_of_you_only_when_a_received_log_is_kept() {
        let dir = ledger(&[record(&"1".repeat(64))]);
        let log = tempfile::tempdir().unwrap();
        std::fs::write(
            log.path().join("received_log.jsonl"),
            "{\"ts\":\"2026-09-27T10:00:00Z\",\"path\":\"evidence-request\",\"requester_id\":\"p\",\"subject_kind\":\"record\",\"status\":\"answered\",\"reason\":null}\n",
        )
        .unwrap();
        let mut source = EvidenceSource {
            ledger_dir: dir.path().into(),
            node_pub_key_pem: None,
            received_log_dir: None,
        };
        let without = source_pane_json(&source, "pane-b", None).unwrap();
        assert!(without["rows"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r.get("asked_of_you").is_none()));
        source.received_log_dir = Some(log.path().into());
        let with = source_pane_json(&source, "pane-b", None).unwrap();
        let rows = with["rows"].as_array().unwrap();
        assert!(!rows.is_empty());
        assert!(rows
            .iter()
            .all(|r| r["asked_of_you"]["entries"].as_array().unwrap().len() == 1));
    }

    #[test]
    fn the_received_log_defaults_to_the_data_dir_and_counts_only_when_it_exists() {
        let data = tempfile::tempdir().unwrap();
        assert_eq!(
            received_log_dir_for(None, data.path()),
            None,
            "no log kept: not shown"
        );
        std::fs::create_dir(data.path().join(DEFAULT_RECEIVED_LOG_SUBDIR)).unwrap();
        assert_eq!(
            received_log_dir_for(None, data.path()),
            Some(data.path().join(DEFAULT_RECEIVED_LOG_SUBDIR))
        );
        let other = tempfile::tempdir().unwrap();
        assert_eq!(
            received_log_dir_for(Some(other.path().as_os_str().to_owned()), data.path()),
            Some(other.path().to_path_buf()),
            "the env var wins"
        );
        assert_eq!(
            received_log_dir_for(Some("".into()), data.path()),
            Some(data.path().join(DEFAULT_RECEIVED_LOG_SUBDIR)),
            "an empty env var is unset"
        );
    }

    #[test]
    fn lookup_finds_this_nodes_record_by_exchange_id_or_nonce() {
        let mut a = record(&"a".repeat(64));
        a["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"] = json!({
            "role": "requested", "client_nonce": "n-1",
            "serving_provenance": {"exchange_id": "ex-1"},
        });
        let records = vec![a];
        let by_id = crate::evidence_panes::find_own_record(&records, Some("ex-1"), None);
        assert_eq!(by_id["found"], json!(true));
        assert_eq!(by_id["capsule_id"], json!("a".repeat(64)));
        // The Evidence page's grouping key, which the link focuses: the
        // request digest first, the host's exchange id only without one.
        assert_eq!(
            by_id["exchange_key"],
            json!(format!("digest:{}", "a".repeat(64)))
        );
        let by_nonce = crate::evidence_panes::find_own_record(&records, None, Some("n-1"));
        assert_eq!(by_nonce["capsule_id"], json!("a".repeat(64)));
        assert_eq!(
            crate::evidence_panes::find_own_record(&records, Some("ex-2"), Some("n-2")),
            json!({"found": false})
        );
        assert_eq!(
            query_text(&Some(json!(12345))).unwrap().as_deref(),
            Some("12345")
        );
        assert_eq!(query_text(&Some(json!(""))).unwrap(), None);
    }
}
