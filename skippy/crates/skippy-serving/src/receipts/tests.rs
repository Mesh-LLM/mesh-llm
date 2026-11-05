use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Result;
use skippy_receipts::binding::{verify_paths, verify_shape};
use skippy_receipts::layer_tensors::{
    GGML_F32, TensorColumn, TensorKind, UbatchColumns, f32_bytes,
};
use skippy_receipts::recorder::{
    KvCell, ReplayStep, ReplayedLayer, ReplayedPosition, Replayer, UbatchRecord,
};
use skippy_receipts::wire::{Challenge, Pick};

use super::downstream::{Downstream, HttpDownstream, StageConnection};
use super::{
    ApiState, CommitRequest, SessionSequences, StageReceipts, TurnTokens, client_routes, router,
    spawn_api,
};

const SESSION: &str = "session-a";
const SEQ: i32 = 3;
const POSITIONS: std::ops::Range<u32> = 4..10;
fn turn() -> TurnTokens {
    TurnTokens {
        generation_start: 7,
        final_token: Some(42),
        sampled: 4,
        sampler: skippy_receipts::turn::GREEDY.to_owned(),
    }
}

struct FixedSessions(BTreeMap<String, i32>);

impl SessionSequences for FixedSessions {
    fn sequence(&self, session: &str) -> Option<i32> {
        self.0.get(session).copied()
    }
}

const KINDS: [TensorKind; 5] = [
    TensorKind::Residual,
    TensorKind::MixingInput,
    TensorKind::MixingOutput,
    TensorKind::MatmulInput,
    TensorKind::MatmulOutput,
];

/// One tensor of each kind in every layer, derived from the position: what
/// a deterministic model would compute.
struct SyntheticColumns(BTreeMap<(u32, TensorKind, usize), Vec<u8>>);

impl SyntheticColumns {
    fn new(record: &UbatchRecord) -> Self {
        let mut values = BTreeMap::new();
        for (col, &pos) in record.pos.iter().enumerate() {
            for layer in 0..5 {
                for kind in KINDS {
                    let value = pos as f32 + layer as f32 / 100.0 + kind.code() as f32 / 1000.0;
                    values.insert((layer, kind, col), f32_bytes(&[value, value + 1.0]));
                }
            }
        }
        Self(values)
    }
}

impl UbatchColumns for SyntheticColumns {
    fn count(&self, _layer: u32, kind: TensorKind) -> u32 {
        u32::from(KINDS.contains(&kind))
    }

    fn column(
        &self,
        layer: u32,
        kind: TensorKind,
        index: u32,
        col: usize,
    ) -> Option<TensorColumn<'_>> {
        let bytes = self.0.get(&(layer, kind, col)).filter(|_| index == 0)?;
        Some(TensorColumn {
            ggml_type: GGML_F32,
            flags: 0,
            name: "blk.w.weight",
            bytes,
            digest: None,
        })
    }
}

struct SyntheticReplayer;

impl Replayer for SyntheticReplayer {
    fn replay(
        &self,
        seq: i32,
        layers: &[u32],
        _scope: skippy_receipts::recorder::ReplayScope<'_>,
        _seed: Option<&skippy_receipts::recorder::ReplaySeed>,
        steps: &[ReplayStep],
    ) -> Result<Vec<ReplayedPosition>> {
        let mut out = Vec::new();
        for step in steps {
            if let ReplayStep::Decode {
                record, capture, ..
            } = step
            {
                for col in 0..record.pos.len() {
                    let position = u32::try_from(record.pos[col])?;
                    if record.seq[col] == seq && capture.contains(&position) {
                        let columns = SyntheticColumns::new(record);
                        for &layer in layers {
                            let values = ReplayedLayer::from_columns(layer, &columns, col)?;
                            out.push((position, layer, values));
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn read_rows(
        &self,
        _layer: u32,
        _cells: &[skippy_receipts::recorder::PrefixCell],
    ) -> Result<Vec<skippy_receipts::recorder::StoredRows>> {
        anyhow::bail!("the synthetic stage keeps no rows")
    }

    fn pin_cells(&self, _cells: &[KvCell]) -> Result<()> {
        Ok(())
    }
}

fn stage(
    index: u32,
    layers: std::ops::Range<u32>,
    downstream: Option<Box<dyn Downstream>>,
) -> StageReceipts {
    let receipts = StageReceipts::new(
        index,
        layers,
        "test".into(),
        downstream,
        "model.gguf".into(),
        ed25519_dalek::SigningKey::from_bytes(&[index as u8 + 1; 32]),
    )
    .expect("stage receipts");
    receipts.set_replayer(Box::new(SyntheticReplayer));
    // the turn's positions, after an earlier turn's it attends to
    for positions in [0..POSITIONS.start, POSITIONS] {
        let (n, positions_end) = (positions.len(), positions.end);
        let record = UbatchRecord {
            token: positions.clone().map(|p| 100 + p as i32).collect(),
            seq: vec![SEQ; n],
            pos: positions.clone().map(|p| p as i32).collect(),
            output: vec![1; n],
            cell: positions.collect(),
            n_kv: 256,
            input: None,
        };
        let columns = SyntheticColumns::new(&record);
        let recorder = receipts.reports.recorder();
        recorder.record_ubatch(record, &columns);
        if positions_end < POSITIONS.end {
            recorder
                .commit("earlier", SEQ, 0..positions_end)
                .expect("earlier turn");
        }
    }
    receipts
}

/// Serves `receipts` on an ephemeral loopback port and returns its URL.
fn serve(receipts: StageReceipts) -> String {
    serve_routes(router(api_state(Arc::new(receipts))))
}

fn api_state(receipts: Arc<StageReceipts>) -> ApiState {
    ApiState {
        receipts,
        sessions: Arc::new(FixedSessions(BTreeMap::from([(SESSION.into(), SEQ)]))),
    }
}

fn serve_routes(routes: axum::Router) -> String {
    let listener =
        std::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("address"));
    spawn_api(listener, routes).expect("serve");
    url
}

/// A two-stage chain: stage 0 holds layers 0..2 in process, stage 1 holds
/// layers 2..4 behind its receipts API.
fn chain() -> (StageReceipts, String) {
    let downstream = serve(stage(1, 2..4, None));
    let http = HttpDownstream::new(&downstream, Duration::from_secs(30)).expect("client");
    (stage(0, 0..2, Some(Box::new(http))), downstream)
}

fn challenge(receipt_id: &str, layers: &[u32]) -> Challenge {
    Challenge {
        receipt_id: receipt_id.into(),
        picks: layers
            .iter()
            .map(|&layer| Pick {
                layer,
                positions: vec![4, 7, 9],
            })
            .collect(),
        proof: None,
        head: Vec::new(),
        attention: Vec::new(),
    }
}

#[test]
fn finish_turn_commits_every_stage() {
    let (stage0, _) = chain();
    let receipt = stage0
        .finish_turn("r1", SESSION, SEQ, turn(), None)
        .expect("receipt");

    assert_eq!(receipt.model, "model.gguf");
    assert_eq!(
        receipt.tokens,
        POSITIONS.map(|p| 100 + p as i32).collect::<Vec<_>>()
    );
    assert_eq!(receipt.sampled_tokens(), vec![107, 108, 109, 42]);
    assert_eq!(
        (receipt.position_start, receipt.position_end),
        (POSITIONS.start, POSITIONS.end)
    );
    let layers: Vec<_> = receipt
        .stages
        .iter()
        .map(|stage| (stage.stage_index, stage.layer_start, stage.layer_end))
        .collect();
    assert_eq!(layers, [(0, 0, 2), (1, 2, 4)]);
    assert_eq!(stage0.receipt("r1"), Some(Ok(receipt)));
}

#[test]
fn opens_every_stage_of_the_chain() {
    let (stage0, _) = chain();
    let receipt = stage0
        .finish_turn("r1", SESSION, SEQ, turn(), None)
        .expect("receipt");
    verify_shape(&receipt, 4).expect("shape");

    for layers in [&[1, 3][..], &[0], &[2, 3]] {
        let challenge = challenge("r1", layers);
        let opening = stage0.open(&challenge).expect("opening");
        let stages: Vec<u32> = opening.stages.iter().map(|s| s.stage_index).collect();
        assert_eq!(stages, [0, 1]);
        assert_eq!(opening.stages[0].anchors.len(), 3);
        verify_paths(&receipt, &challenge, &opening).expect("paths verify");
    }
}

#[test]
fn rejects_layers_and_receipts_nobody_holds() {
    let (stage0, _) = chain();
    stage0
        .finish_turn("r1", SESSION, SEQ, turn(), None)
        .expect("receipt");

    assert!(stage0.open(&challenge("r1", &[1, 9])).is_err());
    assert!(stage0.open(&challenge("missing", &[3])).is_err());
    assert!(stage0.receipt("missing").is_none());
}

#[test]
fn downstream_rejects_an_unknown_session() {
    let (_, downstream) = chain();
    let reply = reqwest::blocking::Client::new()
        .post(format!("{downstream}/receipts/commit"))
        .json(&CommitRequest {
            receipt_id: "r1".into(),
            session: "other".into(),
            position_start: POSITIONS.start,
            position_end: POSITIONS.end,
            challengeable: true,
        })
        .send()
        .expect("reply");

    assert_eq!(reply.status(), reqwest::StatusCode::BAD_REQUEST);
    assert!(reply.text().expect("body").contains("no native sequence"));
}

#[test]
fn refuses_a_remote_bind_unless_allowed() {
    let mut options = super::ReceiptsOptions {
        listen: Some(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 9501))),
        allow_remote: false,
        downstream: None,
        claimed_model: "model.gguf".into(),
        signing_key: None,
    };
    assert!(super::check_listen(&options).is_err());
    options.allow_remote = true;
    assert!(super::check_listen(&options).is_ok());
    options.allow_remote = false;
    options.listen = Some(SocketAddr::from((Ipv4Addr::LOCALHOST, 9501)));
    assert!(super::check_listen(&options).is_ok());
    options.listen = None;
    assert!(super::check_listen(&options).is_ok());
}

#[test]
fn a_turn_without_a_receipt_keeps_the_reason() {
    let (stage0, _) = chain();
    assert!(
        stage0
            .finish_turn("r2", SESSION, SEQ + 1, turn(), None)
            .is_err()
    );
    let reason = stage0.receipt("r2").expect("kept").expect_err("no receipt");
    assert!(reason.contains("recorded nothing"), "{reason}");
}

fn bound_request(client: &skippy_receipts::vrf::SecretKey) -> skippy_inference_api::ReceiptRequest {
    skippy_inference_api::ReceiptRequest {
        client_key: client.public(),
        parent: Some([7; 32]),
    }
}

#[test]
fn a_bound_receipt_is_signed_and_names_the_client() {
    let (stage0, _) = chain();
    let client = skippy_receipts::vrf::SecretKey::from_seed([4; 32]);
    let receipt = stage0
        .finish_turn("r1", SESSION, SEQ, turn(), Some(bound_request(&client)))
        .expect("receipt");
    let verified =
        skippy_receipts::turn::verify(&receipt, Some(&client.public())).expect("bound receipt");
    assert_eq!(
        verified.node_key,
        stage0.signing_key.verifying_key().to_bytes()
    );
    let binding = receipt.binding.expect("binding");
    assert_eq!(binding.parent, Some(hex::encode([7; 32])));
    assert!(binding.runtime.starts_with("skippy-abi "));
}

#[test]
fn a_bound_receipt_opens_one_challenge_with_its_proof() {
    let (stage0, _) = chain();
    let client = skippy_receipts::vrf::SecretKey::from_seed([4; 32]);
    let receipt = stage0
        .finish_turn("r1", SESSION, SEQ, turn(), Some(bound_request(&client)))
        .expect("receipt");
    let root = skippy_receipts::turn::verify(&receipt, None)
        .expect("bound")
        .turn_root;
    let mut challenge = challenge("r1", &[1, 3]);
    let err = stage0.open(&challenge).expect_err("no proof");
    assert!(format!("{err:#}").contains("VRF proof"), "{err:#}");

    // a proof of another input does not open it
    let wrong = client.prove(b"another root");
    challenge.proof = Some(hex::encode(wrong));
    assert!(stage0.open(&challenge).is_err());

    let proof = client.prove(&skippy_receipts::turn::challenge_input(&root));
    challenge.proof = Some(hex::encode(proof));
    let opening = stage0.open(&challenge).expect("opening");
    verify_paths(&receipt, &challenge, &opening).expect("paths verify");
    // stage 0 signs the opening it sends, for a bound receipt only
    let bytes = serde_json::to_vec(&opening).unwrap();
    let signature = stage0.sign_opening("r1", &bytes).unwrap().expect("signed");
    let bound = skippy_receipts::turn::verify(&receipt, None).unwrap();
    skippy_receipts::turn::verify_opening(&bound, &bytes, &signature).expect("signature");
    // the same challenge again is fine; another one is not
    stage0.open(&challenge).expect("repeat");
    let mut other = self::challenge("r1", &[0]);
    other.proof = challenge.proof.clone();
    let err = stage0.open(&other).expect_err("second challenge");
    assert!(format!("{err:#}").contains("already challenged"), "{err:#}");
}

// The attention rows of a bound receipt follow its signed opening, and only
// the client's VRF of that opening reads them.
#[test]
fn a_bound_receipt_opens_rows_only_for_the_clients_draw() {
    use skippy_receipts::attention::opening::BlockChallenge;
    use skippy_receipts::turn;
    let (stage0, _) = chain();
    let client = skippy_receipts::vrf::SecretKey::from_seed([4; 32]);
    let receipt = stage0
        .finish_turn("r1", SESSION, SEQ, turn(), Some(bound_request(&client)))
        .expect("receipt");
    let bound = turn::verify(&receipt, None).expect("bound");
    let mut rows = BlockChallenge {
        receipt_id: "r1".into(),
        picks: Vec::new(),
        proof: Some(hex::encode(client.prove(b"anything"))),
    };
    let err = stage0.rows(&rows).expect_err("not opened");
    assert!(format!("{err:#}").contains("was not opened"), "{err:#}");

    let mut challenge = challenge("r1", &[1, 3]);
    challenge.proof = Some(hex::encode(
        client.prove(&turn::challenge_input(&bound.turn_root)),
    ));
    let opening = stage0.open(&challenge).expect("opening");
    let bytes = serde_json::to_vec(&opening).unwrap();
    stage0.sign_opening("r1", &bytes).unwrap().expect("signed");

    // a proof of another input does not read them, nor does none
    assert!(stage0.rows(&rows).is_err());
    rows.proof = None;
    assert!(stage0.rows(&rows).is_err());
    rows.proof = Some(hex::encode(client.prove(&turn::rows_input(&bytes))));
    let opened = stage0.rows(&rows).expect("rows");
    assert_eq!(opened.receipt_id, "r1");
    let rows_bytes = serde_json::to_vec(&opened).unwrap();
    let signature = stage0
        .sign_rows("r1", &rows_bytes)
        .unwrap()
        .expect("signed");
    turn::verify_rows(&bound, &rows_bytes, &signature).expect("signature");
    // the rows' signature is not an opening's
    assert!(turn::verify_opening(&bound, &rows_bytes, &signature).is_err());
}

#[test]
fn only_a_turn_that_takes_the_best_logit_is_greedy() {
    use skippy_runtime::{LogitBias, SamplingConfig};
    let disabled = SamplingConfig::default();
    assert!(super::greedy_sampling(&disabled, None));
    let mut zero = SamplingConfig {
        enabled: true,
        temperature: 0.0,
        ..SamplingConfig::default()
    };
    assert!(super::greedy_sampling(&zero, Some(r#"{"grammar":""}"#)));
    assert!(!super::greedy_sampling(
        &zero,
        Some(r#"{"grammar":"root ::= \"a\""}"#)
    ));
    zero.logit_bias.push(LogitBias {
        token_id: 5,
        bias: 1.0,
    });
    assert!(!super::greedy_sampling(&zero, None));
    let warm = SamplingConfig {
        enabled: true,
        temperature: 0.7,
        ..SamplingConfig::default()
    };
    assert!(!super::greedy_sampling(&warm, None));
}

#[test]
fn a_sampled_turn_states_its_parameters_and_seed() {
    use skippy_receipts::turn;
    use skippy_runtime::SamplingConfig;
    assert_eq!(
        super::sampler_statement(&SamplingConfig::default(), None),
        turn::GREEDY
    );
    let warm = SamplingConfig {
        enabled: true,
        temperature: 0.7,
        seed: 1234,
        ..SamplingConfig::default()
    };
    let grammar = Some(r#"{"grammar":"root ::= \"a\""}"#);
    let statement = super::sampler_statement(&warm, grammar);
    let parameters = turn::sampler_parameters(&statement).expect("parameters");
    assert_eq!(parameters["seed"], 1234);
    assert!((parameters["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);
    assert_eq!(parameters["grammar"].as_str().map(str::len), Some(64));
    let other = SamplingConfig { seed: 99, ..warm };
    assert_ne!(super::sampler_statement(&other, grammar), statement);
}

// The OpenAI surface opens only a bound receipt, for its client's challenge.
#[test]
fn the_client_route_opens_only_bound_receipts() {
    // each stage of a fixture chain records one turn
    let (unbound, _) = chain();
    unbound
        .finish_turn("unbound", SESSION, SEQ, turn(), None)
        .expect("receipt");
    let (stage0, _) = chain();
    let client = skippy_receipts::vrf::SecretKey::from_seed([4; 32]);
    let receipt = stage0
        .finish_turn("bound", SESSION, SEQ, turn(), Some(bound_request(&client)))
        .expect("receipt");
    let unbound_url = serve_routes(client_routes(api_state(Arc::new(unbound))));
    let url = serve_routes(client_routes(api_state(Arc::new(stage0))));
    let open_at = |url: &str, challenge: &Challenge| {
        let mut body = serde_json::to_value(challenge).unwrap();
        body["model"] = "model".into();
        reqwest::blocking::Client::new()
            .post(format!("{url}/v1/receipts/open"))
            .json(&body)
            .send()
            .expect("reply")
    };

    let open = |challenge: &Challenge| open_at(&url, challenge);
    let reply = open_at(&unbound_url, &challenge("unbound", &[1]));
    assert_eq!(reply.status(), reqwest::StatusCode::FORBIDDEN);
    assert_eq!(
        open(&challenge("missing", &[1])).status(),
        reqwest::StatusCode::NOT_FOUND
    );

    let root = skippy_receipts::turn::verify(&receipt, None)
        .expect("bound")
        .turn_root;
    let mut bound = challenge("bound", &[1, 3]);
    assert_eq!(open(&bound).status(), reqwest::StatusCode::BAD_REQUEST);
    let proof = client.prove(&skippy_receipts::turn::challenge_input(&root));
    bound.proof = Some(hex::encode(proof));
    let reply = open(&bound);
    assert!(reply.status().is_success(), "{}", reply.status());
    let signature = reply
        .headers()
        .get(skippy_receipts::turn::OPENING_SIGNATURE_HEADER)
        .expect("signed")
        .to_str()
        .unwrap()
        .to_owned();
    let bytes = reply.bytes().expect("body");
    let verified = skippy_receipts::turn::verify(&receipt, None).unwrap();
    skippy_receipts::turn::verify_opening(&verified, &bytes, &signature).expect("signature");
}

/// Serves `receipts` the way a stage's binary listener does, on an
/// ephemeral loopback port; returns the endpoint and the count of accepted
/// connections.
fn serve_binary(receipts: StageReceipts) -> (String, Arc<AtomicUsize>) {
    let listener =
        std::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("bind");
    let endpoint = format!("tcp://{}", listener.local_addr().expect("address"));
    let receipts = Arc::new(receipts);
    let accepted = Arc::new(AtomicUsize::new(0));
    let count = accepted.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            count.fetch_add(1, Ordering::SeqCst);
            let receipts = receipts.clone();
            std::thread::spawn(move || {
                skippy_protocol::binary::send_ready(&mut stream).expect("ready");
                let first =
                    skippy_protocol::binary::read_stage_message(&mut stream, 0).expect("request");
                let sessions = FixedSessions(BTreeMap::from([(SESSION.into(), SEQ)]));
                super::serve_connection(Some(receipts), &sessions, stream, first).expect("serve");
            });
        }
    });
    (endpoint, accepted)
}

fn stage_config(downstream: &str) -> skippy_protocol::StageConfig {
    skippy_protocol::StageConfig {
        bind_addr: "127.0.0.1:0".into(),
        downstream: Some(skippy_protocol::PeerConfig {
            stage_id: "stage-1".into(),
            stage_index: 1,
            endpoint: downstream.into(),
            persistent_lane_marker: false,
        }),
        ..Default::default()
    }
}

#[test]
fn the_stage_connection_commits_and_opens_the_next_stage() {
    let (endpoint, accepted) = serve_binary(stage(1, 2..4, None));
    let next = StageConnection::new(&stage_config(&endpoint), Duration::from_secs(30));
    let stage0 = stage(0, 0..2, Some(Box::new(next)));
    let receipt = stage0
        .finish_turn("r1", SESSION, SEQ, turn(), None)
        .expect("receipt");
    let challenge = challenge("r1", &[1, 3]);
    let opening = stage0.open(&challenge).expect("opening");

    let stages: Vec<u32> = opening.stages.iter().map(|s| s.stage_index).collect();
    assert_eq!(stages, [0, 1]);
    verify_paths(&receipt, &challenge, &opening).expect("paths verify");
    assert_eq!(accepted.load(Ordering::SeqCst), 1, "one kept connection");
}

#[test]
fn the_stage_connection_carries_the_next_stage_error() {
    let (endpoint, _) = serve_binary(stage(1, 2..4, None));
    let next = StageConnection::new(&stage_config(&endpoint), Duration::from_secs(30));
    let error = next
        .commit(&CommitRequest {
            receipt_id: "r1".into(),
            session: "other".into(),
            position_start: POSITIONS.start,
            position_end: POSITIONS.end,
            challengeable: true,
        })
        .expect_err("unknown session");
    assert!(
        format!("{error:#}").contains("no native sequence"),
        "{error:#}"
    );
    // the connection survives a refused request
    assert!(next.open(&challenge("missing", &[3])).is_err());
}

#[test]
fn a_turn_ends_at_the_last_token_it_fed() {
    // the last verify window computed drafts at 10 and 11, rejected when
    // the turn stopped
    assert_eq!(super::turn_positions(4..12, &turn()), 4..10);
    assert_eq!(super::turn_positions(POSITIONS, &turn()), POSITIONS);
    let unsampled = TurnTokens {
        sampled: 0,
        ..turn()
    };
    assert_eq!(super::turn_positions(4..12, &unsampled), 4..12);
}

// Only a receipt a client can challenge keeps its cells out of later
// requests' way.
#[test]
fn only_a_bound_turn_pins_its_cells() {
    let pinned = |stage: &StageReceipts| {
        let recorder = stage.reports.recorder();
        // the fixture's earlier turn
        recorder.release_pin("earlier");
        recorder.pinned_cells(std::time::Instant::now())
    };
    let (unbound, _) = chain();
    unbound
        .finish_turn("unbound", SESSION, SEQ, turn(), None)
        .expect("receipt");
    assert!(pinned(&unbound).is_empty());
    let (bound, _) = chain();
    let client = skippy_receipts::vrf::SecretKey::from_seed([4; 32]);
    bound
        .finish_turn("bound", SESSION, SEQ, turn(), Some(bound_request(&client)))
        .expect("receipt");
    assert!(!pinned(&bound).is_empty());
}
