//! Fixtures for the receipts tests, and the tests of a stage's turn
//! lifecycle. The routes, the stage connection and the downstream client
//! have their tests beside them.

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use skippy_protocol::binary::ReceiptsReply;
use skippy_receipts::binding::{verify_paths, verify_shape};
use skippy_receipts::layer_tensors::{
    GGML_F32, TensorColumn, TensorKind, UbatchColumns, f32_bytes,
};
use skippy_receipts::recorder::{
    CHALLENGE_PIN, KvCell, MemoryOp, PrefixCell, Recorder, ReplayScope, ReplaySeed, ReplayStep,
    ReplayedLayer, ReplayedPosition, Replayer, StoredRows, TurnId, UbatchRecord,
};
use skippy_receipts::wire::{CannotOpen, CannotOpenReason, Challenge, Pick, Receipt};

use super::downstream::{ChainRequest, CommitRequest, Downstream, HttpDownstream};
use super::http_api::{ApiState, router, spawn_api};
use super::kept_turns::KeptTurns;
use super::refusal::NotFound;
use super::reply_limits::ReplyLimits;
use super::{SessionTurn, SessionTurns, StageReceipts, StageReplayer, TurnTokens, lock};

pub(super) const SESSION: &str = "session-a";
pub(super) const SEQ: i32 = 3;
pub(super) const POSITIONS: std::ops::Range<u32> = 4..10;
/// A bound on every wait for something another thread does.
pub(super) const GENEROUS: Duration = Duration::from_secs(60);

pub(super) fn turn() -> TurnTokens {
    TurnTokens {
        generation_start: 7,
        final_token: Some(42),
        sampled: 4,
        sampler: skippy_receipts::turn::GREEDY.to_owned(),
    }
}

pub(super) struct FixedSessions(BTreeMap<String, i32>);

impl SessionTurns for FixedSessions {
    fn turn(&self, session: &str) -> Option<SessionTurn> {
        self.0.get(session).map(|&seq| SessionTurn {
            seq,
            turn: TurnId::default(),
        })
    }
}

/// The sessions every fixture stage serves. The routes hold them weakly, as
/// they hold a stage's runtime.
pub(super) fn sessions() -> Arc<FixedSessions> {
    static SESSIONS: LazyLock<Arc<FixedSessions>> =
        LazyLock::new(|| Arc::new(FixedSessions(BTreeMap::from([(SESSION.into(), SEQ)]))));
    SESSIONS.clone()
}

/// The turn the fixture's session computes.
pub(super) fn session_turn() -> SessionTurn {
    sessions().turn(SESSION).expect("the fixture session")
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

/// What a fixture stage's replayer was asked to do.
#[derive(Default)]
pub(super) struct Probe {
    /// Replays run.
    pub replays: AtomicUsize,
    /// The number of cells of every pin, in order.
    pub pins: Mutex<Vec<usize>>,
    /// The cells pinned when each replay ran.
    pub pinned_in_replays: Mutex<Vec<usize>>,
}

impl Probe {
    pub(super) fn replays(&self) -> usize {
        self.replays.load(Ordering::SeqCst)
    }

    /// The cells pinned now.
    pub(super) fn pinned(&self) -> usize {
        lock(&self.pins).last().copied().unwrap_or(0)
    }
}

/// Replays by recomputing the synthetic values. Like the native replayer,
/// it starts a replay only from cells the recorder still knows.
struct SyntheticReplayer {
    probe: Arc<Probe>,
    recorder: Arc<Recorder>,
}

impl Replayer for SyntheticReplayer {
    fn replay(
        &self,
        seq: i32,
        layers: &[u32],
        _scope: ReplayScope<'_>,
        seed: Option<&ReplaySeed>,
        steps: &[ReplayStep],
    ) -> Result<Vec<ReplayedPosition>> {
        self.probe.replays.fetch_add(1, Ordering::SeqCst);
        let pinned = self.probe.pinned();
        lock(&self.probe.pinned_in_replays).push(pinned);
        if let Some(seed) = seed {
            self.recorder.check_seed(seed)?;
        }
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

    /// The synthetic layers store no rows.
    fn read_rows(&self, _layer: u32, _cells: &[PrefixCell]) -> Result<Vec<StoredRows>> {
        Ok(Vec::new())
    }

    fn pin_cells(&self, cells: &[KvCell]) -> Result<()> {
        lock(&self.probe.pins).push(cells.len());
        Ok(())
    }
}

impl StageReplayer for SyntheticReplayer {}

/// Records `positions` of the fixture's sequence on `receipts`.
pub(super) fn record(receipts: &StageReceipts, positions: std::ops::Range<u32>) {
    let n = positions.len();
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
    receipts.reports.recorder().record_ubatch(record, &columns);
}

/// A stage that recorded the fixture's turn at [`POSITIONS`], after an
/// earlier turn it attends to, and the probe of its replayer.
pub(super) fn probed_stage(
    index: u32,
    layers: std::ops::Range<u32>,
    downstream: Option<Box<dyn Downstream>>,
) -> (StageReceipts, Arc<Probe>) {
    let receipts = StageReceipts::new(
        index,
        layers,
        "test".into(),
        downstream,
        "model.gguf".into(),
        ed25519_dalek::SigningKey::from_bytes(&[index as u8 + 1; 32]),
        ReplyLimits::for_width(Some(2)),
    );
    let probe = Arc::new(Probe::default());
    receipts.set_replayer(Box::new(SyntheticReplayer {
        probe: probe.clone(),
        recorder: receipts.reports.shared_recorder(),
    }));
    record(&receipts, 0..POSITIONS.start);
    receipts
        .reports
        .recorder()
        .commit("earlier", SEQ, TurnId::default(), 0..POSITIONS.start)
        .expect("earlier turn");
    record(&receipts, POSITIONS);
    (receipts, probe)
}

pub(super) fn stage(
    index: u32,
    layers: std::ops::Range<u32>,
    downstream: Option<Box<dyn Downstream>>,
) -> StageReceipts {
    probed_stage(index, layers, downstream).0
}

pub(super) fn api_state(receipts: &Arc<StageReceipts>) -> ApiState {
    let sessions: Arc<dyn SessionTurns> = sessions();
    ApiState {
        receipts: Arc::downgrade(receipts),
        sessions: Arc::downgrade(&sessions),
    }
}

/// Serves `routes` on an ephemeral loopback port for the rest of the test
/// run, and returns its URL.
pub(super) fn serve_routes(routes: axum::Router) -> String {
    let listener =
        std::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("address"));
    std::mem::forget(spawn_api(listener, routes).expect("serve"));
    url
}

/// Serves the receipts API of `receipts`, which the test run keeps.
pub(super) fn serve(receipts: StageReceipts) -> String {
    let receipts = Arc::new(receipts);
    let url = serve_routes(router(api_state(&receipts)));
    std::mem::forget(receipts);
    url
}

/// A two-stage chain: stage 0 holds layers 0..2 in process, stage 1 holds
/// layers 2..4 behind its receipts API.
pub(super) struct Chain {
    pub stage0: StageReceipts,
    pub probe0: Arc<Probe>,
    pub probe1: Arc<Probe>,
    pub downstream: String,
}

pub(super) fn chain() -> Chain {
    let (stage1, probe1) = probed_stage(1, 2..4, None);
    let downstream = serve(stage1);
    let http = HttpDownstream::new(&downstream, GENEROUS, GENEROUS).expect("client");
    let (stage0, probe0) = probed_stage(0, 0..2, Some(Box::new(http)));
    Chain {
        stage0,
        probe0,
        probe1,
        downstream,
    }
}

pub(super) fn challenge(receipt_id: &str, layers: &[u32]) -> Challenge {
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

pub(super) fn client() -> skippy_receipts::vrf::SecretKey {
    skippy_receipts::vrf::SecretKey::from_seed([4; 32])
}

pub(super) fn bound_request(
    client: &skippy_receipts::vrf::SecretKey,
) -> skippy_inference_api::ReceiptRequest {
    skippy_inference_api::ReceiptRequest {
        client_key: client.public(),
        parent: Some([7; 32]),
        seed: None,
        id: None,
    }
}

/// Commits the fixture's turn as a receipt bound to [`client`], and
/// returns it with a challenge of `layers` that carries the client's proof.
pub(super) fn bound_turn(stage0: &StageReceipts, id: &str, layers: &[u32]) -> (Receipt, Challenge) {
    let client = client();
    let receipt = stage0
        .finish_turn(
            id,
            SESSION,
            session_turn(),
            turn(),
            Some(bound_request(&client)),
        )
        .expect("receipt");
    let root = skippy_receipts::turn::verify(&receipt, None)
        .expect("bound")
        .turn_root;
    let mut challenge = challenge(id, layers);
    let proof = client.prove(&skippy_receipts::turn::challenge_input(&root));
    challenge.proof = Some(hex::encode(proof));
    (receipt, challenge)
}

/// The [`CannotOpen`] a request failed with.
pub(super) fn cannot_open(error: &anyhow::Error) -> &CannotOpen {
    error
        .downcast_ref::<CannotOpen>()
        .unwrap_or_else(|| panic!("not a cannot-open answer: {error:#}"))
}

/// Serves the binary stage protocol on an ephemeral loopback port: every
/// connection is told the stage is ready, and each receipts request is
/// answered with what `answer` returns; `None` closes the connection, as a
/// stage that predates receipts does. Returns the endpoint and the count of
/// accepted connections.
pub(super) fn fake_stage(
    answer: impl Fn(&ChainRequest) -> Option<ReceiptsReply> + Send + Sync + 'static,
) -> (String, Arc<AtomicUsize>) {
    let listener =
        std::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("bind");
    let endpoint = format!("tcp://{}", listener.local_addr().expect("address"));
    let accepted = Arc::new(AtomicUsize::new(0));
    let count = accepted.clone();
    let answer = Arc::new(answer);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            count.fetch_add(1, Ordering::SeqCst);
            let answer = answer.clone();
            std::thread::spawn(move || {
                use skippy_protocol::binary::{
                    read_stage_message, send_ready, send_receipts_reply,
                };
                if send_ready(&mut stream).is_err() {
                    return;
                }
                while let Ok(message) = read_stage_message(&mut stream, 0) {
                    let request = serde_json::from_slice(&message.raw_bytes).expect("a request");
                    let Some(reply) = answer(&request) else {
                        return;
                    };
                    if send_receipts_reply(&mut stream, &reply).is_err() {
                        return;
                    }
                }
            });
        }
    });
    (endpoint, accepted)
}

pub(super) fn stage_config(downstream: &str) -> skippy_protocol::StageConfig {
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
fn finish_turn_commits_every_stage() {
    let Chain { stage0, .. } = chain();
    let receipt = stage0
        .finish_turn("r1", SESSION, session_turn(), turn(), None)
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
    assert_eq!(stage0.receipt("r1"), Some(Ok(Arc::new(receipt))));
}

#[test]
fn opens_every_stage_of_the_chain() {
    let Chain { stage0, .. } = chain();
    let receipt = stage0
        .finish_turn("r1", SESSION, session_turn(), turn(), None)
        .expect("receipt");
    verify_shape(&receipt, 4).expect("shape");

    for layers in [&[1, 3][..], &[0], &[2, 3]] {
        let challenge = challenge("r1", layers);
        let opening = stage0.open(&challenge, None).expect("opening");
        let stages: Vec<u32> = opening.stages.iter().map(|s| s.stage_index).collect();
        assert_eq!(stages, [0, 1]);
        assert_eq!(opening.stages[0].anchors.len(), 3);
        verify_paths(&receipt, &challenge, &opening).expect("paths verify");
    }
}

#[test]
fn rejects_layers_and_receipts_nobody_holds() {
    let Chain { stage0, .. } = chain();
    stage0
        .finish_turn("r1", SESSION, session_turn(), turn(), None)
        .expect("receipt");

    assert!(stage0.open(&challenge("r1", &[1, 9]), None).is_err());
    let error = stage0.open(&challenge("missing", &[3]), None).unwrap_err();
    assert!(error.downcast_ref::<NotFound>().is_some(), "{error:#}");
    assert!(stage0.receipt("missing").is_none());
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

// The node's key seed is a secret: nothing that logs the options shows it.
#[test]
fn the_signing_key_is_not_written_out() {
    let options = super::ReceiptsOptions::embedded("model.gguf".into(), Some([0xAB; 32]));
    let debug = format!("{options:?} {options:#?}");
    assert!(debug.contains("<redacted>") && debug.contains("model.gguf"));
    assert!(!debug.contains("171") && !debug.to_lowercase().contains("ab, ab"));
    let unset = super::ReceiptsOptions::embedded("model.gguf".into(), None);
    assert!(format!("{unset:?}").contains("signing_key: None"));
}

#[test]
fn a_turn_without_a_receipt_keeps_the_reason() {
    let Chain { stage0, .. } = chain();
    let other = SessionTurn {
        seq: SEQ + 1,
        turn: TurnId::default(),
    };
    assert!(
        stage0
            .finish_turn("r2", SESSION, other, turn(), None)
            .is_err()
    );
    let reason = stage0.receipt("r2").expect("kept").expect_err("no receipt");
    assert!(reason.contains("recorded nothing"), "{reason}");
}

// A commit names its turn: a session that names a turn the sequence does
// not record gets no receipt, not the leaves of the turn that is there.
#[test]
fn a_commit_of_another_turn_gets_no_receipt() {
    let stage0 = stage(0, 0..4, None);
    let stale = SessionTurn {
        seq: SEQ,
        turn: TurnId(99),
    };
    let error = stage0
        .finish_turn("r1", SESSION, stale, turn(), None)
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("recorded nothing"),
        "{error:#}"
    );
    // the turn that is there still commits
    stage0
        .finish_turn("r2", SESSION, session_turn(), turn(), None)
        .expect("receipt");
}

#[test]
fn a_bound_receipt_is_signed_and_names_the_client() {
    let Chain { stage0, .. } = chain();
    let client = client();
    let receipt = stage0
        .finish_turn(
            "r1",
            SESSION,
            session_turn(),
            turn(),
            Some(bound_request(&client)),
        )
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
    let Chain { stage0, .. } = chain();
    let client = client();
    let (receipt, challenge) = bound_turn(&stage0, "r1", &[1, 3]);
    let mut unproven = challenge.clone();
    unproven.proof = None;
    let err = stage0.open(&unproven, None).expect_err("no proof");
    assert!(format!("{err:#}").contains("VRF proof"), "{err:#}");

    // a proof of another input does not open it
    let mut wrong = challenge.clone();
    wrong.proof = Some(hex::encode(client.prove(b"another root")));
    assert!(stage0.open(&wrong, None).is_err());

    let opening = stage0.open(&challenge, None).expect("opening");
    verify_paths(&receipt, &challenge, &opening).expect("paths verify");
    // stage 0 signs the opening it sends, for a bound receipt only
    let bytes = serde_json::to_vec(&opening).unwrap();
    let signature = stage0
        .sign_opening(&challenge, &bytes)
        .unwrap()
        .expect("signed");
    let bound = skippy_receipts::turn::verify(&receipt, None).unwrap();
    skippy_receipts::turn::verify_opening(&bound, &challenge, &bytes, &signature)
        .expect("signature");
    // the same challenge again is fine; another one is not
    stage0.open(&challenge, None).expect("repeat");
    let mut other = self::challenge("r1", &[0]);
    other.proof = challenge.proof.clone();
    let err = stage0.open(&other, None).expect_err("second challenge");
    assert!(format!("{err:#}").contains("already challenged"), "{err:#}");
}

// The attention rows of a bound receipt follow its signed opening, and only
// the client's VRF of that opening reads them.
#[test]
fn a_bound_receipt_opens_rows_only_for_the_clients_draw() {
    use skippy_receipts::attention::opening::BlockChallenge;
    use skippy_receipts::turn;
    let Chain { stage0, .. } = chain();
    let client = client();
    let (receipt, challenge) = bound_turn(&stage0, "r1", &[1, 3]);
    let bound = turn::verify(&receipt, None).expect("bound");
    let mut rows = BlockChallenge {
        receipt_id: "r1".into(),
        picks: Vec::new(),
        proof: Some(hex::encode(client.prove(b"anything"))),
    };
    let err = stage0.rows(&rows, None).expect_err("not opened");
    assert!(format!("{err:#}").contains("was not opened"), "{err:#}");

    let opening = stage0.open(&challenge, None).expect("opening");
    // opened, but stage 0 has not signed an opening yet
    assert!(stage0.rows(&rows, None).is_err());
    let bytes = serde_json::to_vec(&opening).unwrap();
    stage0
        .sign_opening(&challenge, &bytes)
        .unwrap()
        .expect("signed");

    // a proof of another input does not read them, nor does none
    assert!(stage0.rows(&rows, None).is_err());
    rows.proof = None;
    assert!(stage0.rows(&rows, None).is_err());
    rows.proof = Some(hex::encode(client.prove(&turn::rows_input(&bytes))));
    let opened = stage0.rows(&rows, None).expect("rows");
    assert_eq!(opened.receipt_id, "r1");
    let rows_bytes = serde_json::to_vec(&opened).unwrap();
    let signature = stage0
        .sign_rows(&rows, &rows_bytes)
        .unwrap()
        .expect("signed");
    turn::verify_rows(&bound, &rows, &rows_bytes, &signature).expect("signature");
    // the rows' signature is not an opening's
    assert!(turn::verify_opening(&bound, &challenge, &rows_bytes, &signature).is_err());

    // rows of a position the challenge did not pick for attention are refused
    let mut unasked = rows.clone();
    unasked.picks = vec![skippy_receipts::attention::opening::BlockPick {
        position: 4,
        layer: 1,
        blocks: vec![0],
    }];
    let err = stage0.rows(&unasked, None).expect_err("blocks nobody drew");
    assert!(format!("{err:#}").contains("attention pick"), "{err:#}");
}

#[test]
fn a_turn_ends_at_the_last_token_it_fed() {
    // the last verify window computed drafts at 10 and 11, rejected when
    // the turn stopped
    assert_eq!(super::turn_positions(4..12, &turn()).unwrap(), 4..10);
    assert_eq!(
        super::turn_positions(POSITIONS, &turn()).unwrap(),
        POSITIONS
    );
    let unsampled = TurnTokens {
        sampled: 0,
        ..turn()
    };
    assert_eq!(super::turn_positions(4..12, &unsampled).unwrap(), 4..12);
}

// A ubatch the recorder could not receipt, such as one of image embeddings,
// computed the turn's last positions: the receipt would stop short of the
// position its last token was sampled from, so there is none.
#[test]
fn a_turn_that_did_not_record_its_last_fed_position_gets_no_receipt() {
    let error = super::turn_positions(4..8, &turn()).unwrap_err();
    assert!(format!("{error:#}").contains("up to 10"), "{error:#}");
}

// Only a receipt a client can challenge keeps its cells out of later
// requests' way.
#[test]
fn only_a_bound_turn_pins_its_cells() {
    let pinned = |stage: &StageReceipts| {
        let recorder = stage.reports.recorder();
        // the fixture's earlier turn
        recorder.release_pin("earlier");
        recorder.pinned_cells(Instant::now())
    };
    let unbound = chain().stage0;
    unbound
        .finish_turn("unbound", SESSION, session_turn(), turn(), None)
        .expect("receipt");
    assert!(pinned(&unbound).is_empty());
    let bound = chain().stage0;
    bound_turn(&bound, "bound", &[1]);
    assert!(!pinned(&bound).is_empty());
}

// A turn's cells are pinned natively no longer than its pin holds: the
// stage pins anew when the pin runs out, not only at its next commit.
#[test]
fn pins_are_given_up_when_they_run_out() {
    let (stage0, probe) = probed_stage(0, 0..4, None);
    stage0.reports.recorder().release_pin("earlier");
    bound_turn(&stage0, "bound", &[1]);
    assert!(probe.pinned() > 0);
    let now = Instant::now();
    let due = stage0.run_upkeep(now).expect("a pin to give up");
    assert!(probe.pinned() > 0 && due > now && due <= now + CHALLENGE_PIN);
    assert_eq!(stage0.run_upkeep(due), None, "nothing is pinned any more");
    assert_eq!(probe.pinned(), 0);
}

// A challenge pins its turn's cells while it waits and is opened, also
// when the turn has no pin of its own: the opening reads the cells after
// the replay, and a later request must not write them meanwhile.
#[test]
fn an_opening_pins_its_turn_while_it_replays() {
    let (stage0, probe) = probed_stage(0, 0..4, None);
    stage0.reports.recorder().release_pin("earlier");
    stage0
        .finish_turn("r1", SESSION, session_turn(), turn(), None)
        .expect("receipt");
    // a turn bound to no key has no pin of its own
    assert_eq!(probe.pinned(), 0);
    stage0.open(&challenge("r1", &[1]), None).expect("opening");
    let in_replays = lock(&probe.pinned_in_replays).clone();
    assert!(
        !in_replays.is_empty() && in_replays.iter().all(|&cells| cells > 0),
        "{in_replays:?}"
    );
    // an opening without attention picks leaves no rows to read
    assert_eq!(probe.pinned(), 0);

    // a challenge the stage refuses pins nothing past its refusal
    assert!(stage0.open(&challenge("r1", &[9]), None).is_err());
    assert_eq!(probe.pinned(), 0);
}

// A turn whose next stage did not commit has no receipt, so stage 0 keeps
// neither the turn nor a pin on its cells.
#[test]
fn a_failed_downstream_commit_leaves_nothing_pinned() {
    let (endpoint, _) = fake_stage(|_| Some(ReceiptsReply::Refused("no turn".into())));
    let next =
        super::downstream::StageConnection::new(&stage_config(&endpoint), GENEROUS, GENEROUS);
    let (stage0, probe) = probed_stage(0, 0..2, Some(Box::new(next)));
    stage0.reports.recorder().release_pin("earlier");
    let request = Some(bound_request(&client()));
    let error = stage0
        .finish_turn("r1", SESSION, session_turn(), turn(), request)
        .unwrap_err();
    assert!(format!("{error:#}").contains("no turn"), "{error:#}");
    assert!(!stage0.reports.recorder().holds_turn("r1"));
    assert_eq!(probe.pinned(), 0);
    assert!(stage0.receipt("r1").expect("kept").is_err());
}

// A challenge no draw makes is refused before the stage replays anything,
// for a bound receipt and for one bound to no key.
#[test]
fn an_oversized_challenge_is_refused_before_any_replay() {
    let Chain {
        stage0,
        probe0,
        probe1,
        ..
    } = chain();
    stage0
        .finish_turn("unbound", SESSION, session_turn(), turn(), None)
        .expect("receipt");
    let every_position = |id: &str, proof: Option<String>| Challenge {
        receipt_id: id.into(),
        picks: (0..4)
            .map(|layer| Pick {
                layer,
                positions: POSITIONS.collect(),
            })
            .collect(),
        proof,
        head: Vec::new(),
        attention: Vec::new(),
    };
    let error = stage0
        .open(&every_position("unbound", None), None)
        .unwrap_err();
    assert!(format!("{error:#}").contains("a draw picks"), "{error:#}");

    let bound = chain();
    let (_, drawn) = bound_turn(&bound.stage0, "bound", &[1, 3]);
    let oversized = every_position("bound", drawn.proof.clone());
    let error = bound.stage0.open(&oversized, None).unwrap_err();
    assert!(format!("{error:#}").contains("a draw picks"), "{error:#}");
    // the refused challenge was not the one challenge the receipt gets
    bound
        .stage0
        .open(&drawn, None)
        .expect("the drawn challenge");

    assert_eq!(probe0.replays() + probe1.replays(), 0);
    assert!(
        bound.probe0.replays() > 0,
        "only the drawn one was replayed"
    );
}

// A receipt id the stage keeps no record of is refused as not found, also
// when the recorder still holds the turn: it is never opened without the
// proof its record would have asked for.
#[test]
fn an_evicted_receipt_is_not_opened() {
    let (stage0, probe) = probed_stage(0, 0..4, None);
    let (_, challenge) = bound_turn(&stage0, "r1", &[1]);
    // the record is dropped, as when its window has passed
    *lock(&stage0.kept) = KeptTurns::default();
    assert!(stage0.reports.recorder().holds_turn("r1"));
    let mut unproven = challenge.clone();
    unproven.proof = None;
    for challenge in [&challenge, &unproven] {
        let error = stage0.open(challenge, None).unwrap_err();
        assert!(error.downcast_ref::<NotFound>().is_some(), "{error:#}");
    }
    let rows = skippy_receipts::attention::opening::BlockChallenge {
        receipt_id: "r1".into(),
        picks: Vec::new(),
        proof: None,
    };
    let error = stage0.rows(&rows, None).unwrap_err();
    assert!(error.downcast_ref::<NotFound>().is_some(), "{error:#}");
    assert_eq!(probe.replays(), 0);
    assert_eq!(stage0.sign_opening(&challenge, b"{}").unwrap(), None);
}

// A request id that comes back while its receipt can still be challenged
// gets no receipt, whoever asks: another client cannot shadow the receipt,
// and the client it was signed for can still open it.
#[test]
fn a_reused_request_id_keeps_the_receipt_that_can_still_be_challenged() {
    let stage0 = stage(0, 0..4, None);
    let (first, challenge) = bound_turn(&stage0, "r1", &[1]);
    stage0
        .open(&challenge, None)
        .expect("the first turn's challenge");

    // another client asks for a turn under the same id
    record(&stage0, POSITIONS.end..POSITIONS.end + 2);
    let intruder = skippy_receipts::vrf::SecretKey::from_seed([9; 32]);
    let error = stage0
        .finish_turn(
            "r1",
            SESSION,
            session_turn(),
            turn(),
            Some(bound_request(&intruder)),
        )
        .unwrap_err();
    assert!(format!("{error:#}").contains("another client"), "{error:#}");
    let kept = stage0.receipt("r1").expect("a record").expect("a receipt");
    assert_eq!(kept.stages, first.stages);
    stage0
        .open(&challenge, None)
        .expect("still the first turn's");

    // the first client's next turn under the id, inside the window: the
    // receipt it was signed stays the one the id opens
    let tokens = TurnTokens {
        generation_start: POSITIONS.end + 1,
        sampled: 2,
        ..turn()
    };
    let error = stage0
        .finish_turn(
            "r1",
            SESSION,
            session_turn(),
            tokens,
            Some(bound_request(&client())),
        )
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("signed a receipt"),
        "{error:#}"
    );
    // a lookup by id returns a bound receipt without the turn's tokens
    let kept = Receipt {
        tokens: Vec::new(),
        final_token: None,
        ..first
    };
    assert_eq!(stage0.receipt("r1"), Some(Ok(Arc::new(kept))));
    stage0
        .open(&challenge, None)
        .expect("still the first turn's");
}

// A stage after the first opens a bound turn only with the client's proof,
// and what it refuses costs the turn none of its replays.
#[test]
fn a_later_stage_asks_for_the_clients_proof() {
    let (stage1, probe) = probed_stage(1, 2..4, None);
    let client = client();
    let commit = CommitRequest {
        receipt_id: "r1".into(),
        session: SESSION.into(),
        position_start: POSITIONS.start,
        position_end: POSITIONS.end,
        client_key: Some(hex::encode(client.public())),
    };
    stage1
        .commit_session(sessions().as_ref(), &commit)
        .expect("commit");
    let root = [5; 32];
    let proof = hex::encode(client.prove(&skippy_receipts::turn::challenge_input(&root)));
    let forwarded = |challenge: &Challenge, root: Option<[u8; 32]>| super::ForwardedOpen {
        challenge: challenge.clone(),
        turn_root: root.map(hex::encode),
    };
    let unproven = challenge("r1", &[3]);
    let mut proven = unproven.clone();
    proven.proof = Some(proof);
    let mut forged = unproven.clone();
    let stranger = skippy_receipts::vrf::SecretKey::from_seed([9; 32]);
    let stolen = stranger.prove(&skippy_receipts::turn::challenge_input(&root));
    forged.proof = Some(hex::encode(stolen));
    // more refused opens than a receipt gets replays
    for _ in 0..2 * skippy_receipts::recorder::MAX_REPLAYS_PER_RECEIPT {
        for refused in [
            forwarded(&unproven, Some(root)),
            forwarded(&forged, Some(root)),
            forwarded(&proven, None),
            forwarded(&proven, Some([6; 32])),
        ] {
            assert!(stage1.open_forwarded(&refused).is_err());
        }
    }
    assert_eq!(probe.replays(), 0);

    let opening = stage1
        .open_forwarded(&forwarded(&proven, Some(root)))
        .expect("the client's challenge");
    assert_eq!(opening.stages[0].layers[0].layer, 3);
    assert_eq!(probe.replays(), 1);
    // the same challenge again is answered from the opening kept
    stage1
        .open_forwarded(&forwarded(&proven, Some(root)))
        .expect("repeat");
    assert_eq!(probe.replays(), 1);
    // one challenge per bound turn, on every stage
    let mut other = challenge("r1", &[2]);
    other.proof = proven.proof.clone();
    let error = stage1
        .open_forwarded(&forwarded(&other, Some(root)))
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("already challenged"),
        "{error:#}"
    );
}

// When the stage's own cache no longer holds a turn, it says that it cannot
// open it and why, which is not how it refuses a request.
#[test]
fn a_stage_that_lost_a_turns_cells_says_it_cannot_open() {
    let stage0 = stage(0, 0..4, None);
    let (_, challenge) = bound_turn(&stage0, "r1", &[1]);
    let recorder = stage0.reports.recorder();
    // the session ended and the stage could not pin its cells, then a
    // ubatch the recorder could not follow
    recorder.memory_op(MemoryOp::SeqRm, SEQ, -1, -1, -1);
    recorder.pins_not_applied();
    recorder.unreceiptable("two lanes in one recurrent ubatch");
    let error = stage0.open(&challenge, None).unwrap_err();
    let cannot = cannot_open(&error);
    assert_eq!(cannot.reason, CannotOpenReason::CellsUnknown);
    assert_eq!((cannot.receipt_id.as_str(), cannot.stage_index), ("r1", 0));

    // a turn that left the recorder's window
    let stage0 = stage(0, 0..4, None);
    let (_, challenge) = bound_turn(&stage0, "r1", &[1]);
    stage0.reports.recorder().discard("r1");
    let error = stage0.open(&challenge, None).unwrap_err();
    assert_eq!(cannot_open(&error).reason, CannotOpenReason::Expired);
}

// The stage owns its receipts: when it goes, the recorder state and the
// receipts threads go with it, whatever the model's recorder sink, a kept
// router or the API thread still point at.
#[test]
fn dropping_the_stage_drops_its_receipts() {
    use skippy_runtime::{ReceiptMemoryEvent, ReceiptRecorderSink};
    let receipts = Arc::new(stage(0, 0..4, None));
    let recorder = Arc::downgrade(&receipts.reports.shared_recorder());
    // what the native recorder holds for as long as the model lives
    let sink: Arc<dyn ReceiptRecorderSink> = Arc::new(super::native::RecorderSink::new(&receipts));
    receipts.start_upkeep().expect("upkeep");
    let listener =
        std::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("bind");
    let address = listener.local_addr().expect("address");
    let mut api = spawn_api(listener, router(api_state(&receipts))).expect("serve");
    let api_thread = api.take_thread();
    assert!(receipts.api.set(api).is_ok());
    let client_routes = super::http_api::client_routes(api_state(&receipts));
    let url = format!("http://{address}/receipts/r1");
    let reply = reqwest::blocking::get(&url).expect("reply");
    assert_eq!(reply.status(), reqwest::StatusCode::NOT_FOUND);

    drop(receipts);
    assert!(recorder.upgrade().is_none(), "the recorder state is gone");
    // the model may still report, into nothing
    sink.memory(&ReceiptMemoryEvent {
        op: 0,
        seq: -1,
        seq_dst: -1,
        p0: -1,
        p1: -1,
    });
    sink.unreceiptable("the stage is gone");
    drop(client_routes);
    // the API thread was told to stop when the receipts dropped
    api_thread.join().expect("the receipts API thread");
    assert!(std::net::TcpStream::connect(address).is_err());
}

#[path = "prefix_tests.rs"]
mod prefix_tests;

#[test]
fn a_turn_must_compute_the_end_of_its_prompt() {
    // a prompt of 10 tokens whose first 9 were cached: the turn computed
    // the last one and sampled its first token from it
    assert!(super::check_turn_start(10, &(9..14)).is_ok());
    // the whole prompt cached: no receipt covers where the first token
    // came from
    let error = super::check_turn_start(10, &(10..14)).unwrap_err();
    assert!(
        format!("{error:#}").contains("no receipt covers"),
        "{error:#}"
    );
    // a cached prefix longer than the prompt
    assert!(super::check_turn_start(8, &(10..14)).is_err());
    // a turn that fed nothing past its prompt and sampled only its last
    // token is a turn of no positions, which no stage commits
    assert!(super::check_turn_start(10, &(10..10)).is_ok());
}

/// A client that names its receipt id (`x-receipt-id`) gets its receipt
/// under it; else under the request's id, as before.
#[test]
fn a_receipt_is_issued_under_the_id_the_client_asked_for() {
    let asked = uuid::Uuid::parse_str("0b0e7c1e-58a5-4b3f-9c36-0d2f6f1f8a10").unwrap();
    let request = [9; 16];
    assert_eq!(
        super::issued_id(Some(asked), Some(request), 5),
        "0b0e7c1e-58a5-4b3f-9c36-0d2f6f1f8a10"
    );
    assert_eq!(
        super::issued_id(None, Some(request), 5),
        uuid::Uuid::from_bytes(request).hyphenated().to_string()
    );
    assert_eq!(super::issued_id(None, None, 5), "0000000000000005");
}

/// A hybrid stage gives a turn that continues a cached prefix no receipt,
/// so an embedder that owes one for every turn reads that it keeps
/// recurrent state. A runtime without receipts keeps none for them.
#[test]
fn a_runtime_says_whether_its_receipts_keep_recurrent_state() {
    let runtime = Arc::new(Mutex::new(
        crate::runtime_state::RuntimeState::new_modelless_for_test(1),
    ));
    assert!(!super::keeps_recurrent_state(&runtime));
    let receipts = Arc::new(StageReceipts::for_test(0));
    lock(&runtime).set_receipts(receipts.clone());
    assert!(!super::keeps_recurrent_state(&runtime));
    receipts.reports.recorder().set_recurrent(true);
    assert!(super::keeps_recurrent_state(&runtime));
}
