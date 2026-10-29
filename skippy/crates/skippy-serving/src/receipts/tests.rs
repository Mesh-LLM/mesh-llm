use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use anyhow::Result;
use skippy_receipts::audit::verify_paths;
use skippy_receipts::layer_tensors::{
    GGML_F32, TensorColumn, TensorKind, UbatchColumns, f32_bytes,
};
use skippy_receipts::recorder::{ReplayStep, ReplayedLayer, Replayer, UbatchRecord};
use skippy_receipts::wire::Challenge;

use super::{ApiState, CommitRequest, SessionSequences, StageReceipts, spawn_api};

const SESSION: &str = "session-a";
const SEQ: i32 = 3;
const POSITIONS: std::ops::Range<u32> = 4..10;

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
        })
    }
}

struct SyntheticReplayer;

impl Replayer for SyntheticReplayer {
    fn replay(
        &self,
        seq: i32,
        layer: u32,
        steps: &[ReplayStep],
    ) -> Result<Vec<(u32, ReplayedLayer)>> {
        let mut out = Vec::new();
        for step in steps {
            if let ReplayStep::Decode { record, capture } = step {
                for col in 0..record.pos.len() {
                    let position = u32::try_from(record.pos[col])?;
                    if record.seq[col] == seq && capture.contains(&position) {
                        let columns = SyntheticColumns::new(record);
                        out.push((position, ReplayedLayer::from_columns(layer, &columns, col)?));
                    }
                }
            }
        }
        Ok(out)
    }
}

fn stage(index: u32, layers: std::ops::Range<u32>, downstream: Option<String>) -> StageReceipts {
    let receipts = StageReceipts::new(
        index,
        layers,
        "test".into(),
        downstream,
        "model.gguf".into(),
    )
    .expect("stage receipts");
    receipts.set_replayer(Box::new(SyntheticReplayer));
    let n = POSITIONS.len();
    let record = UbatchRecord {
        token: vec![1; n],
        seq: vec![SEQ; n],
        pos: POSITIONS.map(|p| p as i32).collect(),
        output: vec![1; n],
        cell: POSITIONS.collect(),
        n_kv: 256,
        input: None,
    };
    let columns = SyntheticColumns::new(&record);
    receipts
        .reports
        .recorder()
        .record_ubatch(record.clone(), &columns);
    receipts
}

/// Serves `receipts` on an ephemeral loopback port and returns its URL.
fn serve(receipts: StageReceipts) -> String {
    let listener =
        std::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("address"));
    spawn_api(
        listener,
        ApiState {
            receipts: Arc::new(receipts),
            sessions: Arc::new(FixedSessions(BTreeMap::from([(SESSION.into(), SEQ)]))),
        },
    )
    .expect("serve");
    url
}

/// A two-stage chain: stage 0 holds layers 0..2 in process, stage 1 holds
/// layers 2..4 behind its receipts API.
fn chain() -> (StageReceipts, String) {
    let downstream = serve(stage(1, 2..4, None));
    (stage(0, 0..2, Some(downstream.clone())), downstream)
}

fn challenge(receipt_id: &str, layer: u32) -> Challenge {
    Challenge {
        receipt_id: receipt_id.into(),
        layer,
        positions: vec![4, 7, 9],
    }
}

#[test]
fn finish_turn_commits_every_stage() {
    let (stage0, _) = chain();
    let receipt = stage0.finish_turn("r1", SESSION, SEQ).expect("receipt");

    assert_eq!(receipt.model, "model.gguf");
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
fn opens_local_and_downstream_layers() {
    let (stage0, _) = chain();
    let receipt = stage0.finish_turn("r1", SESSION, SEQ).expect("receipt");

    for layer in [1, 3] {
        let challenge = challenge("r1", layer);
        let opening = stage0.open(&challenge).expect("opening");
        assert_eq!(opening.stage_index, u32::from(layer >= 2));
        verify_paths(&receipt, &challenge, &opening).expect("paths verify");
    }
}

#[test]
fn rejects_layers_and_receipts_nobody_holds() {
    let (stage0, _) = chain();
    stage0.finish_turn("r1", SESSION, SEQ).expect("receipt");

    assert!(stage0.open(&challenge("r1", 9)).is_err());
    assert!(stage0.open(&challenge("missing", 3)).is_err());
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
        })
        .send()
        .expect("reply");

    assert_eq!(reply.status(), reqwest::StatusCode::BAD_REQUEST);
    assert!(reply.text().expect("body").contains("no native sequence"));
}

#[test]
fn refuses_a_remote_bind_unless_allowed() {
    let mut options = super::ReceiptsOptions {
        listen: SocketAddr::from((Ipv4Addr::UNSPECIFIED, 9501)),
        allow_remote: false,
        downstream: None,
        claimed_model: "model.gguf".into(),
    };
    assert!(super::check_listen(&options).is_err());
    options.allow_remote = true;
    assert!(super::check_listen(&options).is_ok());
    options.allow_remote = false;
    options.listen = SocketAddr::from((Ipv4Addr::LOCALHOST, 9501));
    assert!(super::check_listen(&options).is_ok());
}

#[test]
fn a_turn_without_a_receipt_keeps_the_reason() {
    let (stage0, _) = chain();
    assert!(stage0.finish_turn("r2", SESSION, SEQ + 1).is_err());
    let reason = stage0.receipt("r2").expect("kept").expect_err("no receipt");
    assert!(reason.contains("recorded nothing"), "{reason}");
}
