use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const DOMAIN: &[u8] = b"mesh-probe-range-v1\0";
pub const SAMPLE_LEN: u32 = 64;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Intent {
    Process,
    Executable,
    Model,
}

impl Intent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Process => "process",
            Self::Executable => "executable",
            Self::Model => "model",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Challenge {
    pub id: String,
    pub peer: String,
    pub round: u8,
    pub nonce: String,
    pub offset: u64,
    pub length: u32,
    pub intents: Vec<Intent>,
    #[serde(default)]
    pub process_binding: Option<ProcessBinding>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProcessBinding {
    pub pid: u64,
    pub start_time: u64,
    pub executable_handle: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Reference {
    pub executable_digest: Option<String>,
    pub model_metadata: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProbeInput {
    pub challenge: Challenge,
    pub question: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolRecord {
    pub tool: Intent,
    pub supported: bool,
    pub source: String,
    pub data: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProbeResponse {
    pub challenge_id: String,
    pub peer: String,
    pub nonce: String,
    pub answer: String,
    pub status: ProbeStatus,
    pub trace: Vec<ToolRecord>,
    #[serde(default)]
    pub model_actions: Vec<String>,
    pub error: Option<String>,
    pub wasm_executed: bool,
    pub model_calls: u32,
    pub tool_calls: u32,
    pub elapsed_ms: u128,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeStatus {
    Completed,
    Unavailable,
    InvalidModelOutput,
    Failed,
}

pub fn range_digest(bytes: &[u8], offset: u64, nonce: &str) -> Result<String> {
    let nonce = hex::decode(nonce)?;
    ensure!(nonce.len() == 32, "nonce must be 32 bytes");
    let length = u32::try_from(bytes.len())?;
    let mut hash = Sha256::new();
    hash.update(DOMAIN);
    hash.update(nonce);
    hash.update(offset.to_be_bytes());
    hash.update(length.to_be_bytes());
    hash.update(bytes);
    Ok(hex::encode(hash.finalize()))
}

fn seeded_nonce(seed: u64, peer: &str, round: u8) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mesh-probe-seed-v2\0");
    hash.update(seed.to_be_bytes());
    hash.update(peer.as_bytes());
    hash.update([round]);
    hex::encode(hash.finalize())
}

pub fn selected_follow_up(seed: u64) -> Intent {
    if seed % 3 == 1 {
        Intent::Model
    } else {
        Intent::Executable
    }
}

pub fn select(seed: u64, peer: &str, artifact: &[u8]) -> Result<Challenge> {
    ensure!(artifact.len() >= SAMPLE_LEN as usize, "artifact too small");
    let nonce = seeded_nonce(seed, peer, 1);
    let choice = hex::decode(&nonce)?;
    let span = artifact.len() - SAMPLE_LEN as usize + 1;
    let offset = (u64::from_be_bytes(choice[..8].try_into()?) % span as u64) as usize;
    Ok(Challenge {
        id: format!("probe-{}", &nonce[..12]),
        peer: peer.into(),
        round: 1,
        nonce,
        offset: offset as u64,
        length: SAMPLE_LEN,
        intents: vec![Intent::Process],
        process_binding: None,
    })
}

pub fn process_binding(response: &ProbeResponse) -> Option<ProcessBinding> {
    let record = response
        .trace
        .iter()
        .find(|record| record.tool == Intent::Process && record.supported)?;
    Some(ProcessBinding {
        pid: record.data["pid"].as_u64()?,
        start_time: record.data["start_time"].as_u64()?,
        executable_handle: record.data["executable_handle"].as_str()?.to_owned(),
    })
}

pub fn follow_up(
    seed: u64,
    first: &Challenge,
    first_response: &ProbeResponse,
    artifact: &[u8],
) -> Result<(Challenge, Reference)> {
    ensure!(first.round == 1, "follow-up requires round one");
    let binding = process_binding(first_response);
    let intent = if binding.is_some() {
        selected_follow_up(seed)
    } else {
        Intent::Model
    };
    let nonce = seeded_nonce(seed, &first.peer, 2);
    let challenge = Challenge {
        id: format!("{}-2", first.id),
        peer: first.peer.clone(),
        round: 2,
        nonce,
        offset: first.offset,
        length: first.length,
        intents: vec![intent],
        process_binding: binding,
    };
    let reference = reference_for(&challenge, artifact)?;
    Ok((challenge, reference))
}

pub fn bundled(seed: u64, peer: &str, artifact: &[u8]) -> Result<(Challenge, Reference)> {
    let mut challenge = select(seed, peer, artifact)?;
    challenge.nonce = seeded_nonce(seed, peer, 2);
    challenge.intents.push(selected_follow_up(seed));
    let reference = reference_for(&challenge, artifact)?;
    Ok((challenge, reference))
}

fn reference_for(challenge: &Challenge, artifact: &[u8]) -> Result<Reference> {
    let executable_digest = if challenge.intents.contains(&Intent::Executable) {
        let start = usize::try_from(challenge.offset)?;
        let end = start
            .checked_add(challenge.length as usize)
            .ok_or_else(|| anyhow::anyhow!("range overflow"))?;
        ensure!(end <= artifact.len(), "reference range outside artifact");
        Some(range_digest(
            &artifact[start..end],
            challenge.offset,
            &challenge.nonce,
        )?)
    } else {
        None
    };
    Ok(Reference {
        executable_digest,
        model_metadata: None,
    })
}

pub fn validate(challenge: &Challenge) -> Result<()> {
    ensure!(
        !challenge.id.is_empty() && challenge.id.len() <= 80,
        "bad challenge id"
    );
    ensure!(
        !challenge.peer.is_empty() && challenge.peer.len() <= 128,
        "bad peer"
    );
    ensure!(hex::decode(&challenge.nonce)?.len() == 32, "bad nonce");
    ensure!(
        challenge.length > 0 && challenge.length <= SAMPLE_LEN,
        "bad range length"
    );
    ensure!((1..=2).contains(&challenge.round), "bad round");
    let valid_intents = match challenge.round {
        1 => {
            challenge.intents == [Intent::Process]
                || challenge.intents == [Intent::Process, Intent::Executable]
                || challenge.intents == [Intent::Process, Intent::Model]
        }
        2 => challenge.intents == [Intent::Executable] || challenge.intents == [Intent::Model],
        _ => false,
    };
    ensure!(valid_intents, "bad round intents");
    if challenge.round == 1 {
        ensure!(
            challenge.process_binding.is_none(),
            "round one has prior binding"
        );
    }
    if challenge.round == 2 && challenge.intents == [Intent::Executable] {
        ensure!(
            challenge.process_binding.is_some(),
            "executable follow-up needs process binding"
        );
    }
    let mut unique = challenge.intents.clone();
    unique.sort_by_key(|intent| intent.as_str());
    unique.dedup();
    ensure!(unique.len() == challenge.intents.len(), "duplicate intent");
    Ok(())
}
