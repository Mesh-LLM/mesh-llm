//! Frozen deterministic conversation; every executed prefix ends on its user turn.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
const VOCAB: &str = "cache prefix token restore restart segment manifest budget eviction prefill decode latency throughput checkpoint durable radix tier admission pipeline stream verify digest commit quarantine pin lease reserve node mesh relay model runtime kernel attention matrix layer head batch queue trace replay harness baseline cohort percentile regression gate promote";
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Turn {
    context: String,
    request: String,
    response: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub schema_version: u64,
    pub kind: String,
    pub seed: u64,
    pub settings: Value,
    pub system: String,
    pub turns: Vec<Turn>,
}
struct Random(u64);
impl Random {
    fn words(&mut self, count: usize) -> String {
        let vocab = VOCAB.split_whitespace().collect::<Vec<_>>();
        (0..count)
            .map(|_| {
                self.0 = self.0.wrapping_mul(25214903917).wrapping_add(11) & 0x0000_ffff_ffff_ffff;
                vocab[((self.0 >> 16) % vocab.len() as u64) as usize]
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
    fn block(&mut self, target: u32, topic: &str) -> String {
        let words = (u64::from(target) * 3 / 4).max(1);
        let chunks = words.div_ceil(8);
        let text = (0..chunks)
            .map(|_| self.words(8))
            .collect::<Vec<_>>()
            .join(" ");
        format!("[{topic}] {text}")
    }
}
pub(super) fn build(turns: u32, target: u32, system: u32) -> DynResult<Manifest> {
    if !(1..=128).contains(&turns)
        || !(1..=65536).contains(&target)
        || !(1..=65536).contains(&system)
        || u64::from(turns) * u64::from(target) + u64::from(system) > 1_000_000
    {
        return Err("restart conversation exceeds bounded turn/token scaffold".into());
    }
    let mut rng = Random(20260909);
    let scaffold = rng.block(system, "scaffold");
    let mut specs = Vec::new();
    for index in 0..turns {
        let context = rng.block(target, &format!("turn-{}-context", index + 1));
        let request = format!(
            "Turn {}: given the project brief above, summarize the {} constraint in one sentence and list the {} next step.",
            index + 1,
            rng.words(6),
            rng.words(4)
        );
        let response = format!(
            "Turn {} answer: preserve the {} constraint. Next step: verify {}.",
            index + 1,
            rng.words(5),
            rng.words(4)
        );
        specs.push(Turn {
            context,
            request,
            response,
        });
    }
    Ok(Manifest {
        schema_version: 1,
        kind: "kv-restart-replay/manifest".into(),
        seed: 20260909,
        settings: json!({"turns":turns,"turn_target_tokens":target,"system_tokens":system,"approx_total_prompt_tokens":u64::from(system)+u64::from(turns)*u64::from(target)}),
        system: scaffold,
        turns: specs,
    })
}
impl Manifest {
    pub fn validate(&self) -> DynResult<()> {
        let turns = u32::try_from(
            self.settings["turns"]
                .as_u64()
                .ok_or("manifest turn count absent")?,
        )?;
        let target = u32::try_from(
            self.settings["turn_target_tokens"]
                .as_u64()
                .ok_or("manifest target absent")?,
        )?;
        let system = u32::try_from(
            self.settings["system_tokens"]
                .as_u64()
                .ok_or("manifest scaffold absent")?,
        )?;
        if serde_json::to_value(self)? != serde_json::to_value(build(turns, target, system)?)? {
            return Err("restart manifest differs from deterministic frozen settings".into());
        }
        Ok(())
    }
    pub fn messages(&self, index: usize) -> DynResult<Vec<Value>> {
        if index >= self.turns.len() {
            return Err("restart turn outside manifest".into());
        }
        let mut messages = vec![json!({"role":"system","content":self.system})];
        for (turn, spec) in self.turns.iter().enumerate().take(index + 1) {
            messages.push(
                json!({"role":"user","content":format!("{}\n\n{}",spec.context,spec.request)}),
            );
            if turn < index {
                messages.push(json!({"role":"assistant","content":spec.response}));
            }
        }
        Ok(messages)
    }
}
