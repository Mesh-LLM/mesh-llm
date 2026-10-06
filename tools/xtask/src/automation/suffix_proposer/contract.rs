use crate::command::DynResult;
use serde::{Deserialize, Serialize};
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Arm {
    pub name: String,
    pub base_url: String,
    pub declared_stages: u32,
    pub declared_mtp_capable: bool,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Workload {
    pub name: String,
    pub prompt: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub arms: Vec<Arm>,
    pub model: String,
    #[serde(default)]
    pub corpus: Option<std::path::PathBuf>,
    #[serde(default = "warmups")]
    pub warmups: u32,
    #[serde(default = "runs")]
    pub runs: u32,
    #[serde(default = "tokens")]
    pub max_tokens: u64,
    #[serde(default = "timeout")]
    pub request_timeout_ms: u64,
    pub execution_timeout_ms: u64,
    #[serde(default = "seed")]
    pub seed: u64,
    #[serde(default = "baseline")]
    pub baseline_arm: String,
    #[serde(default = "required")]
    pub require_drafts_arm: Option<String>,
}
fn warmups() -> u32 {
    2
}
fn runs() -> u32 {
    5
}
fn tokens() -> u64 {
    256
}
fn timeout() -> u64 {
    180000
}
fn seed() -> u64 {
    20260721
}
fn baseline() -> String {
    "off".into()
}
fn required() -> Option<String> {
    Some("suffix".into())
}
fn label(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        let mut names = std::collections::BTreeSet::new();
        if self.schema_version != 1
            || self.arms.is_empty()
            || self.arms.len() > 16
            || self.runs == 0
            || self.runs > 100
            || self.warmups > 100
            || self.max_tokens == 0
            || self.max_tokens > 4096
            || self.request_timeout_ms == 0
            || self.request_timeout_ms > 3600000
            || self.execution_timeout_ms < 1
            || self.execution_timeout_ms > 86400000
            || self.model.is_empty()
            || self.model.len() > 4096
            || self.model.chars().any(char::is_control)
        {
            return Err("invalid bounded suffix input".into());
        }
        for arm in &self.arms {
            let uri: hyper::Uri = arm.base_url.parse()?;
            if !label(&arm.name)
                || !names.insert(&arm.name)
                || !matches!(uri.scheme_str(), Some("http" | "https"))
                || uri.host().is_none()
                || uri.authority().is_none_or(|a| a.as_str().contains('@'))
                || uri.query().is_some()
                || uri.path() != "/"
                || arm.declared_stages < 2
                || !arm.declared_mtp_capable
            {
                return Err("invalid arm URL/name or declared MTP split".into());
            }
        }
        if !names.contains(&self.baseline_arm)
            || self
                .require_drafts_arm
                .as_ref()
                .is_some_and(|n| !names.contains(n))
        {
            return Err("baseline/required arm absent".into());
        }
        Ok(())
    }
}
pub(super) fn workloads(input: &Input) -> DynResult<Vec<Workload>> {
    let edit = "def parse_config(path):\n    with open(path) as handle:\n        raw = handle.read()\n    data = json.loads(raw)\n    result = {}\n    for key, value in data.items():\n        if isinstance(value, str):\n            result[key] = value.strip()\n        else:\n            result[key] = value\n    return result";
    let mut rows=vec![Workload{name:"edit".into(),prompt:format!("Here is a Python function:\n\n```python\n{edit}\n```\n\nRe-emit the entire function verbatim, changing only the name `parse_config` to `load_config`. Output just the code.")},Workload{name:"chat".into(),prompt:"Explain, in two short paragraphs, why generating a token is more expensive than verifying one in speculative decoding.".into()}];
    let corpus = input.corpus.clone().or_else(|| {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../evals/skippy-coding-agent-loop.jsonl");
        p.exists().then_some(p)
    });
    if let Some(path) = corpus {
        let bytes = super::evidence::read(&path, 1048576)?;
        let text = std::str::from_utf8(&bytes)?;
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)?;
            let prompt = v["prompt"]
                .as_str()
                .ok_or("corpus lacks prompt")?
                .to_owned();
            let name = v["id"]
                .as_str()
                .filter(|v| !v.is_empty())
                .map_or_else(|| format!("corpus-{}", i + 1), str::to_owned);
            rows.push(Workload { name, prompt });
        }
    }
    let mut names = std::collections::BTreeSet::new();
    if rows
        .len()
        .saturating_mul(input.arms.len())
        .saturating_mul((input.runs + input.warmups) as usize)
        > 4096
        || rows.len() > 256
        || rows.iter().any(|w| {
            !label(&w.name)
                || !names.insert(&w.name)
                || w.prompt.is_empty()
                || w.prompt.len() > 65536
        })
    {
        return Err("invalid duplicate/bounded suffix corpus".into());
    }
    Ok(rows)
}
