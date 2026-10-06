//! Bounded numeric scheduler observations; capture/phase ownership is separate from parsing.
use serde::Serialize;
use serde_json::Value;
#[derive(Clone, Serialize)]
#[serde(tag = "_event")]
pub(super) enum Scheduler {
    #[serde(rename = "stage.scheduler_feature_iteration")]
    Feature {
        #[serde(rename = "skippy.scheduler.token_count")]
        token_count: u64,
    },
    #[serde(rename = "stage.scheduler_iteration")]
    Detailed {
        #[serde(rename = "skippy.scheduler.prefill_tokens")]
        prefill: u64,
        #[serde(rename = "skippy.scheduler.recompute_tokens")]
        recompute: u64,
        #[serde(rename = "skippy.scheduler.decode_tokens")]
        decode: u64,
    },
}
impl Scheduler {
    pub fn tokens(&self) -> Result<u64, &'static str> {
        match self {
            Self::Feature { token_count } => Ok(*token_count),
            Self::Detailed {
                prefill,
                recompute,
                decode,
            } => prefill
                .checked_add(*recompute)
                .and_then(|n| n.checked_add(*decode))
                .ok_or("scheduler token total overflow"),
        }
    }
    pub fn detailed(&self) -> Option<(u64, u64, u64)> {
        match self {
            Self::Detailed {
                prefill,
                recompute,
                decode,
            } => Some((*prefill, *recompute, *decode)),
            Self::Feature { .. } => None,
        }
    }
}
#[derive(Clone, Serialize)]
pub(super) struct Prefill {
    pub token_count: u64,
    pub chunks: u64,
    pub maximum: u64,
    pub bottleneck_stage: Option<u64>,
}
#[derive(Default)]
pub(super) struct Projection {
    scheduler: Vec<Scheduler>,
    prefills: Vec<Prefill>,
    error: Option<String>,
}
pub(super) struct WarmupBoundary {
    scheduler: usize,
    prefills: usize,
}
#[derive(Serialize)]
pub(super) struct Measured {
    pub scheduler: Vec<Scheduler>,
    pub prefills: Vec<Prefill>,
    pub request_sha256: String,
    pub phase_provenance: &'static str,
    pub prefill_role_provenance: &'static str,
}
fn integer(attrs: &Value, key: &str) -> Result<u64, &'static str> {
    attrs[key]
        .as_u64()
        .ok_or("required mixed numeric counter absent")
}
impl Projection {
    pub fn into_parts(self) -> Result<(Vec<Scheduler>, Vec<Prefill>), &'static str> {
        if self.error.is_some() {
            return Err("mixed typed numeric projection refused");
        }
        Ok((self.scheduler, self.prefills))
    }
    pub fn observe(&mut self, bytes: &[u8]) {
        if self.error.is_some() {
            return;
        }
        if let Err(e) = self.line(bytes) {
            self.error = Some(e.into());
        }
    }
    fn line(&mut self, bytes: &[u8]) -> Result<(), &'static str> {
        if bytes.len() > 8192 {
            return Err("mixed typed telemetry line exceeds owning capture bound");
        }
        let Ok(value) = serde_json::from_slice::<Value>(bytes) else {
            return Ok(());
        };
        let attrs = &value["attributes"];
        if self.scheduler.len() + self.prefills.len() >= 65536 {
            return Err("mixed telemetry row bound exhausted");
        }
        match value["event"].as_str() {
            Some("stage.scheduler_feature_iteration") => self.scheduler.push(Scheduler::Feature {
                token_count: integer(attrs, "skippy.scheduler.token_count")?,
            }),
            Some("stage.scheduler_iteration") => {
                for key in [
                    "skippy.scheduler.failed",
                    "skippy.scheduler.cancelled",
                    "skippy.scheduler.rejected_overload",
                ] {
                    if let Some(v) = attrs.get(key)
                        && v.as_u64() != Some(0)
                    {
                        return Err("mixed scheduler failed/cancelled/rejected evidence");
                    }
                }
                self.scheduler.push(Scheduler::Detailed {
                    prefill: integer(attrs, "skippy.scheduler.prefill_tokens")?,
                    recompute: integer(attrs, "skippy.scheduler.recompute_tokens")?,
                    decode: integer(attrs, "skippy.scheduler.decode_tokens")?,
                });
            }
            Some("stage.openai_prefill") => {
                for key in [
                    "skippy.kv.chain_cache_errors",
                    "skippy.kv.stage0_cache_errors",
                ] {
                    if attrs[key].as_u64() != Some(0) {
                        return Err("mixed prefill cache-error evidence absent or nonzero");
                    }
                }
                let tokens = integer(attrs, "llama_stage.prefill_token_count")?;
                let chunks = integer(attrs, "llama_stage.prefill_chunk_count")?;
                let maximum = integer(attrs, "llama_stage.prefill_max_chunk_size")?;
                if tokens == 0 || chunks == 0 || maximum == 0 {
                    return Err("mixed prefill counter invalid");
                }
                let stage = attrs
                    .get("llama_stage.prefill_bottleneck_stage_index")
                    .map(|v| v.as_u64().ok_or("mixed bottleneck stage invalid"))
                    .transpose()?;
                self.prefills.push(Prefill {
                    token_count: tokens,
                    chunks,
                    maximum,
                    bottleneck_stage: stage,
                });
            }
            _ => {}
        }
        Ok(())
    }
    /// Only the owning coordinator may snapshot after its verified warmup barrier.
    /// A log parser alone cannot establish this phase boundary.
    pub fn warmup_boundary(&self) -> Result<WarmupBoundary, String> {
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        if self.prefills.len() != 1 {
            return Err("warmup prefill completion evidence missing or ambiguous".into());
        }
        Ok(WarmupBoundary {
            scheduler: self.scheduler.len(),
            prefills: self.prefills.len(),
        })
    }
    pub fn measured(
        &self,
        boundary: &WarmupBoundary,
        requests: usize,
        request_sha256: &str,
        capture_complete: bool,
    ) -> Result<Measured, String> {
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        if !capture_complete
            || requests == 0
            || requests > 16
            || self.prefills.len().checked_sub(boundary.prefills) != Some(requests)
            || boundary.scheduler >= self.scheduler.len()
            || request_sha256.len() != 64
            || !request_sha256.bytes().all(|c| c.is_ascii_hexdigit())
        {
            return Err("mixed phase/capture, scheduler or prefill evidence incomplete".into());
        }
        let scheduler = self.scheduler[boundary.scheduler..].to_vec();
        for row in &scheduler {
            row.tokens()?;
        }
        Ok(Measured {
            scheduler,
            prefills: self.prefills[boundary.prefills..].to_vec(),
            request_sha256: request_sha256.into(),
            phase_provenance: "owner-snapshotted-verified-warmup-barrier-required",
            prefill_role_provenance: "token-count-ranked-original-heuristic-not-request-ID-correlation",
        })
    }
}
