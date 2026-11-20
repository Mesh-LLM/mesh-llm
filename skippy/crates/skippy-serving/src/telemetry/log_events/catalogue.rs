//! The events skippy defines. Each type fixes its event name, its default
//! severity and its attributes, as OpenTelemetry's event conventions
//! require: one name, one structure. `skippy/docs/TELEMETRY_EVENTS.md`
//! describes each event and when it fires.

use std::collections::BTreeMap;

use serde_json::{Value, json};
use skippy_runtime::native_tracing::gpu_commands::AttentionPath;

use super::{Event, Severity};

/// A tier of the KV prefix cache: L1 is the resident radix cache, L2 the
/// host-RAM mirror of L3 fills, L3 the durable store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvTier {
    L1,
    L2,
    L3,
}

impl KvTier {
    fn name(self) -> &'static str {
        match self {
            Self::L1 => "l1",
            Self::L2 => "l2",
            Self::L3 => "l3",
        }
    }
}

fn attrs<const N: usize>(pairs: [(&str, Value); N]) -> BTreeMap<String, Value> {
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

/// `skippy.kv.lookup`: a prefix lookup at one tier found reusable state or
/// did not. A hit carries how many prompt tokens it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvLookup {
    pub(crate) tier: KvTier,
    /// `Some(tokens)` for a hit; `None` for a miss.
    pub(crate) matched_tokens: Option<u64>,
}

impl Event for KvLookup {
    const NAME: &'static str = "skippy.kv.lookup";
    const SEVERITY: Severity = Severity::Info;

    fn attributes(&self) -> BTreeMap<String, Value> {
        let mut attributes = attrs([
            ("skippy.kv.tier", json!(self.tier.name())),
            (
                "skippy.kv.result",
                json!(if self.matched_tokens.is_some() {
                    "hit"
                } else {
                    "miss"
                }),
            ),
        ]);
        if let Some(tokens) = self.matched_tokens {
            attributes.insert("skippy.kv.matched_tokens".to_string(), json!(tokens));
        }
        attributes
    }
}

/// `skippy.kv.eviction`: a tier released cached prefixes to make room.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvEviction {
    pub(crate) tier: KvTier,
    pub(crate) entries: u64,
    pub(crate) tokens: u64,
}

impl Event for KvEviction {
    const NAME: &'static str = "skippy.kv.eviction";
    const SEVERITY: Severity = Severity::Info;

    fn attributes(&self) -> BTreeMap<String, Value> {
        attrs([
            ("skippy.kv.tier", json!(self.tier.name())),
            ("skippy.kv.evicted_entries", json!(self.entries)),
            ("skippy.kv.evicted_tokens", json!(self.tokens)),
        ])
    }
}

/// `skippy.kv.admission_shortfall`: a request was admitted below the KV
/// pool's free-space watermark, after every releasable prefix was evicted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvAdmissionShortfall {
    pub(crate) deficit_tokens: u64,
}

impl Event for KvAdmissionShortfall {
    const NAME: &'static str = "skippy.kv.admission_shortfall";
    const SEVERITY: Severity = Severity::Warn;

    fn attributes(&self) -> BTreeMap<String, Value> {
        attrs([("skippy.kv.shortfall_tokens", json!(self.deficit_tokens))])
    }
}

/// Native work that stalls serving when it happens after the model is
/// ready: each event names one kind, with what it worked on and how long
/// it took.
macro_rules! native_stall_event {
    ($(#[$doc:meta])* $type:ident, $name:literal) => {
        $(#[$doc])*
        #[derive(Clone, Debug, PartialEq)]
        pub(crate) struct $type {
            pub(crate) detail: Option<String>,
            pub(crate) duration_ms: f64,
        }

        impl Event for $type {
            const NAME: &'static str = $name;
            const SEVERITY: Severity = Severity::Warn;

            fn attributes(&self) -> BTreeMap<String, Value> {
                let mut attributes = attrs([("skippy.native.duration_ms", json!(self.duration_ms))]);
                if let Some(detail) = &self.detail {
                    attributes.insert("skippy.native.detail".to_string(), json!(detail));
                }
                attributes
            }
        }
    };
}

native_stall_event!(
    /// `skippy.graph.reserved`: the backend scheduler reserved compute
    /// buffers again while serving (`llama.sched_reserve`), which blocks
    /// the next graph.
    GraphReserved,
    "skippy.graph.reserved"
);
native_stall_event!(
    /// `skippy.gpu.pipeline_compiled`: a GPU kernel pipeline was compiled
    /// on first use while serving (`mtl.pipeline_compile`).
    PipelineCompiled,
    "skippy.gpu.pipeline_compiled"
);
native_stall_event!(
    /// `skippy.gpu.buffer_allocated`: a device buffer was allocated while
    /// serving (`mtl.buffer_alloc`), such as a grown compute buffer.
    BufferAllocated,
    "skippy.gpu.buffer_allocated"
);

/// `skippy.attention.path`: how attention reads the KV cache, when first
/// seen on a backend and whenever it changes: flash attention over each
/// sequence's row list, flash attention over the whole masked K/V span, or
/// unfused attention over the whole span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttentionPathSeen {
    pub(crate) backend: &'static str,
    pub(crate) path: AttentionPath,
    pub(crate) kv_cells: u32,
    pub(crate) queries: u32,
}

impl Event for AttentionPathSeen {
    const NAME: &'static str = "skippy.attention.path";
    const SEVERITY: Severity = Severity::Info;

    fn attributes(&self) -> BTreeMap<String, Value> {
        attrs([
            ("skippy.gpu.backend", json!(self.backend)),
            ("skippy.attention.path", json!(self.path.name())),
            ("skippy.attention.kv_cells", json!(self.kv_cells)),
            ("skippy.attention.queries", json!(self.queries)),
        ])
    }
}

/// `skippy.gpu.op_profiling`: a backend splits graphs at op class
/// boundaries, so command buffers time each class alone. Totals are
/// inflated; shares are what such a trace measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GpuOpProfiling {
    pub(crate) backend: &'static str,
}

impl Event for GpuOpProfiling {
    const NAME: &'static str = "skippy.gpu.op_profiling";
    const SEVERITY: Severity = Severity::Warn;

    fn attributes(&self) -> BTreeMap<String, Value> {
        attrs([("skippy.gpu.backend", json!(self.backend))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lookup_says_its_tier_result_and_coverage() {
        let hit = KvLookup {
            tier: KvTier::L3,
            matched_tokens: Some(512),
        }
        .attributes();
        assert_eq!(hit["skippy.kv.tier"], json!("l3"));
        assert_eq!(hit["skippy.kv.result"], json!("hit"));
        assert_eq!(hit["skippy.kv.matched_tokens"], json!(512));
        let miss = KvLookup {
            tier: KvTier::L1,
            matched_tokens: None,
        }
        .attributes();
        assert_eq!(miss["skippy.kv.result"], json!("miss"));
        assert!(!miss.contains_key("skippy.kv.matched_tokens"));
    }

    #[test]
    fn native_stalls_carry_their_detail_when_there_is_one() {
        let event = GraphReserved {
            detail: Some("ubatch 4".to_string()),
            duration_ms: 250.5,
        };
        assert_eq!(GraphReserved::NAME, "skippy.graph.reserved");
        assert_eq!(
            event.attributes()["skippy.native.detail"],
            json!("ubatch 4")
        );
        let bare = PipelineCompiled {
            detail: None,
            duration_ms: 1.0,
        };
        assert!(!bare.attributes().contains_key("skippy.native.detail"));
    }
}
