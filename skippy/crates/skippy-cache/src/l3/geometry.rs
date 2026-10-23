//! Payload geometry: how a manifest's segments are cut.
//!
//! Cutting on token-row windows rather than byte offsets is what lets a longer
//! prefix reuse a shorter one's segments.

/// How a payload is laid out, so segments can be cut where a growing prefix
/// keeps its bytes still.
///
/// The runtime exports exact state as a sequence of runs, each holding one
/// token-row per token in token order: every layer's K, then every layer's V,
/// then the indexer rows. Adding tokens extends every run, so a cut at a fixed
/// byte offset lands in a different place each turn and nothing dedupes — the
/// measured cost was 8x the newly committed bytes at 8 MiB segments.
///
/// Cutting each run into fixed windows of token-rows instead means turn N+1's
/// segments are byte-identical to turn N's up to the last partial window, and
/// only genuinely new state is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadGeometry {
    /// Runs in wire order.
    pub blocks: Vec<GeometryBlock>,
    /// Token-rows in every run.
    pub rows: u64,
    /// Rows per segment. Must depend only on the model's own shape, never on
    /// how many tokens this particular entry holds, or the boundaries move
    /// between turns and the dedupe is lost.
    pub window_rows: u64,
    /// Trailing bytes with no row structure (a recurrent snapshot), cut at the
    /// store's default segment size.
    pub tail_bytes: u64,
}

/// One run of token-rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryBlock {
    /// Bytes per token-row.
    pub stride: u64,
    /// What this run holds, for the segment metadata: `k`, `v` or `kidx`.
    pub kind: GeometryKind,
    /// Which layer, relative to the exported range.
    pub layer: u32,
    /// Sub-run within the layer. Always 0 except for transposed V, where each
    /// embedding column is its own token-contiguous run.
    pub column: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryKind {
    Key,
    Value,
    KeyIndex,
}

impl GeometryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Key => "k",
            Self::Value => "v",
            Self::KeyIndex => "kidx",
        }
    }
}

impl PayloadGeometry {
    /// Bytes the geometry claims, for checking it against the payload it is
    /// meant to describe.
    pub fn total_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .map(|block| block.stride.saturating_mul(self.rows))
            .fold(0u64, u64::saturating_add)
            .saturating_add(self.tail_bytes)
    }

    /// A geometry that does not describe this payload exactly is not usable:
    /// cutting to it would produce segments that reassemble to different
    /// bytes. Callers fall back to fixed-size cutting.
    pub fn matches(&self, payload_bytes: u64) -> bool {
        self.rows > 0
            && self.window_rows > 0
            && !self.blocks.is_empty()
            && self.blocks.iter().all(|block| block.stride > 0)
            && self.total_bytes() == payload_bytes
    }

    /// The cuts, as `(offset, len, label)` in wire order.
    pub fn plan(&self, tail_segment_bytes: u64) -> Vec<(u64, u64, String)> {
        let mut cuts = Vec::new();
        let mut offset = 0u64;
        for block in &self.blocks {
            let mut row = 0u64;
            while row < self.rows {
                let rows = self.window_rows.min(self.rows - row);
                let len = rows.saturating_mul(block.stride);
                cuts.push((
                    offset,
                    len,
                    format!(
                        "{}:{}:{}:{row}",
                        block.kind.as_str(),
                        block.layer,
                        block.column
                    ),
                ));
                offset = offset.saturating_add(len);
                row += rows;
            }
        }
        let tail_cut = tail_segment_bytes.max(1);
        let mut remaining = self.tail_bytes;
        while remaining > 0 {
            let len = tail_cut.min(remaining);
            cuts.push((offset, len, "tail".to_string()));
            offset = offset.saturating_add(len);
            remaining -= len;
        }
        cuts
    }
}
