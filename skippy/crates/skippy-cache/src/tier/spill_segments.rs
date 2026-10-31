//! Preserve component-aware disk segment boundaries without concatenating payloads.
use super::CacheGenKvPayload;
use crate::l3::{PayloadGeometry, SegmentCodecIdentity};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SegmentRepresentation {
    Raw,
    NativeKvPage,
    CacheGenKv {
        decoded_len: u64,
        calibration_digest: String,
    },
}

impl SegmentRepresentation {
    pub(super) fn identity(&self, encoded_len: u64) -> SegmentCodecIdentity {
        match self {
            Self::Raw => SegmentCodecIdentity::raw(encoded_len),
            Self::NativeKvPage => SegmentCodecIdentity::native_kv_page(encoded_len),
            Self::CacheGenKv {
                decoded_len,
                calibration_digest,
            } => SegmentCodecIdentity::cachegen_archive(*decoded_len, calibration_digest.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SegmentCut {
    pub(super) offset: u64,
    pub(super) len: u64,
    pub(super) label: String,
    pub(super) representation: SegmentRepresentation,
}

fn append_fixed_cuts(
    cuts: &mut Vec<SegmentCut>,
    offset: &mut u64,
    bytes: u64,
    segment_bytes: u64,
    representation: SegmentRepresentation,
) {
    let mut remaining = bytes;
    while remaining > 0 {
        let len = segment_bytes.max(1).min(remaining);
        cuts.push(SegmentCut {
            offset: *offset,
            len,
            label: String::new(),
            representation: representation.clone(),
        });
        *offset = offset.saturating_add(len);
        remaining -= len;
    }
}

pub(super) struct SpillSegmentPlan<'a> {
    pub(super) segment_bytes: u64,
    pub(super) kv_bytes: u64,
    pub(super) recurrent_bytes: u64,
    pub(super) native_kv: bool,
    pub(super) cachegen: Option<&'a CacheGenKvPayload>,
}

impl SpillSegmentPlan<'_> {
    pub(super) fn cuts(&self, geometry: Option<&PayloadGeometry>) -> Vec<SegmentCut> {
        match geometry {
            Some(geometry) => geometry
                .plan(self.segment_bytes)
                .into_iter()
                .map(|(offset, len, label)| SegmentCut {
                    offset,
                    len,
                    representation: if self.native_kv && offset < self.kv_bytes {
                        SegmentRepresentation::NativeKvPage
                    } else {
                        SegmentRepresentation::Raw
                    },
                    label,
                })
                .collect(),
            None => {
                let mut cuts = Vec::new();
                let mut offset = 0u64;
                if let Some(cachegen) = self.cachegen {
                    cuts.push(SegmentCut {
                        offset: 0,
                        len: self.kv_bytes,
                        representation: SegmentRepresentation::CacheGenKv {
                            decoded_len: cachegen.decoded_len,
                            calibration_digest: cachegen.calibration_digest.clone(),
                        },
                        label: "cachegen-kv-archive".to_string(),
                    });
                    offset = self.kv_bytes;
                    append_fixed_cuts(
                        &mut cuts,
                        &mut offset,
                        self.recurrent_bytes,
                        self.segment_bytes,
                        SegmentRepresentation::Raw,
                    );
                } else if self.native_kv {
                    append_fixed_cuts(
                        &mut cuts,
                        &mut offset,
                        self.kv_bytes,
                        self.segment_bytes,
                        SegmentRepresentation::NativeKvPage,
                    );
                    append_fixed_cuts(
                        &mut cuts,
                        &mut offset,
                        self.recurrent_bytes,
                        self.segment_bytes,
                        SegmentRepresentation::Raw,
                    );
                } else {
                    append_fixed_cuts(
                        &mut cuts,
                        &mut offset,
                        self.kv_bytes.saturating_add(self.recurrent_bytes),
                        self.segment_bytes,
                        SegmentRepresentation::Raw,
                    );
                }
                cuts
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;
    use crate::payload::ExactStatePayload;
    use crate::tier::spill_payload::SpillPayload;

    #[test]
    fn raw_segments_cross_component_boundary_with_only_one_bounded_copy() {
        let payload = ExactStatePayload::kv_recurrent(vec![1, 2, 3], vec![4, 5, 6]);
        let view = SpillPayload::new(&payload, None, None).unwrap();
        let cuts = SpillSegmentPlan {
            segment_bytes: 4,
            kv_bytes: 3,
            recurrent_bytes: 3,
            native_kv: false,
            cachegen: None,
        }
        .cuts(None);
        let segments = cuts
            .iter()
            .map(|cut| view.segment(cut.offset, cut.len, 4).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(segments[0].as_ref(), &[1, 2, 3, 4]);
        assert_eq!(segments[1].as_ref(), &[5, 6]);
        assert!(matches!(segments[0], Cow::Owned(_)));
        assert!(matches!(segments[1], Cow::Borrowed(_)));
    }

    #[test]
    fn native_kv_cut_boundary_never_materializes_a_composite_segment() {
        let payload = ExactStatePayload::kv_recurrent(vec![1, 2, 3], vec![4, 5, 6]);
        let view = SpillPayload::new(&payload, None, None).unwrap();
        let cuts = SpillSegmentPlan {
            segment_bytes: 4,
            kv_bytes: 3,
            recurrent_bytes: 3,
            native_kv: true,
            cachegen: None,
        }
        .cuts(None);
        assert_eq!(cuts.len(), 2);
        assert_eq!(cuts[0].representation, SegmentRepresentation::NativeKvPage);
        assert_eq!(cuts[1].representation, SegmentRepresentation::Raw);
        for cut in cuts {
            assert!(matches!(
                view.segment(cut.offset, cut.len, 4).unwrap(),
                Cow::Borrowed(_)
            ));
        }
    }
}
