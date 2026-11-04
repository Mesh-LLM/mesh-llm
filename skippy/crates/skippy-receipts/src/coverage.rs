//! Whether this verifier can audit a receipt at all.
//!
//! An audit needs the node's runtime, a rounding profile for every weight
//! type each stage multiplies on its backend, and a reference decoding of
//! the output head. When one is missing the checks would fail an honest
//! node, so the receipt is not audited instead: an audit that cannot run
//! says nothing about the node (`mesh/docs/design/RECEIPTS.md`).
//!
//! A node can choose a runtime or backend no verifier covers and so never
//! be audited. Its receipts then count as unaudited turns, which a client's
//! trust policy can weigh, and never as passes.

use crate::audit::{BackendFamily, profiles_for};
use crate::head::head_weight;
use crate::weights::{WeightFile, type_name};
use crate::wire::Receipt;

/// Why this verifier, which runs `runtime`, cannot audit `receipt`, if it
/// cannot.
pub fn unauditable(receipt: &Receipt, weights: &WeightFile, runtime: &str) -> Option<String> {
    if let Some(binding) = &receipt.binding
        && binding.runtime != runtime
    {
        return Some(format!(
            "the node ran {}, and this verifier runs {runtime}",
            binding.runtime
        ));
    }
    for stage in &receipt.stages {
        let Some(family) = BackendFamily::of(&stage.backend) else {
            return Some(format!(
                "no rounding profiles for backend {} yet",
                stage.backend
            ));
        };
        let layers = stage.layer_start..stage.layer_end;
        let uncovered = weights.tensors.values().find(|t| {
            t.is_matrix()
                && t.layer().is_some_and(|l| layers.contains(&l))
                && profiles_for(family, t.ggml_type).is_none()
        });
        if let Some(tensor) = uncovered {
            return Some(format!(
                "no rounding profile for {} weights ({}) on {family:?}",
                type_name(tensor.ggml_type),
                tensor.name
            ));
        }
    }
    match weights.tensors.get(head_weight(weights)) {
        Some(tensor) if !crate::dequant::supports(tensor.ggml_type) => Some(format!(
            "no reference decoding of {} output weights",
            type_name(tensor.ggml_type)
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::{GGML_F32, GGML_Q4_1, GGML_Q8_0};
    use crate::turn::TurnBinding;
    use crate::weights::test_gguf;
    use crate::wire::{RECEIPT_VERSION, StageCommitment};

    const RUNTIME: &str = "skippy-abi 1.2.3";

    fn receipt(backend: &str, runtime: Option<&str>) -> Receipt {
        Receipt {
            version: RECEIPT_VERSION,
            receipt_id: "r".into(),
            model: "m".into(),
            position_start: 0,
            position_end: 2,
            tokens: vec![1, 2],
            generation_start: 1,
            final_token: None,
            stages: vec![StageCommitment {
                stage_index: 0,
                layer_start: 0,
                layer_end: 1,
                root: "00".into(),
                leaf_count: 1,
                backend: backend.into(),
                cache_types: Vec::new(),
                prefix: None,
                ubatches: Vec::new(),
                recurrent_state: None,
            }],
            binding: runtime.map(|runtime| TurnBinding {
                client_key: String::new(),
                parent: None,
                runtime: runtime.into(),
                sampler: crate::turn::GREEDY.into(),
                node_key: String::new(),
                turn_root: String::new(),
                signature: String::new(),
            }),
        }
    }

    /// Only the tensor types matter here, so the tensors are written as f32
    /// and given their types after. Layer 1, which no stage of the receipt
    /// holds, has a type without profiles.
    fn weights(layer_type: u32, head_type: u32) -> WeightFile {
        let names = [
            "blk.0.attn_q.weight",
            "blk.1.attn_q.weight",
            "output.weight",
        ];
        let tensors: Vec<_> = names
            .iter()
            .map(|name| (*name, vec![32, 4], vec![0.0; 128]))
            .collect();
        let mut weights = test_gguf::open(&[], &tensors);
        for (name, ggml_type) in names.iter().zip([layer_type, GGML_Q4_1, head_type]) {
            weights.tensors.get_mut(*name).unwrap().ggml_type = ggml_type;
        }
        weights
    }

    #[test]
    fn a_covered_receipt_is_auditable() {
        let weights = weights(GGML_Q8_0, GGML_Q8_0);
        let bound = receipt("CUDA0", Some(RUNTIME));
        assert_eq!(unauditable(&bound, &weights, RUNTIME), None);
        assert_eq!(unauditable(&receipt("CPU", None), &weights, RUNTIME), None);
    }

    #[test]
    fn what_this_verifier_cannot_check_is_not_audited() {
        let covered = weights(GGML_Q8_0, GGML_F32);
        let other = receipt("CPU", Some("skippy-abi 1.2.4"));
        assert!(unauditable(&other, &covered, RUNTIME).is_some());
        let rocm = receipt("ROCm0", None);
        assert!(unauditable(&rocm, &covered, RUNTIME).is_some());
        let q4_1 = weights(GGML_Q4_1, GGML_F32);
        let reason = unauditable(&receipt("CUDA0", None), &q4_1, RUNTIME).unwrap();
        assert!(reason.contains("blk.0.attn_q.weight"), "{reason}");
        let head = weights(GGML_Q8_0, 19);
        assert!(unauditable(&receipt("CPU", None), &head, RUNTIME).is_some());
    }
}
